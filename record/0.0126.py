## -*- coding: utf-8 -*-
"""
练习1_fixed.py
修复版本：LightGCN + Feature MultiHeadAttention + Seq Transformer + HardNegative + 时间特征 + 贝叶斯优化初始化 + L2正则化
数据目录（请确认）: /Users/rchen/Desktop/个人赛题结果
文件: user.csv, item.csv, inter_preliminary.csv
输出: submission.csv
"""
import os
import random
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
import csv
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization  # 添加贝叶斯优化库

# ---------------- CONFIG ----------------
USERS_CSV = os.path.join("user.csv")
ITEMS_CSV = os.path.join("item.csv")
INTER_CSV = os.path.join("inter_preliminary.csv")
OUT_CSV = os.path.join("submission.csv")

ID_EMB_DIM = 128
FEAT_EMB_DIM = 128  # 必须等于 ID_EMB_DIM
NUM_LAYERS = 3
EPOCHS = 50
BATCH_SIZE = 2048
LR = 0.00005
WEIGHT_DECAY = 1e-5
L2_REG = 1e-7  # L2正则化系数
VAL_HIT_K = 10
EARLY_STOP_PATIENCE = 10
SEED = 42
MAX_SEQ_LEN = 50
HARD_NEG_PROB = 0.5
ALPHA_POP = 0.6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Ensure dims match
if ID_EMB_DIM != FEAT_EMB_DIM:
    print(f"Warning: ID_EMB_DIM ({ID_EMB_DIM}) != FEAT_EMB_DIM ({FEAT_EMB_DIM}), setting FEAT_EMB_DIM = ID_EMB_DIM")
    FEAT_EMB_DIM = ID_EMB_DIM


# --------------- utils -------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything()


def parse_time(x):
    if pd.isna(x) or x == "":
        return 0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return int(datetime.strptime(str(x), fmt).timestamp())
        except Exception:
            continue
    try:
        return int(float(x))
    except Exception:
        return 0


# ------------- data load & preprocess -------------
def load_and_process():
    print("Loading CSVs...")
    users = pd.read_csv(USERS_CSV, dtype=str).fillna("")
    items = pd.read_csv(ITEMS_CSV, dtype=str).fillna("")
    inter = pd.read_csv(INTER_CSV, dtype=str).fillna("")

    # normalize column names
    if "借阅人" not in users.columns and "user_id" in users.columns:
        users = users.rename(columns={"user_id": "借阅人"})
    if "book_id" not in items.columns and "图书ID" in items.columns:
        items = items.rename(columns={"图书ID": "book_id"})
    if "user_id" not in inter.columns and "借阅人" in inter.columns:
        inter = inter.rename(columns={"借阅人": "user_id"})

    # ensure key columns
    if "借阅人" not in users.columns:
        raise RuntimeError("user.csv must contain column '借阅人' or 'user_id'")
    if "book_id" not in items.columns:
        raise RuntimeError("item.csv must contain column 'book_id'")
    if "user_id" not in inter.columns or "book_id" not in inter.columns:
        raise RuntimeError("inter_preliminary.csv must contain 'user_id' and 'book_id'")

    # 添加时间相关特征处理
    print("Processing time-related features...")

    # 解析时间字段
    if "借阅时间" in inter.columns:
        inter["borrow_ts"] = inter["借阅时间"].apply(parse_time)
    else:
        inter["borrow_ts"] = 0

    if "还书时间" in inter.columns:
        inter["return_ts"] = inter["还书时间"].apply(parse_time)
    else:
        inter["return_ts"] = 0

    if "续借时间" in inter.columns:
        inter["renew_ts"] = inter["续借时间"].apply(parse_time)
    else:
        inter["renew_ts"] = 0

    if "续借次数" in inter.columns:
        # 将续借次数转换为数值
        inter["renew_count"] = pd.to_numeric(inter["续借次数"], errors='coerce').fillna(0).astype(int)
    else:
        inter["renew_count"] = 0

    # 计算借阅时长（如果没有还书时间，可以使用当前时间或固定值）
    inter["borrow_duration"] = inter["return_ts"] - inter["borrow_ts"]
    # 处理异常值
    inter.loc[inter["borrow_duration"] < 0, "borrow_duration"] = 0
    inter.loc[inter["borrow_duration"] > 365 * 24 * 3600, "borrow_duration"] = 365 * 24 * 3600

    # 解析借阅时间戳
    inter["time_ts"] = inter.get("借阅时间", "").apply(parse_time) if "借阅时间" in inter.columns else 0
    inter["time_ts"] = inter["time_ts"].astype(int)

    # feature columns
    user_feat_cols = ["性别", "DEPT", "年级", "类型"]
    item_feat_cols = ["一级分类", "二级分类", "作者", "出版社"]
    # 添加时间相关特征
    time_feat_cols = ["borrow_duration", "renew_count"]

    for c in user_feat_cols:
        if c not in users.columns:
            users[c] = ""
    for c in item_feat_cols:
        if c not in items.columns:
            items[c] = ""

    # factorize categorical columns (stable integer ids)
    print("Factorizing user/item categorical features...")
    for c in user_feat_cols:
        users[c], _ = pd.factorize(users[c].astype(str))
    for c in item_feat_cols:
        items[c], _ = pd.factorize(items[c].astype(str))

    # 只保留在交互数据中出现过的用户
    inter_users = inter["user_id"].astype(str).unique()
    users = users[users["借阅人"].astype(str).isin(inter_users)].copy()

    # build unique lists & mappings (use users.csv and items.csv canonical order)
    users_unique = users["借阅人"].astype(str).unique().tolist()
    items_unique = items["book_id"].astype(str).unique().tolist()
    uid2idx = {u: i for i, u in enumerate(users_unique)}
    iid2idx = {v: i for i, v in enumerate(items_unique)}
    print("num users (from users.csv with interactions):", len(uid2idx), "num items (from item.csv):", len(iid2idx))

    # filter interactions to those present in mappings
    inter = inter[inter["user_id"].astype(str).isin(uid2idx) & inter["book_id"].astype(str).isin(iid2idx)].copy()
    if inter.shape[0] == 0:
        raise RuntimeError("No interactions remain after filtering by users/items. Check CSVs and IDs.")

    inter["uidx"] = inter["user_id"].astype(str).map(uid2idx)
    inter["iidx"] = inter["book_id"].astype(str).map(iid2idx)
    inter["uidx"] = inter["uidx"].astype(int)
    inter["iidx"] = inter["iidx"].astype(int)

    inter = inter.sort_values(["uidx", "time_ts"]).reset_index(drop=True)

    # popularity
    num_items = len(iid2idx)
    item_pop = inter["iidx"].value_counts().reindex(range(num_items), fill_value=0).values

    # item category mapping (by iidx)
    item_cat_by_iidx = [""] * num_items
    if "一级分类" in items.columns:
        for _, r in items.iterrows():
            bid = str(r["book_id"])
            if bid in iid2idx:
                item_cat_by_iidx[iid2idx[bid]] = r.get("一级分类", "")
    # build cat2items
    cat2items = defaultdict(list)
    for iidx, cat in enumerate(item_cat_by_iidx):
        cat2items[cat].append(iidx)

    # user sequences
    user_seq = inter.groupby("uidx")["iidx"].apply(list).to_dict()

    # 为每个用户计算平均时间特征
    print("Calculating user time features...")
    user_time_features = inter.groupby("uidx").agg({
        "borrow_duration": "mean",
        "renew_count": "mean"
    }).reset_index()

    # 填充缺失值
    user_time_features = user_time_features.fillna(0)

    # 标准化时间特征
    scaler = StandardScaler()
    time_feats_scaled = scaler.fit_transform(user_time_features[time_feat_cols])
    user_time_features[time_feat_cols] = time_feats_scaled

    # 将时间特征合并到用户数据中
    users = users.merge(user_time_features, left_index=True, right_on="uidx", how="left")
    users[time_feat_cols] = users[time_feat_cols].fillna(0)

    return users, items, inter, uid2idx, iid2idx, item_pop, cat2items, user_seq, user_feat_cols, item_feat_cols, time_feat_cols


# --------------- Model components ----------------
class FeatureAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        # we'll use batch_first=False for nn.MultiheadAttention; input shape (seq_len, batch, embed)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=False)

    def forward(self, emb_list):
        # emb_list: list of (batch, embed)
        stacked = torch.stack(emb_list, dim=0)  # (n_feats, batch, embed)
        out, _ = self.mha(stacked, stacked, stacked)
        out = out.mean(dim=0)  # (batch, embed)
        return out


class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_items, user_feat_dims, item_feat_dims, time_feat_dim,
                 id_emb_dim=ID_EMB_DIM, feat_emb_dim=FEAT_EMB_DIM, num_layers=NUM_LAYERS, init_std=0.1, l2_reg=L2_REG):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_emb = nn.Embedding(num_users, id_emb_dim)
        self.item_emb = nn.Embedding(num_items, id_emb_dim)
        self.l2_reg = l2_reg  # 存储L2正则化系数

        # feature embeddings
        self.user_feat_embs = nn.ModuleList([nn.Embedding(d, feat_emb_dim) for d in user_feat_dims])
        self.item_feat_embs = nn.ModuleList([nn.Embedding(d, feat_emb_dim) for d in item_feat_dims])

        # 添加时间特征处理
        self.time_emb = nn.Linear(time_feat_dim, feat_emb_dim)  # 将时间特征映射到嵌入空间

        self.user_attn = FeatureAttention(feat_emb_dim, n_heads=4)
        self.item_attn = FeatureAttention(feat_emb_dim, n_heads=4)

        # sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_emb_dim, nhead=4, batch_first=True)
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.num_layers = num_layers
        self.init_std = init_std  # 存储初始化标准差
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, std=self.init_std)
        nn.init.normal_(self.item_emb.weight, std=self.init_std)
        for emb in self.user_feat_embs:
            nn.init.normal_(emb.weight, std=self.init_std)
        for emb in self.item_feat_embs:
            nn.init.normal_(emb.weight, std=self.init_std)
        # 初始化时间特征线性层
        nn.init.normal_(self.time_emb.weight, std=self.init_std)
        nn.init.constant_(self.time_emb.bias, 0)

    def l2_loss(self):
        # 计算所有可训练参数的L2正则化
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, 2) ** 2
        return self.l2_reg * l2_loss

    def combine_feats(self, emb_list, attn):
        return attn(emb_list)  # emb_list: list of (batch, dim) -> out (batch, dim)

    def propagate(self, adj, user_init, item_init):
        u, v = user_init, item_init
        all_u, all_v = [u], [v]
        for _ in range(self.num_layers):
            x = torch.cat([u, v], dim=0)
            x = torch.sparse.mm(adj, x)
            u, v = torch.split(x, [u.size(0), v.size(0)], dim=0)
            all_u.append(u)
            all_v.append(v)
        u_final = torch.mean(torch.stack(all_u, dim=0), dim=0)
        v_final = torch.mean(torch.stack(all_v, dim=0), dim=0)
        return u_final, v_final

    def forward(self, adj, user_feat_idx, item_feat_idx, time_feats=None, seq_tensor=None, seq_mask=None):
        # user_feat_idx: (num_users, n_user_feats)
        # item_feat_idx: (num_items, n_item_feats)
        num_users = user_feat_idx.size(0)
        num_items = item_feat_idx.size(0)

        # base id embs
        u_id = self.user_emb.weight  # (num_users, dim)
        i_id = self.item_emb.weight  # (num_items, dim)

        # user feature embeddings
        u_feat_embs = [emb(user_feat_idx[:, j]) for j, emb in enumerate(self.user_feat_embs)]

        # 添加时间特征
        if time_feats is not None:
            time_emb = self.time_emb(time_feats)  # (num_users, dim)
            u_feat_embs.append(time_emb)  # 将时间特征添加到特征列表中

        u_feat = self.combine_feats(u_feat_embs, self.user_attn)  # (num_users, dim)

        i_feat_embs = [emb(item_feat_idx[:, j]) for j, emb in enumerate(self.item_feat_embs)]
        i_feat = self.combine_feats(i_feat_embs, self.item_attn)  # (num_items, dim)

        user_init = u_id + u_feat
        item_init = i_id + i_feat

        # propagate
        user_gcn, item_gcn = self.propagate(adj, user_init, item_init)

        # sequence part: seq_tensor contains item indices (-1 for pad)
        if seq_tensor is not None:
            # create padded item embedding with a zero row at index 0; map real items -> +1
            dim = item_gcn.size(1)
            zero = torch.zeros(1, dim, device=item_gcn.device)
            item_emb_padded = torch.cat([zero, item_gcn], dim=0)  # shape (num_items+1, dim)
            seq_idx = (seq_tensor + 1).clamp(min=0)  # -1 -> 0
            seq_item_emb = item_emb_padded[seq_idx]  # (num_users, seq_len, dim)

            # mask: seq_mask True means valid; src_key_padding_mask expects True for masked positions
            src_key_padding_mask = None
            if seq_mask is not None:
                src_key_padding_mask = ~seq_mask  # invert: True to mask

            # encode
            seq_encoded = self.seq_encoder(seq_item_emb, src_key_padding_mask=src_key_padding_mask)

            # aggregate by valid counts
            if seq_mask is not None:
                counts = seq_mask.sum(dim=1, keepdim=True).clamp(min=1).to(seq_encoded.dtype)
                seq_sum = (seq_encoded * seq_mask.unsqueeze(-1)).sum(dim=1)
                seq_vec = seq_sum / counts
            else:
                seq_vec = seq_encoded.mean(dim=1)

            user_final = user_gcn + seq_vec
        else:
            user_final = user_gcn

        return user_final, item_gcn


# --------------- graph builder ------------------
def build_adj_sparse(num_users, num_items, inter_df, u_col="uidx", i_col="iidx"):
    rows = inter_df[u_col].astype(int).values
    cols = inter_df[i_col].astype(int).values + num_users
    data = np.ones(len(rows), dtype=np.float32)
    N = num_users + num_items
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
    mat = mat + mat.transpose()
    deg = np.array(mat.sum(axis=1)).flatten()
    deg[deg == 0] = 1e-7
    d_inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt.dot(mat).dot(D_inv_sqrt).tocoo()
    indices = np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
    values = norm_adj.data.astype(np.float32)
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = torch.Size(norm_adj.shape)
    sp_adj = torch.sparse.FloatTensor(i, v, shape).to(DEVICE)
    return sp_adj


# --------------- negative sampler ----------------
class NegativeSampler:
    def __init__(self, item_pop, cat2items, alpha=ALPHA_POP, hard_prob=HARD_NEG_PROB):
        self.num_items = len(item_pop)
        pop = np.array(item_pop, dtype=np.float64) + 1.0
        pop = np.power(pop, alpha)
        self.pop_prob = pop / pop.sum()
        self.cat2items = cat2items
        self.hard_prob = hard_prob

    def sample(self, pos_item, user_pos_set):
        if random.random() < self.hard_prob:
            cat = None
            # try find a category that contains pos_item
            for cat_k, items in self.cat2items.items():
                if pos_item in items:
                    cat = cat_k
                    break
            if cat is not None:
                candidates = [c for c in self.cat2items.get(cat, []) if c not in user_pos_set and c != pos_item]
                if candidates:
                    return random.choice(candidates)
        # fallback/pop sample
        while True:
            neg = int(np.random.choice(self.num_items, p=self.pop_prob))
            if neg not in user_pos_set:
                return neg


# --------------- eval/train ---------------------
def evaluate(model, adj, user_feat_idx, item_feat_idx, val_users, val_pos_items,
             K=10, time_feats=None, seq_tensor=None, seq_mask=None):
    model.eval()
    with torch.no_grad():
        u_emb, i_emb = model(adj, user_feat_idx, item_feat_idx, time_feats, seq_tensor, seq_mask)
        scores = torch.matmul(u_emb, i_emb.t())
        hits = 0
        f1 = 0.0
        for u in val_users:
            pos = val_pos_items[u]
            sc = scores[u].cpu().numpy()
            topk = np.argsort(-sc)[:K]
            if pos in topk:
                hits += 1
                f1 += 2.0 / (K + 1)
        hitk = hits / max(1, len(val_users))
        f1k = f1 / max(1, len(val_users))
    return hitk, f1k


# 贝叶斯优化目标函数
def bayesian_optimization_target(init_std, lr=LR, weight_decay=WEIGHT_DECAY):
    # 设置随机种子以确保结果可重复
    seed_everything(SEED)

    # 使用给定的超参数创建模型
    model = RecommenderModel(
        num_users, num_items, user_feat_dims, item_feat_dims, len(time_feat_cols),
        id_emb_dim=ID_EMB_DIM, feat_emb_dim=FEAT_EMB_DIM, num_layers=NUM_LAYERS,
        init_std=init_std
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 简化的训练循环（只训练几个epoch以评估初始化效果）
    best_hit = 0
    for epoch in range(1, 5):  # 只训练少量epoch以快速评估
        model.train()
        random.shuffle(user_list)
        total_loss = 0.0
        cnt = 0
        for i in range(0, len(user_list), BATCH_SIZE):
            batch_users = user_list[i:i + BATCH_SIZE]
            u_batch = []
            pos_batch = []
            neg_batch = []
            for u in batch_users:
                pos_candidates = train_user_pos[u]
                if not pos_candidates:
                    continue
                pos = random.choice(pos_candidates)
                neg = neg_sampler.sample(pos, train_user_pos_sets[u])
                u_batch.append(u)
                pos_batch.append(pos)
                neg_batch.append(neg)
            if not u_batch:
                continue
            u_b = torch.LongTensor(u_batch).to(DEVICE)
            pos_b = torch.LongTensor(pos_batch).to(DEVICE)
            neg_b = torch.LongTensor(neg_batch).to(DEVICE)

            user_rep, item_rep = model(adj, user_feat_idx, item_feat_idx, time_feat_tensor, seq_tensor, seq_mask)
            u_emb = user_rep[u_b]
            pos_emb = item_rep[pos_b]
            neg_emb = item_rep[neg_b]
            pos_score = (u_emb * pos_emb).sum(dim=1)
            neg_score = (u_emb * neg_emb).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            # 添加L2正则化损失
            l2_loss = model.l2_loss()
            total_loss = loss + l2_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # 在验证集上评估
        hitk, f1k = evaluate(model, adj, user_feat_idx, item_feat_idx, val_users, val_pos_items,
                             K=VAL_HIT_K, time_feats=time_feat_tensor, seq_tensor=seq_tensor, seq_mask=seq_mask)
        if hitk > best_hit:
            best_hit = hitk

    return best_hit


def train_loop():
    global users, items, inter, uid2idx, iid2idx, item_pop, cat2items, user_seq, user_feat_cols, item_feat_cols, time_feat_cols
    global num_users, num_items, adj, user_feat_idx, item_feat_idx, time_feat_tensor, seq_tensor, seq_mask
    global user_feat_dims, item_feat_dims, user_groups, val_users, val_pos_items, train_user_pos, train_user_pos_sets, user_list, neg_sampler

    users, items, inter, uid2idx, iid2idx, item_pop, cat2items, user_seq, user_feat_cols, item_feat_cols, time_feat_cols = load_and_process()
    num_users = len(uid2idx)
    num_items = len(iid2idx)
    print("num_users, num_items:", num_users, num_items)

    adj = build_adj_sparse(num_users, num_items, inter, u_col="uidx", i_col="iidx")

    user_feat_idx = torch.LongTensor(users[user_feat_cols].values.astype(int)).to(DEVICE)
    item_feat_idx = torch.LongTensor(items[item_feat_cols].values.astype(int)).to(DEVICE)

    # 准备时间特征张量
    time_feat_tensor = torch.FloatTensor(users[time_feat_cols].values.astype(float)).to(DEVICE)

    # build seq tensors
    seq_tensor = torch.full((num_users, MAX_SEQ_LEN), -1, dtype=torch.long)
    seq_mask = torch.zeros((num_users, MAX_SEQ_LEN), dtype=torch.bool)
    for u, seq in user_seq.items():
        seq_last = seq[-MAX_SEQ_LEN:]
        start = MAX_SEQ_LEN - len(seq_last)
        if len(seq_last) > 0:
            seq_tensor[u, start:] = torch.LongTensor(seq_last)
            seq_mask[u, start:] = True
    seq_tensor = seq_tensor.to(DEVICE)
    seq_mask = seq_mask.to(DEVICE)

    user_feat_dims = [int(users[c].max()) + 1 for c in user_feat_cols]
    item_feat_dims = [int(items[c].max()) + 1 for c in item_feat_cols]

    # train/val split: leave last per user as val if >=2
    user_groups = inter.groupby("uidx")["iidx"].apply(list).to_dict()
    val_users = []
    val_pos_items = {}
    train_user_pos = defaultdict(list)
    for u, seq in user_groups.items():
        if len(seq) >= 2:
            val_users.append(u)
            val_pos_items[u] = seq[-1]
            train_user_pos[u] = seq[:-1]
        else:
            train_user_pos[u] = seq
    train_user_pos_sets = {u: set(v) for u, v in train_user_pos.items()}

    user_list = list(train_user_pos.keys())
    neg_sampler = NegativeSampler(item_pop, cat2items, alpha=ALPHA_POP, hard_prob=HARD_NEG_PROB)

    # 贝叶斯优化寻找最佳初始化参数
    print("Starting Bayesian optimization for initialization parameters...")
    optimizer = BayesianOptimization(
        f=bayesian_optimization_target,
        pbounds={'init_std': (0.01, 0.3)},  # 初始化标准差的范围
        random_state=SEED,
        verbose=2
    )

    optimizer.maximize(init_points=3, n_iter=7)  # 初始点和迭代次数

    # 获取最佳参数
    best_init_std = optimizer.max['params']['init_std']
    print(f"Best initialization std: {best_init_std}")

    # 使用最佳参数创建最终模型
    model = RecommenderModel(
        num_users, num_items, user_feat_dims, item_feat_dims, len(time_feat_cols),
        id_emb_dim=ID_EMB_DIM, feat_emb_dim=FEAT_EMB_DIM, num_layers=NUM_LAYERS,
        init_std=best_init_std, l2_reg=L2_REG
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_hit = 0.0
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(user_list)
        total_loss = 0.0
        cnt = 0
        for i in range(0, len(user_list), BATCH_SIZE):
            batch_users = user_list[i:i + BATCH_SIZE]
            u_batch = []
            pos_batch = []
            neg_batch = []
            for u in batch_users:
                pos_candidates = train_user_pos[u]
                if not pos_candidates:
                    continue
                pos = random.choice(pos_candidates)
                neg = neg_sampler.sample(pos, train_user_pos_sets[u])
                u_batch.append(u)
                pos_batch.append(pos)
                neg_batch.append(neg)
            if not u_batch:
                continue
            u_b = torch.LongTensor(u_batch).to(DEVICE)
            pos_b = torch.LongTensor(pos_batch).to(DEVICE)
            neg_b = torch.LongTensor(neg_batch).to(DEVICE)

            user_rep, item_rep = model(adj, user_feat_idx, item_feat_idx, time_feat_tensor, seq_tensor, seq_mask)
            u_emb = user_rep[u_b]
            pos_emb = item_rep[pos_b]
            neg_emb = item_rep[neg_b]
            pos_score = (u_emb * pos_emb).sum(dim=1)
            neg_score = (u_emb * neg_emb).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            # 添加L2正则化损失
            l2_loss = model.l2_loss()
            total_loss_batch = loss + l2_loss

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item() * len(u_b)
            cnt += len(u_b)
        avg_loss = total_loss / max(1, cnt)
        hitk, f1k = evaluate(model, adj, user_feat_idx, item_feat_idx, val_users, val_pos_items,
                             K=VAL_HIT_K, time_feats=time_feat_tensor, seq_tensor=seq_tensor, seq_mask=seq_mask)
        print(f"Epoch {epoch} loss={avg_loss:.6f} Hit@{VAL_HIT_K}={hitk:.6f} F1approx={f1k:.6f}")
        if hitk > best_hit:
            best_hit = hitk
            patience = 0
            torch.save(model.state_dict(), os.path.join("best_model_fixed.pth"))
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("Early stopping. Best hit:", best_hit)
                break

    # final predict & write submission
    model.load_state_dict(torch.load(os.path.join("best_model_fixed.pth"), map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        user_rep, item_rep = model(adj, user_feat_idx, item_feat_idx, time_feat_tensor, seq_tensor, seq_mask)
        scores = torch.matmul(user_rep, item_rep.t()).cpu().numpy()
        top1 = np.argmax(scores, axis=1)

    # 只保留在交互数据中出现过的用户
    inter_uids = inter["user_id"].astype(str).unique()

    # 创建反向映射
    idx2uid = {idx: uid for uid, idx in uid2idx.items()}
    idx2iid = {idx: iid for iid, idx in iid2idx.items()}

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "book_id"])
        for u_idx, item_idx in enumerate(top1):
            uid = idx2uid[u_idx]
            # 只输出在交互数据中出现过的用户
            if uid in inter_uids:
                iid = idx2iid[item_idx]
                writer.writerow([uid, iid])
    print("Submission written to", OUT_CSV)
    print(f"Total users with interactions: {len(inter_uids)}")


if __name__ == "__main__":
    train_loop()
