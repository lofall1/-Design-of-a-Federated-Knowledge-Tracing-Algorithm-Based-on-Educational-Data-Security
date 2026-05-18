# import pandas as pd
# import torch
# from torch.utils.data import Dataset, Subset

# class KTDataset(Dataset):
#     def __init__(self, file_path, seq_len=50):
#         df = pd.read_csv(file_path)

#         # 排序（关键）
#         df = df.sort_values(["user_id", "log_id"])

#         # ID映射（关键）
#         unique_q = df["sequence_id"].unique()
#         q2idx = {q:i for i, q in enumerate(unique_q)}
#         df["sequence_id"] = df["sequence_id"].map(q2idx)
#         df["correct"] = df["correct"].astype(int)
#         self.num_questions = len(q2idx)
#         grouped = df.groupby("user_id")

#         self.data = []

#         for _, group in grouped:
#             q = group["sequence_id"].values
#             a = group["correct"].values

#             for i in range(len(q) - seq_len):
#                 self.data.append((
#                     torch.LongTensor(q[i:i+seq_len].copy()),
#                     torch.FloatTensor(a[i:i+seq_len].copy())
#                 ))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# # IID划分
# def split_iid(dataset, num_clients=5):
#     size = len(dataset) // num_clients
#     clients = []
#     for i in range(num_clients):
#         idxs = list(range(i*size, (i+1)*size))
#         clients.append(Subset(dataset, idxs))
#     return clients


# # Non-IID（按学生划分）
# def split_noniid(dataset, num_clients=5):
#     clients = [[] for _ in range(num_clients)]

#     for i in range(len(dataset)):
#         clients[i % num_clients].append(i)

#     return [Subset(dataset, c) for c in clients]


import pandas as pd
import torch
from torch.utils.data import Dataset

# class KTDataset(Dataset):
#     def __init__(self, df, seq_len=50):
#         self.data = []

#         grouped = df.groupby("user_id")

#         for _, group in grouped:
#             q = group["sequence_id"].values
#             a = group["correct"].values

#             #  不滑窗，避免数据泄漏
#             if len(q) >= seq_len:
#                 self.data.append((
#                     torch.LongTensor(q[:seq_len]),
#                     torch.FloatTensor(a[:seq_len])
#                 ))
class KTDataset(Dataset):
    def __init__(self, df, seq_len=50):
        self.data = []

        grouped = df.groupby("user_id")

        for _, group in grouped:
            q = group["sequence_id"].values
            a = group["correct"].values
            if len(q) < 10:
                continue
            #  恢复滑窗（但只在train/test内部）
            for i in range(len(q) - seq_len):
                if i + seq_len <= len(q):
                    self.data.append((
                        torch.LongTensor(q[i:i+seq_len].copy()),
                        torch.FloatTensor(a[i:i+seq_len].copy())
                    ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(file_path):
    df = pd.read_csv(file_path)

    # 排序（必须）
    df = df.sort_values(["user_id", "log_id"])

    # 标签清洗
    df["correct"] = df["correct"].astype(int)

    # ID映射
    unique_q = df["sequence_id"].unique()
    q2idx = {q:i for i, q in enumerate(unique_q)}
    df["sequence_id"] = df["sequence_id"].map(q2idx)

    return df, len(q2idx)


#  按学生划分
def train_test_split(df, test_ratio=0.2):
    users = df["user_id"].unique()

    import numpy as np
    np.random.shuffle(users)

    split = int(len(users) * (1 - test_ratio))
    train_users = users[:split]
    test_users = users[split:]

    train_df = df[df["user_id"].isin(train_users)]
    test_df = df[df["user_id"].isin(test_users)]

    return train_df, test_df
