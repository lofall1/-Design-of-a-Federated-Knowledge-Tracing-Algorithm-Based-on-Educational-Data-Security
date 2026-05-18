import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from model.dataset import KTDataset, split_clients, split_clients_noniid
from model.dkt import DKT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for q, a in loader:
            q = q.to(device)
            a = a.to(device)

            out = model(q, a)
            out = out[:, :-1, :]
            target = a[:, 1:]
            q_next = q[:, 1:]

            pred = out.gather(2, q_next.unsqueeze(-1)).squeeze(-1)

            # ensure types: preds=float, targets=int
            preds.extend(pred.flatten().cpu().numpy().astype(float))
            targets.extend(target.flatten().cpu().numpy().astype(int))

    preds = np.array(preds, dtype=float)
    targets = np.array(targets, dtype=int)

    # Accuracy uses binarized predictions
    bin_preds = (preds >= 0.5).astype(int)
    acc = accuracy_score(targets, bin_preds)

    # AUC requires both classes present; handle degenerate cases
    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        auc = float('nan')

    # RMSE: root mean squared error between probability and binary target
    try:
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
    except Exception:
        rmse = float('nan')

    return acc, auc, rmse


def train_local(model, loader, device, local_epochs=1, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for _ in range(local_epochs):
        for q, a in loader:
            q = q.to(device)
            a = a.to(device)

            opt.zero_grad()
            out = model(q, a)
            out = out[:, :-1, :]
            target = a[:, 1:]
            q_next = q[:, 1:]
            pred = out.gather(2, q_next.unsqueeze(-1)).squeeze(-1)

            loss = criterion(pred, target)
            loss.backward()
            opt.step()

    return model.state_dict()


def fed_avg(global_model, client_states, client_sizes):
    """Weighted federated averaging by client sample counts."""
    total = sum(client_sizes) if client_sizes else len(client_states)
    new_state = {}
    g_state = global_model.state_dict()

    for key in g_state.keys():
        accum = None
        for state, size in zip(client_states, client_sizes):
            tensor = state[key].cpu().float()
            weighted = tensor * (size / total)
            if accum is None:
                accum = weighted
            else:
                accum = accum + weighted
        new_state[key] = accum

    global_model.load_state_dict(new_state)
    return global_model


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = os.path.join(SCRIPT_DIR, '2015_100_skill_builders_main_problems.csv')
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

    print('Loading dataset...')
    dataset = KTDataset(csv_path)
    num_questions = dataset.num_questions

    # split indices for central train/val/test (70/15/15)
    n = len(dataset)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    t1 = int(0.7 * n)
    t2 = int(0.85 * n)
    train_idx, val_idx, test_idx = idx[:t1], idx[t1:t2], idx[t2:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32)

    # Centralized training
    print('===== Centralized DKT training =====')
    central_model = DKT(num_questions).to(device)
    opt = torch.optim.Adam(central_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    epochs = 5

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f'Central Epoch {epoch+1}/{epochs}')
        central_model.train()
        for q, a in pbar:
            q = q.to(device)
            a = a.to(device)
            opt.zero_grad()
            out = central_model(q, a)
            out = out[:, :-1, :]
            target = a[:, 1:]
            q_next = q[:, 1:]
            pred = out.gather(2, q_next.unsqueeze(-1)).squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()
            opt.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    acc_c, auc_c, rmse_c = evaluate(central_model, test_loader, device)
    print(f'Centralized -> ACC: {acc_c:.4f}, AUC: {auc_c:.4f}, RMSE: {rmse_c:.4f}')

    # Federated IID
    print('\n===== Federated DKT (IID) =====')
    num_clients = 5
    clients = split_clients(dataset, num_clients=num_clients)

    global_model = DKT(num_questions).to(device)
    rounds = 5
    local_epochs = 1

    # print client sizes
    iid_sizes = [len(c) for c in clients]
    print('Client sample counts (IID):', iid_sizes)

    for r in range(rounds):
        client_states = []
        client_sizes = []
        pbar_clients = tqdm(enumerate(clients), total=len(clients), desc=f'Fed-IID Round {r+1}/{rounds}')
        for i, client_ds in pbar_clients:
            loader = DataLoader(client_ds, batch_size=16, shuffle=True)
            local_model = DKT(num_questions).to(device)
            local_model.load_state_dict(global_model.state_dict())
            state = train_local(local_model, loader, device, local_epochs=local_epochs)
            # ensure cpu tensors
            client_states.append({k: (v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state.items()})
            size = len(client_ds)
            client_sizes.append(size)
            pbar_clients.set_postfix({'client': i, 'size': size})
        global_model = fed_avg(global_model, client_states, client_sizes)

        # evaluate global after this round
        acc_r, auc_r, rmse_r = evaluate(global_model, test_loader, device)
        print(f'After round {r+1} -> ACC: {acc_r:.4f}, AUC: {auc_r:.4f}, RMSE: {rmse_r:.4f}')

    acc_f_iid, auc_f_iid, rmse_f_iid = evaluate(global_model, test_loader, device)
    print(f'Fed IID -> ACC: {acc_f_iid:.4f}, AUC: {auc_f_iid:.4f}, RMSE: {rmse_f_iid:.4f}')

    # Federated non-IID
    print('\n===== Federated DKT (Non-IID) =====')
    clients_ni = split_clients_noniid(dataset, num_clients=num_clients)

    ni_sizes = [len(c) for c in clients_ni]
    print('Client sample counts (Non-IID):', ni_sizes)

    global_model_ni = DKT(num_questions).to(device)
    for r in range(rounds):
        client_states = []
        client_sizes = []
        pbar_clients = tqdm(enumerate(clients_ni), total=len(clients_ni), desc=f'Fed-NonIID Round {r+1}/{rounds}')
        for i, client_ds in pbar_clients:
            loader = DataLoader(client_ds, batch_size=16, shuffle=True)
            local_model = DKT(num_questions).to(device)
            local_model.load_state_dict(global_model_ni.state_dict())
            state = train_local(local_model, loader, device, local_epochs=local_epochs)
            client_states.append({k: (v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state.items()})
            size = len(client_ds)
            client_sizes.append(size)
            pbar_clients.set_postfix({'client': i, 'size': size})
        global_model_ni = fed_avg(global_model_ni, client_states, client_sizes)

        acc_r, auc_r, rmse_r = evaluate(global_model_ni, test_loader, device)
        print(f'After round {r+1} (Non-IID) -> ACC: {acc_r:.4f}, AUC: {auc_r:.4f}, RMSE: {rmse_r:.4f}')

    acc_f_ni, auc_f_ni, rmse_f_ni = evaluate(global_model_ni, test_loader, device)
    print(f'Fed Non-IID -> ACC: {acc_f_ni:.4f}, AUC: {auc_f_ni:.4f}, RMSE: {rmse_f_ni:.4f}')


if __name__ == '__main__':
    run()



# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# import numpy as np
# from sklearn.metrics import roc_auc_score, accuracy_score
# from tqdm import tqdm

# from model.dataset import KTDataset, split_clients, split_clients_noniid
# from model.dkt import DKT


# def evaluate(model, loader, device):
#     model.eval()
#     preds = []
#     targets = []
#     with torch.no_grad():
#         for q, a in loader:
#             q = q.to(device)
#             a = a.to(device)

#             out = model(q, a)
#             out = out[:, :-1, :]
#             target = a[:, 1:]
#             q_next = q[:, 1:]

#             pred = out.gather(2, q_next.unsqueeze(-1)).squeeze(-1)

#             # ensure types: preds=float, targets=int
#             preds.extend(pred.flatten().cpu().numpy().astype(float))
#             targets.extend(target.flatten().cpu().numpy().astype(int))

#     preds = np.array(preds, dtype=float)
#     targets = np.array(targets, dtype=int)

#     # Accuracy uses binarized predictions
#     bin_preds = (preds >= 0.5).astype(int)
#     acc = accuracy_score(targets, bin_preds)

#     # AUC requires both classes present; handle degenerate cases
#     try:
#         auc = roc_auc_score(targets, preds)
#     except ValueError:
#         auc = float('nan')
#     return acc, auc


# def train_local(model, loader, device, local_epochs=1, lr=1e-3):
#     model.train()
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCELoss()

#     for _ in range(local_epochs):
#         for q, a in loader:
#             q = q.to(device)
#             a = a.to(device)

#             opt.zero_grad()
#             out = model(q, a)
#             out = out[:, :-1, :]
#             target = a[:, 1:]
#             q_next = q[:, 1:]
#             pred = out.gather(2, q_next.unsqueeze(-1)).squeeze(-1)

#             loss = criterion(pred, target)
#             loss.backward()
#             opt.step()

#     return model.state_dict()


# def fed_avg(global_model, client_states):
#     new_state = {}
#     g_state = global_model.state_dict()
#     for key in g_state.keys():
#         new_state[key] = sum([client_state[key].float() for client_state in client_states]) / len(client_states)
#     global_model.load_state_dict(new_state)
#     return global_model


# def run():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     csv_path = '2015_100_skill_builders_main_problems.csv'
#     assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

#     print('Loading dataset...')
#     dataset = KTDataset(csv_path)
#     num_questions = dataset.num_questions

#     # split indices for central train/val/test (70/15/15)
#     n = len(dataset)
#     idx = np.arange(n)
#     np.random.seed(42)
#     np.random.shuffle(idx)
#     t1 = int(0.7 * n)
#     t2 = int(0.85 * n)
#     train_idx, val_idx, test_idx = idx[:t1], idx[t1:t2], idx[t2:]

#     train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
#     val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32)
#     test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32)

#     # Centralized training
#     print('===== Centralized DKT training =====')
#     central_model = DKT(num_questions).to(device)
#     opt = torch.optim.Adam(central_model.parameters(), lr=1e-3)
#     criterion = nn.BCELoss()
#     epochs = 5

#     for epoch in range(epochs):
#         pbar = tqdm(train_loader, desc=f'Central Epoch {epoch+1}/{epochs}')
#         central_model.train()
#         for q, a in pbar:
#             q = q.to(device)
#             a = a.to(device)
#             opt.zero_grad()
#             out = central_model(q, a)
#             out = out[:, :-1, :]
#             target = a[:, 1:]
#             q_next = q[:, 1:]
#             pred = out.gather(2, q_next.unsqueeze(-1)).squeeze(-1)
#             loss = criterion(pred, target)
#             loss.backward()
#             opt.step()
#             pbar.set_postfix({'loss': f'{loss.item():.4f}'})

#     acc_c, auc_c = evaluate(central_model, test_loader, device)
#     print(f'Centralized -> ACC: {acc_c:.4f}, AUC: {auc_c:.4f}')

#     # Federated IID
#     print('\n===== Federated DKT (IID) =====')
#     num_clients = 5
#     clients = split_clients(dataset, num_clients=num_clients)

#     global_model = DKT(num_questions).to(device)
#     rounds = 5
#     local_epochs = 1

#     for r in range(rounds):
#         client_states = []
#         pbar_clients = tqdm(enumerate(clients), total=len(clients), desc=f'Fed-IID Round {r+1}/{rounds}')
#         for i, client_ds in pbar_clients:
#             loader = DataLoader(client_ds, batch_size=16, shuffle=True)
#             local_model = DKT(num_questions).to(device)
#             local_model.load_state_dict(global_model.state_dict())
#             state = train_local(local_model, loader, device, local_epochs=local_epochs)
#             # ensure tensors
#             client_states.append({k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in state.items()})
#             pbar_clients.set_postfix({'client': i})
#         global_model = fed_avg(global_model, client_states)

#     acc_f_iid, auc_f_iid = evaluate(global_model, test_loader, device)
#     print(f'Fed IID -> ACC: {acc_f_iid:.4f}, AUC: {auc_f_iid:.4f}')

#     # Federated non-IID
#     print('\n===== Federated DKT (Non-IID) =====')
#     clients_ni = split_clients_noniid(dataset, num_clients=num_clients)

#     global_model_ni = DKT(num_questions).to(device)
#     for r in range(rounds):
#         client_states = []
#         pbar_clients = tqdm(enumerate(clients_ni), total=len(clients_ni), desc=f'Fed-NonIID Round {r+1}/{rounds}')
#         for i, client_ds in pbar_clients:
#             loader = DataLoader(client_ds, batch_size=16, shuffle=True)
#             local_model = DKT(num_questions).to(device)
#             local_model.load_state_dict(global_model_ni.state_dict())
#             state = train_local(local_model, loader, device, local_epochs=local_epochs)
#             client_states.append({k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in state.items()})
#             pbar_clients.set_postfix({'client': i})
#         global_model_ni = fed_avg(global_model_ni, client_states)

#     acc_f_ni, auc_f_ni = evaluate(global_model_ni, test_loader, device)
#     print(f'Fed Non-IID -> ACC: {acc_f_ni:.4f}, AUC: {auc_f_ni:.4f}')


# if __name__ == '__main__':
#     run()
