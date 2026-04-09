# =============================================================================
# utils.py — Shared utilities for Continual Learning Prompt Injection Detection
# Kaggle: dataset **pi-detection-utils** → /kaggle/input/pi-detection-utils/
# Put utils.py at that folder’s root, or under **antidote/**. Notebooks use:
#   sys.path.append(...) then: from utils import *
# Requires torch at import time (Kaggle GPU images include it).
# =============================================================================

import os, json, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# Global config — edit here, propagates to all notebooks
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def resolve_training_device():
    """
    Pick cuda only if this PyTorch build includes kernels for the visible GPU.
    New wheels often support sm_70+ only; Tesla P100 is sm_60 and then fails at
    runtime with: no kernel image is available for execution on the device.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    major, minor = torch.cuda.get_device_capability(0)
    tag = f'sm_{major}{minor}'
    try:
        built_for = torch.cuda.get_arch_list()
    except Exception:
        built_for = []
    if built_for and tag not in built_for:
        return torch.device('cpu')
    if built_for:
        return torch.device('cuda')
    try:
        t = torch.ones(1, device='cuda')
        _ = (t + 1).cpu()
        torch.cuda.synchronize()
        return torch.device('cuda')
    except Exception:
        return torch.device('cpu')


DEVICE = resolve_training_device()
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
elif torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(
        f'[utils] Training on CPU: {torch.cuda.get_device_name(0)} ({major}.{minor}) '
        f'has no kernels in this PyTorch build (needs a GPU with compute capability '
        f'≥ 7.0, e.g. T4, or an older PyTorch wheel with sm_{major}{minor}).'
    )

CFG = {
    # Model
    'model_name': 'microsoft/deberta-v3-base',
    'max_len': 256,
    'num_labels': 2,
    # Training
    'batch_size': 32,
    'lr': 2e-5,
    'epochs_per_task': 3,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    # Data
    'max_samples_per_task': 20000,
    'val_split': 0.15,
    'test_split': 0.15,
    # EWC
    'ewc_lambda': 1000,
    'fisher_samples': 1000,
    # Replay
    'replay_ratio': 0.10,
    'replay_buffer_max': 5000,
    # I/O — parquet files written by NB1, read by NB2/3/4
    'data_dir': '/kaggle/input/pi-detection-data',
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'results_dir': '/kaggle/working/results',
    'replay_dir': '/kaggle/working/replay_buffer',
}


# -----------------------------------------------------------------------------
# Data hygiene (invalid labels → NaN loss in CrossEntropyLoss)
# -----------------------------------------------------------------------------
def sanitize_text_label_df(df, context='data'):
    """
    Require columns text, label with labels in {0, ..., num_labels-1}.
    Parquet nullable ints, strings, or bad exports often produce values outside
    that range; CE loss then becomes NaN with no clear Python error.
    """
    if df.empty:
        raise ValueError(f'{context}: empty dataframe')
    need = {'text', 'label'}
    if not need.issubset(df.columns):
        raise ValueError(f'{context}: need columns {need}, got {list(df.columns)}')
    out = df[list(need)].copy()
    out['text'] = out['text'].astype(str)
    lab = pd.to_numeric(out['label'], errors='coerce')
    if lab.isna().any():
        n = int(lab.isna().sum())
        raise ValueError(
            f'{context}: {n} rows have non-numeric label (NaN). '
            f'Fix the parquet or drop those rows in NB1.'
        )
    lab = lab.astype(np.int64)
    hi = CFG['num_labels'] - 1
    bad = (lab < 0) | (lab > hi)
    if bad.any():
        bad_vals = sorted(lab[bad].unique().tolist())
        raise ValueError(
            f'{context}: labels must be integers in 0..{hi} (num_labels={CFG["num_labels"]}), '
            f'found {bad_vals}. This causes NaN CrossEntropyLoss.'
        )
    out['label'] = lab
    if out['label'].nunique() < 2:
        print(
            f'WARNING {context}: only one class in split ({out["label"].iloc[0]}); '
            f'F1/loss may be misleading.'
        )
    return out


# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class PIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# Data splitting & loader creation
# -----------------------------------------------------------------------------
def split_dataset(df, val_split, test_split, seed=SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n       = len(df)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    return df.iloc[n_test + n_val:], df.iloc[n_test:n_test + n_val], df.iloc[:n_test]


def make_loaders(df, tokenizer, batch_size=None, val_split=None, test_split=None):
    batch_size = batch_size or CFG['batch_size']
    val_split  = val_split  or CFG['val_split']
    test_split = test_split or CFG['test_split']
    df = sanitize_text_label_df(df, context='make_loaders')
    train_df, val_df, test_df = split_dataset(df, val_split, test_split)
    train_ds = PIDataset(train_df, tokenizer, CFG['max_len'])
    val_ds   = PIDataset(val_df,   tokenizer, CFG['max_len'])
    test_ds  = PIDataset(test_df,  tokenizer, CFG['max_len'])
    kw = dict(num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        df,
    )


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG['model_name'], num_labels=CFG['num_labels']
    )
    return model.to(DEVICE)


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'  Checkpoint saved: {path}')


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def save_results(results_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f'  Results saved: {path}')


def load_results(path):
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion  = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            loss           = criterion(outputs.logits, labels)
            total_loss    += loss.item()
            preds          = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    n_batches = len(loader)
    avg_loss = (total_loss / n_batches) if n_batches else float('nan')
    return f1, avg_loss


# -----------------------------------------------------------------------------
# CL Metrics
# -----------------------------------------------------------------------------
def compute_cl_metrics(results_matrix):
    """
    results_matrix[i][j] = test F1 on task j after training on task i.
    BWT: negative = forgetting. Less negative = better.
    AvgAcc: mean final F1 across all tasks.
    """
    T = len(results_matrix)
    if T < 2:
        return {}
    bwt = sum(results_matrix[T-1][i] - results_matrix[i][i] for i in range(T-1)) / (T-1)
    avg_acc = np.mean(results_matrix[T-1])
    return {'BWT': round(bwt, 4), 'AvgAcc': round(avg_acc, 4)}


# -----------------------------------------------------------------------------
# EWC
# -----------------------------------------------------------------------------
class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    After each task: compute diagonal Fisher + store optimal weights.
    During next task: add quadratic penalty to CE loss.
    """
    def __init__(self, model, lam=None):
        self.lam        = lam or CFG['ewc_lambda']
        self.params     = {}
        self.fisher     = {}
        self.task_count = 0

    def compute_fisher(self, model, data_loader, n_samples=None):
        n_samples   = n_samples or CFG['fisher_samples']
        model.eval()
        fisher_diag = {n: torch.zeros_like(p)
                       for n, p in model.named_parameters() if p.requires_grad}
        criterion   = nn.CrossEntropyLoss()
        samples_seen = 0
        for batch in data_loader:
            if samples_seen >= n_samples:
                break
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_diag[n] += p.grad.detach() ** 2
            samples_seen += len(labels)
        n_batches = max(1, samples_seen // data_loader.batch_size)
        for n in fisher_diag:
            fisher_diag[n] /= n_batches
        return fisher_diag

    def register_task(self, model, data_loader, task_name):
        print(f'  Computing Fisher matrix for {task_name}...')
        self.fisher[task_name] = self.compute_fisher(model, data_loader)
        self.params[task_name] = {n: p.detach().clone()
                                  for n, p in model.named_parameters() if p.requires_grad}
        self.task_count += 1
        print(f'  EWC registered {task_name}. Total tasks: {self.task_count}')

    def penalty(self, model):
        if self.task_count == 0:
            return torch.tensor(0.0, device=DEVICE)
        loss = torch.tensor(0.0, device=DEVICE)
        for task_name in self.params:
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.fisher[task_name]:
                    fisher = self.fisher[task_name][n].to(DEVICE)
                    mean   = self.params[task_name][n].to(DEVICE)
                    loss  += (fisher * (p - mean) ** 2).sum()
        return self.lam * loss


# -----------------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    """
    Fixed-size random sample store from previous tasks.
    Mixed into new-task training batches to combat forgetting.
    """
    def __init__(self, max_size=None, replay_ratio=None):
        self.max_size     = max_size     or CFG['replay_buffer_max']
        self.replay_ratio = replay_ratio or CFG['replay_ratio']
        self.buffer       = []

    def add_task(self, df, task_name, tokenizer):
        if len(self.buffer) >= self.max_size:
            keep         = int(self.max_size * 0.5)
            self.buffer  = random.sample(self.buffer, keep)
        n_add      = min(self.max_size - len(self.buffer), len(df))
        sample_df  = df.sample(n=n_add, random_state=SEED)
        for _, row in sample_df.iterrows():
            self.buffer.append({'text': row['text'], 'label': int(row['label'])})
        print(f'  Replay buffer: +{n_add} from {task_name}. Total: {len(self.buffer)}')

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.buffer, f)

    def load(self, path):
        with open(path) as f:
            self.buffer = json.load(f)
        print(f'  Replay buffer loaded: {len(self.buffer)} samples')

    def sample_loader(self, n, tokenizer, batch_size=None):
        if not self.buffer:
            return None
        batch_size = batch_size or CFG['batch_size']
        n          = min(n, len(self.buffer))
        samples    = random.sample(self.buffer, n)
        replay_df  = pd.DataFrame(samples)
        replay_ds  = PIDataset(replay_df, tokenizer, CFG['max_len'])
        return DataLoader(replay_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    def is_empty(self):
        return len(self.buffer) == 0


# -----------------------------------------------------------------------------
# Core training loop
# -----------------------------------------------------------------------------
def train_task(model, task_name, train_loader, val_loader,
               tokenizer, ewc=None, replay_buffer=None, epochs=None):
    epochs     = epochs or CFG['epochs_per_task']
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay']
    )
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * CFG['warmup_ratio'])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss()
    best_val_f1  = 0.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches  = 0

        replay_iter = None
        if replay_buffer and not replay_buffer.is_empty():
            n_replay     = max(1, int(len(train_loader.dataset) * CFG['replay_ratio']))
            replay_loader = replay_buffer.sample_loader(n_replay, tokenizer)
            replay_iter  = iter(replay_loader) if replay_loader else None

        for batch in train_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)

            if ewc is not None:
                loss = loss + ewc.penalty(model)

            if replay_iter is not None:
                try:
                    r_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = None
                    r_batch = None
                if r_batch is not None:
                    r_out  = model(input_ids=r_batch['input_ids'].to(DEVICE),
                                   attention_mask=r_batch['attention_mask'].to(DEVICE))
                    r_loss = criterion(r_out.logits, r_batch['labels'].to(DEVICE))
                    loss   = loss + r_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            li = loss.item()
            if not np.isfinite(li):
                with torch.no_grad():
                    mx = outputs.logits.max().item()
                    mn = outputs.logits.min().item()
                raise RuntimeError(
                    'Non-finite loss during training. Common causes: labels outside '
                    f'0..{CFG["num_labels"] - 1}, or bad inputs. '
                    f'Batch label min/max: {labels.min().item()}/{labels.max().item()}; '
                    f'logits min/max: {mn}/{mx}.'
                )
            total_loss += li
            n_batches  += 1

        val_f1, val_loss = evaluate(model, val_loader)
        print(f'  Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {total_loss/n_batches:.4f} | '
              f'Val F1: {val_f1:.4f} | Val Loss: {val_loss:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
    print(f'  Best val F1: {best_val_f1:.4f}')
    return best_val_f1


# -----------------------------------------------------------------------------
# Full experiment runner
# -----------------------------------------------------------------------------
def run_experiment(experiment_name, tasks, tokenizer,
                   use_ewc=False, use_replay=False, joint_training=False):
    print(f'\n{"="*60}\nEXPERIMENT: {experiment_name}')
    print(f'  EWC: {use_ewc} | Replay: {use_replay} | Joint: {joint_training}\n{"="*60}')

    model         = load_model()
    ewc           = EWC(model) if use_ewc else None
    replay_buffer = ReplayBuffer() if use_replay else None
    task_names    = list(tasks.keys())
    T             = len(task_names)
    results_matrix = [[None] * T for _ in range(T)]
    per_task_f1   = {}

    if joint_training:
        all_dfs      = [tasks[tn]['df'] for tn in task_names]
        combined_df  = pd.concat(all_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
        tr, va, _, _ = make_loaders(combined_df, tokenizer)
        train_task(model, 'joint', tr, va, tokenizer)
        for j, tn in enumerate(task_names):
            f1, _                  = evaluate(model, tasks[tn]['test'])
            results_matrix[T-1][j] = f1
            per_task_f1[tn]        = f1
            print(f'  Joint → {tn} test F1: {f1:.4f}')
    else:
        for i, task_name in enumerate(task_names):
            print(f'\n--- Training on {task_name} (step {i+1}/{T}) ---')
            train_task(
                model, task_name,
                tasks[task_name]['train'], tasks[task_name]['val'],
                tokenizer, ewc=ewc, replay_buffer=replay_buffer,
            )
            if ewc:
                ewc.register_task(model, tasks[task_name]['train'], task_name)
            if replay_buffer:
                replay_buffer.add_task(tasks[task_name]['df'], task_name, tokenizer)
                replay_buffer.save(
                    f'{CFG["replay_dir"]}/{experiment_name}.json'
                )
            for j, eval_task in enumerate(task_names):
                if j <= i:
                    f1, _                  = evaluate(model, tasks[eval_task]['test'])
                    results_matrix[i][j]   = f1
                    print(f'  After {task_name} → {eval_task} test F1: {f1:.4f}')
            save_checkpoint(
                model,
                f'{CFG["checkpoint_dir"]}/{experiment_name}_after_{task_name}.pt'
            )

        print('\n--- Final evaluation on all tasks ---')
        for j, tn in enumerate(task_names):
            f1, _                  = evaluate(model, tasks[tn]['test'])
            results_matrix[T-1][j] = f1
            per_task_f1[tn]        = f1
            print(f'  Final → {tn} test F1: {f1:.4f}')

    clean_matrix = [[v if v is not None else 0.0 for v in row] for row in results_matrix]
    cl_metrics   = compute_cl_metrics(clean_matrix)
    results      = {
        'experiment':     experiment_name,
        'per_task_f1':    per_task_f1,
        'avg_f1':         float(np.mean(list(per_task_f1.values()))),
        'cl_metrics':     cl_metrics,
        'results_matrix': clean_matrix,
    }
    print(f'\nSUMMARY | Avg F1: {results["avg_f1"]:.4f} | CL: {cl_metrics}')
    return results, model
