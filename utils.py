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
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Config
# FIX 1: lr dropped 2e-5 → 1e-5, eps raised to 1e-6, batch_size halved to 16
# FIX 2: added adam_eps key for explicit control
# -----------------------------------------------------------------------------
CFG = {
    # Model
    # DeBERTa-v3-base works after the pooler init fix below.
    # If you still get NaN after updating utils.py, switch to:
    #   'distilbert-base-uncased'  (faster, more stable, slightly lower ceiling)
    'model_name': 'microsoft/deberta-v3-base',
    'max_len': 256,
    'num_labels': 2,

    # Training
    'batch_size': 16,       # was 32 — smaller batches reduce gradient variance
    'lr': 1e-5,             # was 2e-5 — DeBERTa-v3 needs lower lr
    'adam_eps': 1e-6,       # was default 1e-8 — higher eps = more stable
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

    # I/O
    'data_dir':       '/kaggle/input/pi-detection-data',
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'results_dir':    '/kaggle/working/results',
    'replay_dir':     '/kaggle/working/replay_buffer',
}


# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class PIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts     = df['text'].tolist()
        self.labels    = df['label'].tolist()
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
    df      = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n       = len(df)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    return df.iloc[n_test + n_val:], df.iloc[n_test:n_test + n_val], df.iloc[:n_test]


def make_loaders(df, tokenizer, batch_size=None, val_split=None, test_split=None):
    batch_size = batch_size or CFG['batch_size']
    val_split  = val_split  or CFG['val_split']
    test_split = test_split or CFG['test_split']
    train_df, val_df, test_df = split_dataset(df, val_split, test_split)
    train_ds = PIDataset(train_df, tokenizer, CFG['max_len'])
    val_ds   = PIDataset(val_df,   tokenizer, CFG['max_len'])
    test_ds  = PIDataset(test_df,  tokenizer, CFG['max_len'])
    kw = dict(num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG['model_name'],
        num_labels=CFG['num_labels'],
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,   # ADD THIS — forces fp32 weights
    )
    for name, param in model.named_parameters():
        if 'classifier' in name or 'pooler' in name:
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(param)
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
    total_loss = 0.0
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
    f1       = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    avg_loss = total_loss / max(1, len(loader))
    return f1, avg_loss


# -----------------------------------------------------------------------------
# CL Metrics
# -----------------------------------------------------------------------------
def compute_cl_metrics(results_matrix):
    T = len(results_matrix)
    if T < 2:
        return {}
    bwt     = sum(results_matrix[T-1][i] - results_matrix[i][i] for i in range(T-1)) / (T-1)
    avg_acc = float(np.mean(results_matrix[T-1]))
    return {'BWT': round(float(bwt), 4), 'AvgAcc': round(avg_acc, 4)}


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
        n_samples    = n_samples or CFG['fisher_samples']
        model.eval()
        fisher_diag  = {n: torch.zeros_like(p)
                        for n, p in model.named_parameters() if p.requires_grad}
        criterion    = nn.CrossEntropyLoss()
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
            if torch.isnan(loss):
                continue
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
        self.params[task_name] = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
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
    def __init__(self, max_size=None, replay_ratio=None):
        self.max_size     = max_size     or CFG['replay_buffer_max']
        self.replay_ratio = replay_ratio or CFG['replay_ratio']
        self.buffer       = []

    def add_task(self, df, task_name, tokenizer):
        if len(self.buffer) >= self.max_size:
            keep        = int(self.max_size * 0.5)
            self.buffer = random.sample(self.buffer, keep)
        n_add     = min(self.max_size - len(self.buffer), len(df))
        sample_df = df.sample(n=n_add, random_state=SEED)
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
    epochs    = epochs or CFG['epochs_per_task']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay'],
        eps=CFG['adam_eps'],
    )
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * CFG['warmup_ratio'])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss()

    use_amp = False   # disabled — DeBERTa-v3 in fp32 doesn't need AMP
    scaler  = None

    best_val_f1 = 0.0
    best_state  = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        n_nan      = 0

        replay_iter = None
        if replay_buffer and not replay_buffer.is_empty():
            n_replay      = max(1, int(len(train_loader.dataset) * CFG['replay_ratio']))
            replay_loader = replay_buffer.sample_loader(n_replay, tokenizer)
            replay_iter   = iter(replay_loader) if replay_loader else None

        for batch in train_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            optimizer.zero_grad()

            try:
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss    = criterion(outputs.logits, labels)
                        if ewc is not None:
                            loss = loss + ewc.penalty(model)
                        if replay_iter is not None:
                            try:
                                r_batch = next(replay_iter)
                            except StopIteration:
                                replay_iter = None
                                r_batch     = None
                            if r_batch is not None:
                                r_out  = model(
                                    input_ids=r_batch['input_ids'].to(DEVICE),
                                    attention_mask=r_batch['attention_mask'].to(DEVICE),
                                )
                                loss = loss + criterion(r_out.logits, r_batch['labels'].to(DEVICE))

                    if torch.isnan(loss) or torch.isinf(loss):
                        n_nan += 1
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss    = criterion(outputs.logits, labels)
                    if ewc is not None:
                        loss = loss + ewc.penalty(model)
                    if replay_iter is not None:
                        try:
                            r_batch = next(replay_iter)
                        except StopIteration:
                            replay_iter = None
                            r_batch     = None
                        if r_batch is not None:
                            r_out  = model(
                                input_ids=r_batch['input_ids'].to(DEVICE),
                                attention_mask=r_batch['attention_mask'].to(DEVICE),
                            )
                            loss = loss + criterion(r_out.logits, r_batch['labels'].to(DEVICE))

                    if torch.isnan(loss) or torch.isinf(loss):
                        n_nan += 1
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                total_loss += loss.item()
                n_batches  += 1

            except RuntimeError as e:
                print(f'  RuntimeError skipped: {e}')
                continue

        if n_nan > 0:
            print(f'  WARNING: {n_nan}/{n_batches + n_nan} batches skipped (NaN/Inf)')

        if n_batches == 0:
            print(f'  ERROR: all batches in epoch {epoch+1} produced NaN.')
            print(f'  Try setting model_name = "distilbert-base-uncased" in CFG.')
            break

        val_f1, val_loss = evaluate(model, val_loader)
        avg_loss = total_loss / n_batches
        print(f'  Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {avg_loss:.4f} | '
              f'Val F1: {val_f1:.4f} | '
              f'Val Loss: {val_loss:.4f}')

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
        all_dfs     = [tasks[tn]['df'] for tn in task_names]
        combined_df = pd.concat(all_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
        tr, va, _   = make_loaders(combined_df, tokenizer)
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
                replay_buffer.save(f'{CFG["replay_dir"]}/{experiment_name}.json')
            for j, eval_task in enumerate(task_names):
                if j <= i:
                    f1, _                = evaluate(model, tasks[eval_task]['test'])
                    results_matrix[i][j] = f1
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
        'avg_f1':         float(np.mean(list(per_task_f1.values()))) if per_task_f1 else 0.0,
        'cl_metrics':     cl_metrics,
        'results_matrix': clean_matrix,
    }
    print(f'\nSUMMARY | Avg F1: {results["avg_f1"]:.4f} | CL: {cl_metrics}')
    return results, model
