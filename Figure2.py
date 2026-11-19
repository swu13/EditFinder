#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 pipeline (A/B/C/D/E/F/G)

[Figure to output mapping]
    Figure 2A/B: Alpha scan across sequence lengths (data source CSV)
    Figure 2C   : Full model vs OneMer+Score ablation ROC
    Figure 2D   : Full model vs Sequence-Only ablation ROC
    Figure 2E   : Full model vs Score-Only ablation ROC
    Figure 2F   : Model comparison ROC on held-out test set
    Figure 2G   : Model comparison ROC on external real FASTA set

[External data (must be prepared in advance; fill paths in CONFIG['input'])]
    scan section :
        FASTA files for each length
        (development positive/negative, test positive/negative)
    ablation section :
        111 bp FASTA files (development.fa, development2.fa, test.fa, test2.fa)
    test_set section :
        ATTIC result CSV, EditPredict result TXT, pretrained model .pth,
        111 bp FASTA files, NW score CSVs (RNAtest_scores.csv, SNPtest_scores.csv)
    real_fa section :
        EditPredict edit/snp result TXT, ATTIC edit/snp result CSV,
        pretrained model .pth, external edit/snp FASTA, NW score CSVs

[Figure to file mapping]
    Figure 2A/B: parameter_scan/alpha_scan_all_lengths.csv
    Figure 2C  : ablation/roc_Full_vs_OneMer.png
    Figure 2D  : ablation/roc_Full_vs_SeqOnly.png
    Figure 2E  : ablation/roc_Full_vs_ScoreOnly.png
    Figure 2F  : models_compare/Test_Set/roc_curves.png
    Figure 2G  : models_compare/RealFa/roc_curves.png
"""

import os
import re
import csv
import copy
import time
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG  (replace every path below with your own before running)
# ============================================================================
CONFIG = {
    "input": {
        "scan": {
            "base_dir_111":  "/path/to/Input_data/111",
            "other_dir":     "/path/to/Input_data/111/111VSother",
            "lengths":       [111, 51, 61, 71, 81, 91, 101],
        },
        "ablation": {
            "pos_development_fa": "/path/to/Input_data/111/development.fa",
            "neg_development_fa": "/path/to/Input_data/111/development2.fa",
            "pos_test_fa":        "/path/to/Input_data/111/test.fa",
            "neg_test_fa":        "/path/to/Input_data/111/test2.fa",
        },
        "test_set": {
            "attic_result":       "/path/to/models_compare/Test_Set/ATTICresult.csv",
            "editpredict_result": "/path/to/models_compare/Test_Set/EditPrdictResults.txt",
            "model_path":         "/path/to/Model_final/final_model/final_model_seed333.pth",
            "pos_test_fa":        "/path/to/Input_data/111/test.fa",
            "neg_test_fa":        "/path/to/Input_data/111/test2.fa",
            "pos_scores_csv":     "/path/to/RNAtest_scores.csv",
            "neg_scores_csv":     "/path/to/SNPtest_scores.csv",
            "global_min":         41.0,
            "global_max":         105.0,
        },
        "real_fa": {
            "editpredict_edit":   "/path/to/other_model_results/edit_EditPredict_result.txt",
            "editpredict_snp":    "/path/to/other_model_results/snp_Predict_result.txt",
            "attic_edit":         "/path/to/other_model_results/ATTIC_edit.csv",
            "attic_snp":          "/path/to/other_model_results/ATTIC_snp.csv",
            "model_path":         "/path/to/Model_final/final_model/final_model_seed333.pth",
            "edit_fa":            "/path/to/region_fastq/REDIportal_edit_111_final_sequences_A.fasta",
            "snp_fa":             "/path/to/region_fastq/snp/snp_111_final_sequences_A.fasta",
            "edit_scores_csv":    "/path/to/NW_results/edit/alignment_scores.csv",
            "snp_scores_csv":     "/path/to/NW_results/snp/alignment_scores.csv",
            "global_min":         41.0,
            "global_max":         105.0,
            "max_n_ratio":        0.3,
        },
    },
    "output": {
        "out_dir": "/path/to/Model_fixed/Figure2",
    },
    "model": {
        "alpha":              279,
        "conv1_out_channels": 128,
        "conv2_out_channels": 256,
        "fc1_out_features":   256,
        "fc2_out_features":   128,
        "dropout_conv":       0.4,
        "dropout_fc":         0.7,
    },
    "scan": {
        "alpha_values": list(range(1, 301, 1)),
        "num_epochs":   50,
        "k_folds":      10,
        "batch_size":   128,
        "lr":           1e-4,
        "weight_decay": 1e-5,
        "patience":     5,
    },
    "ablation": {
        "random_seed": 42,
        "k_folds":     10,
        "num_epochs":  50,
        "batch_size":  128,
        "lr":          1e-4,
        "weight_decay": 1e-5,
        "patience":    5,
        "scheduler_factor":   0.5,
        "scheduler_patience": 3,
        "min_lr":      1e-6,
        "final_seeds": [42, 0, 1, 2, 7, 13, 314, 2718, 111, 222, 333, 444,
                        555, 666, 777, 888, 999, 123, 1234, 12345, 123456,
                        3407, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027],
    },
    "align": {
        "match":      1,
        "mismatch":   0,
        "open_gap":   -10,
        "extend_gap": -0.1,
        "mode":       "global",
    },
}


# ============================================================================
# Derived paths and parameters
# ============================================================================
OUT_ROOT  = CONFIG["output"]["out_dir"]
SCAN_DIR  = os.path.join(OUT_ROOT, "parameter_scan")
SCORE_DIR = os.path.join(SCAN_DIR, "NW_score")
ABL_DIR   = os.path.join(OUT_ROOT, "ablation")
TEST_DIR  = os.path.join(OUT_ROOT, "models_compare", "Test_Set")
REAL_DIR  = os.path.join(OUT_ROOT, "models_compare", "RealFa")
for d in (SCAN_DIR, SCORE_DIR, ABL_DIR, TEST_DIR, REAL_DIR):
    os.makedirs(d, exist_ok=True)

SCAN_CSV = os.path.join(SCAN_DIR, "alpha_scan_all_lengths.csv")

# Model hyperparameters
ALPHA              = CONFIG["model"]["alpha"]
CONV1_OUT_CHANNELS = CONFIG["model"]["conv1_out_channels"]
CONV2_OUT_CHANNELS = CONFIG["model"]["conv2_out_channels"]
FC1_OUT_FEATURES   = CONFIG["model"]["fc1_out_features"]
FC2_OUT_FEATURES   = CONFIG["model"]["fc2_out_features"]
DROPOUT_CONV       = CONFIG["model"]["dropout_conv"]
DROPOUT_FC         = CONFIG["model"]["dropout_fc"]

# Alignment parameters
ALIGN_MATCH      = CONFIG["align"]["match"]
ALIGN_MISMATCH   = CONFIG["align"]["mismatch"]
ALIGN_OPEN_GAP   = CONFIG["align"]["open_gap"]
ALIGN_EXTEND_GAP = CONFIG["align"]["extend_gap"]
ALIGN_MODE       = CONFIG["align"]["mode"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Common utilities
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_fasta(file_path, with_headers=False):
    sequences, headers = [], []
    with open(file_path, 'r') as f:
        seq, header = '', ''
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    headers.append(header)
                    seq = ''
                header = line.strip()
                continue
            seq += line.strip()
        if seq:
            sequences.append(seq)
            headers.append(header)
    if with_headers:
        return sequences, headers, len(sequences[0]) if sequences else 0
    return sequences, len(sequences[0]) if sequences else 0


def load_sequences_upper(path):
    """Return [(header, seq_upper)] for pairwise alignment stage."""
    seqs, header, seq = [], None, ""
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    seqs.append((header, seq.upper()))
                header, seq = line, ""
            else:
                seq += line
        if header is not None:
            seqs.append((header, seq.upper()))
    return seqs


def make_aligner():
    a = PairwiseAligner()
    a.match_score = ALIGN_MATCH
    a.mismatch_score = ALIGN_MISMATCH
    a.open_gap_score = ALIGN_OPEN_GAP
    a.extend_gap_score = ALIGN_EXTEND_GAP
    a.target_end_gap_score = 0
    a.query_end_gap_score = 0
    a.mode = ALIGN_MODE
    return a


def precompute_pair_scores(query_seqs, ref_seqs, log_interval=2000):
    aligner = make_aligner()
    n_q = len(query_seqs)
    n_r = len(ref_seqs)
    M = np.full((n_q, n_r), -np.inf, dtype=np.float32)
    ref_norm = [t.strip().upper() for t in ref_seqs]

    print(f"Precomputing alignment scores: "
          f"{n_q} queries x {n_r} refs = {n_q * n_r:,} alignments")
    t0 = time.time()
    for i, q in enumerate(query_seqs):
        q = q.strip().upper()
        if not q:
            continue
        row = M[i]
        for j, t in enumerate(ref_norm):
            if not t or q == t:
                continue
            row[j] = aligner.score(q, t)
        if (i + 1) % log_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n_q - i - 1) / rate
            print(f"  [{i + 1}/{n_q}] {rate:.1f} q/s, ETA {eta / 60:.1f} min",
                  flush=True)
    print(f"Precompute done in {time.time() - t0:.1f}s")
    return M


def one_hot_3mer(seq):
    combinations = [''.join(n1 + n2 + n3)
                    for n1 in 'ATGC' for n2 in 'ATGC' for n3 in 'ATGC']
    encoding = {comb: [1 if comb == triple else 0 for triple in combinations]
                for comb in combinations}
    three_mers = [seq[i:i + 3] for i in range(len(seq) - 2)]
    return [encoding.get(mer, [0] * 64) for mer in three_mers]


def one_hot_1mer(seq):
    d = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
         'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    return [d.get(b, [0] * 4) for b in seq]


def seq2tensor(seq_list, ch, device=DEVICE):
    arr = np.stack([np.array(s) for s in seq_list])
    return torch.from_numpy(arr).float().to(device).permute(0, 2, 1)


def sequences_to_tensor_oh(seq_list, device=DEVICE):
    arr = np.stack([np.array(s) for s in seq_list])
    return torch.tensor(arr, dtype=torch.float, device=device).permute(0, 2, 1)


# ============================================================================
# Model definitions
# ============================================================================
class SimpleCNN(nn.Module):
    """Main model used for Figure 2A/B/F/G."""

    def __init__(self, seq_len, alpha=ALPHA):
        super().__init__()
        self.seq_len = seq_len
        self.alpha = alpha

        self.conv1 = nn.Conv1d(64, CONV1_OUT_CHANNELS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(CONV1_OUT_CHANNELS)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(DROPOUT_CONV)

        self.conv2 = nn.Conv1d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(CONV2_OUT_CHANNELS)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(DROPOUT_CONV)

        self.fc_input_size = self._calculate_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size + 1, FC1_OUT_FEATURES)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(DROPOUT_FC)

        self.fc2 = nn.Linear(FC1_OUT_FEATURES, FC2_OUT_FEATURES)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(DROPOUT_FC)

        self.fc3 = nn.Linear(FC2_OUT_FEATURES, 1)

    def _calculate_fc_input_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 64, self.seq_len)
            dummy = self.dropout1(self.relu1(self.bn1(self.conv1(dummy))))
            dummy = self.dropout2(self.relu2(self.bn2(self.conv2(dummy))))
            return dummy.view(1, -1).size(1)

    def forward(self, x, scores):
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        weighted_scores = (scores * self.alpha).unsqueeze(1)
        x = torch.cat((x, weighted_scores), dim=1)
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.dropout5(self.relu5(self.fc2(x)))
        return self.fc3(x).squeeze(1)


class CNNCompare(nn.Module):
    """Flexible variant used for ablation (1mer / 3mer / zero input / score on/off)."""

    def __init__(self, in_channels, use_score, alpha, input_len):
        super().__init__()
        self.in_channels = in_channels
        self.use_score = use_score
        self.alpha = alpha
        self.input_len = input_len

        self.conv1 = nn.Conv1d(in_channels, CONV1_OUT_CHANNELS,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(CONV1_OUT_CHANNELS)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(DROPOUT_CONV)

        self.conv2 = nn.Conv1d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(CONV2_OUT_CHANNELS)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(DROPOUT_CONV)

        self.fc_in = self._get_fc_dim()
        add_dim = 1 if use_score else 0
        self.fc1 = nn.Linear(self.fc_in + add_dim, FC1_OUT_FEATURES)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(DROPOUT_FC)
        self.fc2 = nn.Linear(FC1_OUT_FEATURES, FC2_OUT_FEATURES)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(DROPOUT_FC)
        self.fc3 = nn.Linear(FC2_OUT_FEATURES, 1)

    def _get_fc_dim(self):
        dummy = torch.randn(1, self.in_channels, self.input_len).to(
            next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            x = self.dropout1(self.relu1(self.bn1(self.conv1(dummy))))
            x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        self.train()
        return x.view(1, -1).size(1)

    def forward(self, x, score):
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        if self.use_score:
            x = torch.cat([x, (score * self.alpha).unsqueeze(1)], dim=1)
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.dropout5(self.relu5(self.fc2(x)))
        return self.fc3(x).squeeze(1)


# ============================================================================
# Figure 2A/B  -  alpha scan across lengths
# ============================================================================
def _scan_fasta_paths(length):
    base111 = CONFIG["input"]["scan"]["base_dir_111"]
    other   = CONFIG["input"]["scan"]["other_dir"]
    if length == 111:
        return (os.path.join(base111, "development.fa"),
                os.path.join(base111, "development2.fa"),
                os.path.join(base111, "test.fa"),
                os.path.join(base111, "test2.fa"))
    return (os.path.join(other, f"{length}development.fa"),
            os.path.join(other, f"{length}development2.fa"),
            os.path.join(other, f"{length}test.fa"),
            os.path.join(other, f"{length}test2.fa"))


def _scan_compute_pair_scores(length):
    cache_file = os.path.join(SCORE_DIR, f"pair_scores_{length}.npy")
    meta_file  = os.path.join(SCORE_DIR, f"pair_scores_{length}_meta.npy")

    development_pos_path, development_neg_path, test_pos_path, test_neg_path = \
        _scan_fasta_paths(length)
    pos_development = load_sequences_upper(development_pos_path)
    neg_development = load_sequences_upper(development_neg_path)
    pos_test        = load_sequences_upper(test_pos_path)
    neg_test        = load_sequences_upper(test_neg_path)

    n_pos_development = len(pos_development)
    n_neg_development = len(neg_development)
    n_pos_test        = len(pos_test)
    n_neg_test        = len(neg_test)
    N_q = n_pos_development + n_neg_development + n_pos_test + n_neg_test

    if os.path.exists(cache_file) and os.path.exists(meta_file):
        M = np.load(cache_file)
        meta = np.load(meta_file)
        if (M.shape == (N_q, n_pos_development)
                and tuple(meta.tolist()) == (n_pos_development, n_neg_development,
                                             n_pos_test, n_neg_test)):
            print(f"  [skip] cached {cache_file} shape={M.shape}")
            return M, n_pos_development, n_neg_development, n_pos_test, n_neg_test

    ref_seqs   = [s for _, s in pos_development]
    query_seqs = [s for _, s in (pos_development + neg_development +
                                 pos_test + neg_test)]
    ref_norm   = [t.strip().upper() for t in ref_seqs]
    q_norm     = [q.strip().upper() for q in query_seqs]

    n_q, n_r = len(q_norm), len(ref_norm)
    M = np.full((n_q, n_r), -np.inf, dtype=np.float32)
    aligner = make_aligner()
    print(f"  Computing pair scores: length={length}, "
          f"{n_q} queries x {n_r} refs = {n_q * n_r:,} alignments")
    t0 = time.time()
    for i, q in enumerate(q_norm):
        if not q:
            continue
        row = M[i]
        for j, t in enumerate(ref_norm):
            if not t or q == t:
                continue
            row[j] = aligner.score(q, t)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n_q - i - 1) / rate
            print(f"    [{i + 1}/{n_q}] {rate:.1f} q/s, ETA {eta / 60:.1f} min",
                  flush=True)

    np.save(cache_file, M)
    np.save(meta_file, np.array([n_pos_development, n_neg_development,
                                 n_pos_test, n_neg_test]))
    print(f"  Saved {cache_file} shape={M.shape}, elapsed={time.time() - t0:.1f}s")
    return M, n_pos_development, n_neg_development, n_pos_test, n_neg_test


def _scan_precompute_fold_scores(pair_scores, n_pos_development,
                                 n_neg_development, y_development):
    K = CONFIG["scan"]["k_folds"]
    N_development = n_pos_development + n_neg_development
    test_offset = N_development

    with np.errstate(invalid='ignore'):
        development_global_max = np.max(pair_scores[:N_development, :], axis=1)
    development_global_max = np.where(np.isfinite(development_global_max),
                                      development_global_max, -1.0).astype(np.float64)
    development_valid = development_global_max[development_global_max >= 0]
    if development_valid.size == 0:
        raise RuntimeError("No valid alignment score in Development set.")
    mn = float(development_valid.min())
    mx = float(development_valid.max())
    scale = (mx - mn) + 1e-8
    print(f"    [Global norm] n_valid={development_valid.size}, "
          f"mn={mn:.4f}, mx={mx:.4f}")

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    fold_data = []
    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(N_development), y_development)):
        ref_indices = np.array([int(i) for i in train_idx if i < n_pos_development],
                               dtype=np.int64)
        if len(ref_indices) == 0:
            raise RuntimeError(f"Fold {fold + 1}: no positive sample in train split")
        with np.errstate(invalid='ignore'):
            fold_max = np.max(pair_scores[:, ref_indices], axis=1)
        fold_max = np.where(np.isfinite(fold_max), fold_max, -1.0)

        fold_data.append({
            'train_idx':    train_idx,
            'val_idx':      val_idx,
            'train_scores': (fold_max[train_idx] - mn) / scale,
            'val_scores':   (fold_max[val_idx] - mn) / scale,
            'test_scores':  (fold_max[test_offset:] - mn) / scale,
        })
    return fold_data


def _scan_load_dataset_onehot(length):
    development_pos_path, development_neg_path, test_pos_path, test_neg_path = \
        _scan_fasta_paths(length)
    pos_development, _ = read_fasta(development_pos_path)
    neg_development, _ = read_fasta(development_neg_path)
    pos_test,  _ = read_fasta(test_pos_path)
    neg_test,  _ = read_fasta(test_neg_path)

    X_development = ([one_hot_3mer(s) for s in pos_development] +
                     [one_hot_3mer(s) for s in neg_development])
    y_development = [1] * len(pos_development) + [0] * len(neg_development)
    X_test  = ([one_hot_3mer(s) for s in pos_test] +
               [one_hot_3mer(s) for s in neg_test])
    y_test  = [1] * len(pos_test) + [0] * len(neg_test)
    seq_len = len(pos_development[0]) - 2
    return X_development, y_development, X_test, y_test, seq_len


def _scan_evaluate(model, loader, criterion):
    model.eval()
    y_true, y_pred, total_loss = [], [], 0.0
    with torch.no_grad():
        for seq, sc, lab in loader:
            seq, sc, lab = seq.to(DEVICE), sc.to(DEVICE), lab.to(DEVICE)
            out  = model(seq, sc)
            loss = criterion(out, lab)
            total_loss += loss.item() * len(lab)
            preds = (torch.sigmoid(out) > 0.5).long()
            y_true.extend(lab.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return accuracy_score(y_true, y_pred), total_loss / len(y_true)


def _scan_run_cv(fold_data, X_development_oh, y_development,
                 X_test_oh, y_test, seq_len, alpha):
    cfg = CONFIG["scan"]
    X_development_t = seq2tensor(X_development_oh, 64).to(DEVICE)
    y_development_t = torch.tensor(y_development, dtype=torch.float).to(DEVICE)
    X_test_t = seq2tensor(X_test_oh, 64).to(DEVICE)
    y_test_t = torch.tensor(y_test,  dtype=torch.float).to(DEVICE)

    train_accs, val_accs, test_accs = [], [], []
    criterion = nn.BCEWithLogitsLoss()

    for fd in fold_data:
        s_train_t = torch.tensor(fd['train_scores'], dtype=torch.float).to(DEVICE)
        s_val_t   = torch.tensor(fd['val_scores'],   dtype=torch.float).to(DEVICE)
        s_test_t  = torch.tensor(fd['test_scores'],  dtype=torch.float).to(DEVICE)

        tr = TensorDataset(X_development_t[fd['train_idx']], s_train_t,
                           y_development_t[fd['train_idx']])
        va = TensorDataset(X_development_t[fd['val_idx']], s_val_t,
                           y_development_t[fd['val_idx']])
        tr_loader = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True)
        va_loader = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False)

        model = SimpleCNN(seq_len, alpha).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"])
        sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3,
                                min_lr=1e-6)

        best_val, cnt, best_state = float('inf'), 0, None
        for _ in range(cfg["num_epochs"]):
            model.train()
            for xb, sb, yb in tr_loader:
                opt.zero_grad()
                loss = criterion(model(xb, sb), yb)
                loss.backward()
                opt.step()
            _, val_loss = _scan_evaluate(model, va_loader, criterion)
            sch.step(val_loss)
            if val_loss < best_val:
                best_val, cnt = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                cnt += 1
                if cnt >= cfg["patience"]:
                    break

        model.load_state_dict(best_state)
        train_acc, _ = _scan_evaluate(model, tr_loader, criterion)
        val_acc,   _ = _scan_evaluate(model, va_loader, criterion)
        te_loader = DataLoader(TensorDataset(X_test_t, s_test_t, y_test_t),
                               batch_size=cfg["batch_size"], shuffle=False)
        test_acc, _ = _scan_evaluate(model, te_loader, criterion)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    return np.mean(train_accs), np.mean(val_accs), np.mean(test_accs)


def run_alpha_scan():
    print("\n" + "=" * 70)
    print("Figure 2A/B - Alpha scan across sequence lengths")
    print("=" * 70)
    cfg = CONFIG["scan"]
    lengths = CONFIG["input"]["scan"]["lengths"]
    alpha_values = cfg["alpha_values"]

    pair_scores_by_length, shape_info_by_length = {}, {}
    print("=== Stage 1: Precomputing pairwise alignment matrices ===")
    for length in lengths:
        print(f"\nLength {length}...")
        M, npt, nnt, nptest, nntest = _scan_compute_pair_scores(length)
        pair_scores_by_length[length] = M
        shape_info_by_length[length]  = (npt, nnt, nptest, nntest)

    print("\n=== Stage 2: Alpha scanning for each length ===")
    all_results = {}
    for length in lengths:
        print(f"\n{'=' * 60}\nLength: {length} bp\n{'=' * 60}")
        X_development, y_development, X_test, y_test, seq_len = \
            _scan_load_dataset_onehot(length)
        M = pair_scores_by_length[length]
        npt, nnt, _, _ = shape_info_by_length[length]
        fold_data = _scan_precompute_fold_scores(M, npt, nnt, y_development)

        results = []
        for alpha in alpha_values:
            print(f"  alpha = {alpha} ...", end='', flush=True)
            tr_acc, va_acc, te_acc = _scan_run_cv(
                fold_data, X_development, y_development, X_test, y_test,
                seq_len, alpha)
            results.append((alpha, tr_acc, va_acc, te_acc))
            print(f" Train={tr_acc:.4f}, Val={va_acc:.4f}, Test={te_acc:.4f}")
        all_results[length] = results

    with open(SCAN_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header_lengths = []
        for length in lengths:
            header_lengths.extend([str(length), "", "", ""])
        writer.writerow(header_lengths)
        header_names = []
        for _ in lengths:
            header_names.extend(["alpha", "Train_ACC", "Val_ACC", "Test_ACC"])
        writer.writerow(header_names)
        for i in range(len(alpha_values)):
            row = []
            for length in lengths:
                a, tr, va, te = all_results[length][i]
                row.extend([a, f"{tr:.4f}", f"{va:.4f}", f"{te:.4f}"])
            writer.writerow(row)
    print(f"\n[Saved Figure 2A/B] {SCAN_CSV}")


# ============================================================================
# Figure 2C/D/E  -  ablation
# ============================================================================
def _abl_evaluate(model, loader, criterion):
    model.eval()
    y_true, y_prob, total_loss = [], [], 0.0
    with torch.no_grad():
        for seq, s, lab in loader:
            seq, s, lab = seq.to(DEVICE), s.to(DEVICE), lab.to(DEVICE)
            out = model(seq, s)
            loss = criterion(out, lab).mean()
            total_loss += loss.item()
            prob = torch.sigmoid(out)
            y_true.extend(lab.tolist())
            y_prob.extend(prob.tolist())
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return (total_loss / len(loader),
            np.array(y_true), np.array(y_prob), auc(fpr, tpr))


def _abl_train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for seq, s, lab in loader:
        seq, s, lab = seq.to(DEVICE), s.to(DEVICE), lab.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(seq, s), lab)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def _abl_smooth_roc(y_true, y_prob, n_points=500):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr_i = np.linspace(0, 1, n_points)
    tpr_i = np.interp(fpr_i, fpr, tpr)
    tpr_i[0], tpr_i[-1] = 0.0, 1.0
    return fpr_i, tpr_i


def run_ablation():
    print("\n" + "=" * 70)
    print("Figure 2C/D/E - Ablation (Full vs OneMer / SeqOnly / ScoreOnly)")
    print("=" * 70)
    cfg = CONFIG["ablation"]
    random_seed   = cfg["random_seed"]
    K_FOLDS       = cfg["k_folds"]
    NUM_EPOCHS    = cfg["num_epochs"]
    BATCH_SIZE    = cfg["batch_size"]
    LEARNING_RATE = cfg["lr"]
    WEIGHT_DECAY  = cfg["weight_decay"]
    PATIENCE      = cfg["patience"]
    SCHED_FACTOR  = cfg["scheduler_factor"]
    SCHED_PATIENT = cfg["scheduler_patience"]
    MIN_LR        = cfg["min_lr"]
    FINAL_SEEDS   = cfg["final_seeds"]

    set_seed(random_seed)

    pos_development_seq, pos_len = read_fasta(
        CONFIG["input"]["ablation"]["pos_development_fa"])
    neg_development_seq, neg_len = read_fasta(
        CONFIG["input"]["ablation"]["neg_development_fa"])
    assert pos_len == neg_len, "Development sequences length mismatch"
    pos_test_seq,  test_pos_len = read_fasta(
        CONFIG["input"]["ablation"]["pos_test_fa"])
    neg_test_seq,  test_neg_len = read_fasta(
        CONFIG["input"]["ablation"]["neg_test_fa"])
    assert test_pos_len == test_neg_len == pos_len, "Test sequence length mismatch"

    N_pos, N_neg = len(pos_development_seq), len(neg_development_seq)
    N_pos_test, N_neg_test = len(pos_test_seq), len(neg_test_seq)
    N_development = N_pos + N_neg
    N_test        = N_pos_test + N_neg_test
    print(f"Development pos/neg: {N_pos}/{N_neg}; "
          f"Test pos/neg: {N_pos_test}/{N_neg_test}")

    print("One-hot encoding ...")
    pos_development_3 = [one_hot_3mer(s) for s in pos_development_seq]
    neg_development_3 = [one_hot_3mer(s) for s in neg_development_seq]
    pos_development_1 = [one_hot_1mer(s) for s in pos_development_seq]
    neg_development_1 = [one_hot_1mer(s) for s in neg_development_seq]
    development_3mer = pos_development_3 + neg_development_3
    development_1mer = pos_development_1 + neg_development_1

    pos_test_3 = [one_hot_3mer(s) for s in pos_test_seq]
    neg_test_3 = [one_hot_3mer(s) for s in neg_test_seq]
    pos_test_1 = [one_hot_1mer(s) for s in pos_test_seq]
    neg_test_1 = [one_hot_1mer(s) for s in neg_test_seq]
    test_3mer = pos_test_3 + neg_test_3
    test_1mer = pos_test_1 + neg_test_1

    seq_len_3mer = pos_len - 2
    seq_len_1mer = pos_len

    zero_development_3 = [[[0.0] * 64 for _ in range(seq_len_3mer)]
                          for _ in range(N_development)]
    zero_test_3        = [[[0.0] * 64 for _ in range(seq_len_3mer)]
                          for _ in range(N_test)]

    development_labels = np.array([1] * N_pos + [0] * N_neg)
    test_labels        = np.array([1] * N_pos_test + [0] * N_neg_test)
    test_label_t       = torch.from_numpy(test_labels).float().to(DEVICE)

    print("Precomputing pairwise alignment scores ...")
    all_query = (pos_development_seq + neg_development_seq +
                 pos_test_seq + neg_test_seq)
    ref_seqs  = pos_development_seq
    pair_scores = precompute_pair_scores(all_query, ref_seqs)

    development_pair_max = np.max(pair_scores[:N_development, :], axis=1)
    valid = development_pair_max[
        np.isfinite(development_pair_max) & (development_pair_max >= 0)]
    if valid.size == 0:
        raise RuntimeError("No valid alignment score in Development set.")
    GLOBAL_MIN, GLOBAL_MAX = float(valid.min()), float(valid.max())
    GLOBAL_SCALE = (GLOBAL_MAX - GLOBAL_MIN) + 1e-8
    print(f"[Global norm] min={GLOBAL_MIN:.4f}, max={GLOBAL_MAX:.4f}")

    def normalize(v):
        return (v - GLOBAL_MIN) / GLOBAL_SCALE

    development_scores_final = normalize(
        np.where(np.isfinite(development_pair_max),
                 development_pair_max, -1.0).astype(np.float64))
    test_pair_max = np.max(pair_scores[N_development:, :], axis=1)
    test_scores_final = normalize(
        np.where(np.isfinite(test_pair_max),
                 test_pair_max, -1.0).astype(np.float64))

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                          random_state=random_seed)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.arange(N_development), development_labels)):
        pos_local = [int(i) for i in tr_idx if i < N_pos]
        if not pos_local:
            raise RuntimeError(f"Fold {fold + 1}: no positive in train split")
        ref_idx = np.array(pos_local, dtype=np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fold_max = np.max(pair_scores[:, ref_idx], axis=1)
        fold_max = np.where(np.isfinite(fold_max), fold_max, -1.0)
        fold_scores.append({
            'train_idx': tr_idx.copy(),
            'val_idx':   va_idx.copy(),
            'train': normalize(fold_max[tr_idx].astype(np.float64)),
            'val':   normalize(fold_max[va_idx].astype(np.float64)),
        })

    case_info = [
        {"name": "Full_3mer_Score", "short_name": "Full",
         "inch": 64, "use_score": True,
         "seq_data": development_3mer, "test_seq_data": test_3mer,
         "alpha": ALPHA, "in_len": seq_len_3mer,
         "color": "red", "compare_title": None},
        {"name": "OneMer_Score", "short_name": "OneMer",
         "inch": 4, "use_score": True,
         "seq_data": development_1mer, "test_seq_data": test_1mer,
         "alpha": ALPHA, "in_len": seq_len_1mer,
         "color": "#00bcfa", "compare_title": "Full vs 1mer+Score"},
        {"name": "Score_Only", "short_name": "ScoreOnly",
         "inch": 64, "use_score": True,
         "seq_data": zero_development_3, "test_seq_data": zero_test_3,
         "alpha": ALPHA, "in_len": seq_len_3mer,
         "color": "#00bcfa", "compare_title": "Full vs Score-Only"},
        {"name": "Sequence_Only", "short_name": "SeqOnly",
         "inch": 64, "use_score": False,
         "seq_data": development_3mer, "test_seq_data": test_3mer,
         "alpha": 0, "in_len": seq_len_3mer,
         "color": "#00bcfa", "compare_title": "Full vs Sequence-Only"},
    ]

    # ----- Stage A: CV to determine best epoch per variant -----
    print(f"\n{'#' * 72}\n# Stage A: 10-fold CV per variant (choose epoch)\n{'#' * 72}")
    variant_epochs = {}
    for case in case_info:
        name = case["name"]
        print(f"\n{'=' * 60}\n[Stage A] CV for {name}\n{'=' * 60}")
        all_seq_t = seq2tensor(case["seq_data"], case["inch"])
        label_t   = torch.from_numpy(development_labels).float().to(DEVICE)

        per_fold_epoch, per_fold_val_auc = [], []
        for fold in range(K_FOLDS):
            fs = fold_scores[fold]
            tr_idx, va_idx = fs['train_idx'], fs['val_idx']
            tr_scores_t = torch.from_numpy(fs['train']).float().to(DEVICE)
            va_scores_t = torch.from_numpy(fs['val']).float().to(DEVICE)
            tr_loader = DataLoader(TensorDataset(all_seq_t[tr_idx], tr_scores_t,
                                                 label_t[tr_idx]),
                                   batch_size=BATCH_SIZE, shuffle=True)
            va_loader = DataLoader(TensorDataset(all_seq_t[va_idx], va_scores_t,
                                                 label_t[va_idx]),
                                   batch_size=BATCH_SIZE, shuffle=False)

            set_seed(random_seed + fold)
            model = CNNCompare(case["inch"], case["use_score"],
                               case["alpha"], case["in_len"]).to(DEVICE)
            opt  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)
            crit = nn.BCEWithLogitsLoss()
            sch  = ReduceLROnPlateau(opt, mode='min', factor=SCHED_FACTOR,
                                     patience=SCHED_PATIENT, min_lr=MIN_LR)

            best_val, best_ep, best_state, cnt = float('inf'), 0, None, 0
            for ep in range(NUM_EPOCHS):
                _abl_train_one_epoch(model, tr_loader, opt, crit)
                va_loss, _, _, _ = _abl_evaluate(model, va_loader, crit)
                sch.step(va_loss)
                if va_loss < best_val:
                    best_val, best_ep, cnt = va_loss, ep + 1, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    cnt += 1
                    if cnt >= PATIENCE:
                        break

            model.load_state_dict(best_state)
            _, _, _, val_auc = _abl_evaluate(model, va_loader, crit)
            per_fold_epoch.append(best_ep)
            per_fold_val_auc.append(val_auc)
            print(f"  Fold {fold + 1:>2}: best_ep={best_ep:>2}, "
                  f"best_val_loss={best_val:.4f}, val_AUC={val_auc:.4f}")

        mean_ep = int(round(np.mean(per_fold_epoch)))
        variant_epochs[name] = mean_ep
        print(f"  >>> {name}: per-fold epochs = {per_fold_epoch}")
        print(f"  >>> mean = {mean_ep}, chosen for final = {mean_ep}")
        print(f"  >>> CV val AUC = {np.mean(per_fold_val_auc):.4f} "
              f"± {np.std(per_fold_val_auc):.4f}")

    # ----- Stage B: full-Development × N seeds per variant -----
    print(f"\n{'#' * 72}\n# Stage B: Full-Development × "
          f"{len(FINAL_SEEDS)} seeds per variant\n{'#' * 72}")
    test_true_all = None
    variant_test = {}

    for case in case_info:
        name     = case["name"]
        n_epochs = variant_epochs[name]
        print(f"\n{'=' * 60}\n[Stage B] {name}  "
              f"({n_epochs} epochs × {len(FINAL_SEEDS)} seeds)\n{'=' * 60}")

        all_seq_t  = seq2tensor(case["seq_data"], case["inch"])
        test_seq_t = seq2tensor(case["test_seq_data"], case["inch"])
        label_t    = torch.from_numpy(development_labels).float().to(DEVICE)
        dev_scores_t = torch.from_numpy(development_scores_final).float().to(DEVICE)
        te_scores_t  = torch.from_numpy(test_scores_final).float().to(DEVICE)

        development_loader = DataLoader(
            TensorDataset(all_seq_t, dev_scores_t, label_t),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        te_loader = DataLoader(
            TensorDataset(test_seq_t, te_scores_t, test_label_t),
            batch_size=BATCH_SIZE, shuffle=False)
        crit = nn.BCEWithLogitsLoss()

        seed_probs, seed_aucs = [], []
        for seed in FINAL_SEEDS:
            set_seed(seed)
            model = CNNCompare(case["inch"], case["use_score"],
                               case["alpha"], case["in_len"]).to(DEVICE)
            opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
            sch = CosineAnnealingLR(opt, T_max=n_epochs)
            for _ in range(n_epochs):
                _abl_train_one_epoch(model, development_loader, opt, crit)
                sch.step()

            _, y_t, p_t, test_auc = _abl_evaluate(model, te_loader, crit)
            seed_probs.append(p_t)
            seed_aucs.append(test_auc)
            if test_true_all is None:
                test_true_all = y_t
            print(f"  seed={seed:<7}: test AUC = {test_auc:.6f}")

        arr = np.array(seed_aucs)
        variant_test[name] = {
            'seed_probs':    seed_probs,
            'seed_aucs':     seed_aucs,
            'seed_auc_mean': float(arr.mean()),
            'seed_auc_std':  float(arr.std()),
        }
        print(f"  >>> {name}: AUC = {arr.mean():.4f} ± {arr.std():.4f}")

    # ----- Stage C: paired tests (for in-figure text box) -----
    print(f"\n{'#' * 72}\n# Stage C: Paired t-test / Wilcoxon\n{'#' * 72}")
    base_name = case_info[0]["name"]
    base_result = variant_test[base_name]
    paired = {}
    base_aucs = np.array(base_result['seed_aucs'])
    for case in case_info[1:]:
        name = case["name"]
        cmp_aucs = np.array(variant_test[name]['seed_aucs'])
        t_stat, p_t = stats.ttest_rel(base_aucs, cmp_aucs)
        try:
            _, p_w = stats.wilcoxon(base_aucs, cmp_aucs)
        except ValueError:
            p_w = np.nan
        paired[name] = {'p_t': float(p_t), 'p_w': float(p_w)}
        print(f"  Full vs {name}: paired t-test p = {p_t:.4e}, "
              f"Wilcoxon p = {p_w:.4e}")

    # ----- Draw Figure 2C/D/E -----
    plt.rcParams['font.size'] = 10
    base_seed_aucs = np.array(base_result['seed_aucs'])
    PLOT_SEED_IDX  = int(np.argmax(base_seed_aucs))
    PLOT_SEED      = FINAL_SEEDS[PLOT_SEED_IDX]
    print(f"\n[Plot 2C/D/E] Using max-AUC seed = {PLOT_SEED} "
          f"(Full seed-AUC = {base_seed_aucs[PLOT_SEED_IDX]:.6f})")

    for case in case_info[1:]:
        name  = case["name"]
        short = case["short_name"]
        title = case["compare_title"]

        p_base_seed = base_result['seed_probs'][PLOT_SEED_IDX]
        p_cmp_seed  = variant_test[name]['seed_probs'][PLOT_SEED_IDX]

        fpr_b, tpr_b = _abl_smooth_roc(test_true_all, p_base_seed)
        fpr_c, tpr_c = _abl_smooth_roc(test_true_all, p_cmp_seed)
        fpr_b_raw, tpr_b_raw, _ = roc_curve(test_true_all, p_base_seed)
        fpr_c_raw, tpr_c_raw, _ = roc_curve(test_true_all, p_cmp_seed)
        auc_b_seed = auc(fpr_b_raw, tpr_b_raw)
        auc_c_seed = auc(fpr_c_raw, tpr_c_raw)

        base_mean, base_std = (base_result['seed_auc_mean'],
                               base_result['seed_auc_std'])
        cmp_mean,  cmp_std  = (variant_test[name]['seed_auc_mean'],
                               variant_test[name]['seed_auc_std'])

        pt_p = paired[name]['p_t']
        pw_p = paired[name]['p_w']

        plt.figure(figsize=(7, 5.8), dpi=300)
        plt.plot(fpr_c, tpr_c, color=case["color"], lw=2.5, linestyle='-',
                 label=(f"{name}\n"
                        f"  seed-{PLOT_SEED} AUC={auc_c_seed:.4f}, "
                        f"mean±std={cmp_mean:.4f}±{cmp_std:.4f}"))
        plt.plot(fpr_b, tpr_b, color='red', lw=2.5,
                 label=(f"{base_name}\n"
                        f"  seed-{PLOT_SEED} AUC={auc_b_seed:.4f}, "
                        f"mean±std={base_mean:.4f}±{base_std:.4f}"))

        stats_text = (f"Paired t-test p = {pt_p:.3e}\n"
                      f"Wilcoxon   p = {pw_p:.3e}\n"
                      f"(n = {len(FINAL_SEEDS)} seeds)")
        plt.text(0.03, 0.03, stats_text,
                 transform=plt.gca().transAxes, fontsize=8.5,
                 va='bottom', ha='left',
                 bbox=dict(boxstyle='round,pad=0.35',
                           facecolor='white', edgecolor='gray', alpha=0.9))

        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(title, fontweight='bold')
        plt.legend(loc="lower right", framealpha=0.9, fontsize=8)
        plt.tight_layout()

        out_png = os.path.join(ABL_DIR, f"roc_Full_vs_{short}.png")
        plt.savefig(out_png, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[Saved] {out_png}")


# ============================================================================
# Figure 2F  -  Test_Set model comparison
# ============================================================================
def _load_attic_test_set(csv_path):
    df = pd.read_csv(csv_path)
    df['true_label']  = df['True Label'].astype(int)
    df['probability'] = df['Probability'].astype(float)
    return df[['true_label', 'probability']]


def _load_editpredict_test_set(txt_path):
    df = pd.read_csv(txt_path, sep='\t')
    df['true_label'] = df['True Label'].astype(int)
    if 'YES_prob' in df.columns and 'NO_prob' in df.columns:
        df['YES_prob'] = df['YES_prob'].fillna(1 - df['NO_prob'])
    df['probability'] = df['YES_prob'].astype(float)
    return df[['true_label', 'probability']].dropna()


def _predict_simplecnn(model_path, pos_fa, neg_fa,
                       pos_scores_csv, neg_scores_csv,
                       global_min, global_max):
    print(f"Using device: {DEVICE}")
    pos_seqs, pos_len = read_fasta(pos_fa)
    neg_seqs, neg_len = read_fasta(neg_fa)
    if pos_len != neg_len:
        raise ValueError("Positive/negative sequence length mismatch")
    seq_len = pos_len - 2

    pos_raw = np.loadtxt(pos_scores_csv)
    neg_raw = np.loadtxt(neg_scores_csv)
    print(f"[Normalization] Hardcoded range [{global_min:.4f}, {global_max:.4f}]")
    scale = global_max - global_min + 1e-8
    pos_scores = (pos_raw - global_min) / scale
    neg_scores = (neg_raw - global_min) / scale

    pos_oh = [one_hot_3mer(s) for s in pos_seqs]
    neg_oh = [one_hot_3mer(s) for s in neg_seqs]
    pos_t = sequences_to_tensor_oh(pos_oh, DEVICE)
    neg_t = sequences_to_tensor_oh(neg_oh, DEVICE)
    pos_sc_t = torch.tensor(pos_scores, dtype=torch.float, device=DEVICE)
    neg_sc_t = torch.tensor(neg_scores, dtype=torch.float, device=DEVICE)

    model = SimpleCNN(seq_len=seq_len, alpha=ALPHA).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pos_probs = torch.sigmoid(model(pos_t, pos_sc_t)).cpu().numpy()
        neg_probs = torch.sigmoid(model(neg_t, neg_sc_t)).cpu().numpy()

    pos_df = pd.DataFrame({'true_label': 1, 'probability': pos_probs})
    neg_df = pd.DataFrame({'true_label': 0, 'probability': neg_probs})
    return pd.concat([pos_df, neg_df], ignore_index=True)


def _plot_three_roc(ep_df, attic_df, cnn_df, save_path, cnn_label):
    plt.figure(figsize=(10, 8))
    fpr1, tpr1, auc1 = _roc_metrics(ep_df)
    fpr2, tpr2, auc2 = _roc_metrics(attic_df)
    fpr3, tpr3, auc3 = _roc_metrics(cnn_df)

    plt.plot(fpr1, tpr1, color='orange', lw=2,
             label=f'EditPredict (AUC = {auc1:.4f})')
    plt.plot(fpr2, tpr2, color='green', lw=2,
             label=f'ATTIC (AUC = {auc2:.4f})')
    plt.plot(fpr3, tpr3, color='red', lw=2,
             label=f'{cnn_label} (AUC = {auc3:.4f})')

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve Comparison of Three Models',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")
    print(f"  EditPredict AUC = {auc1:.4f}")
    print(f"  ATTIC       AUC = {auc2:.4f}")
    print(f"  {cnn_label:<11} AUC = {auc3:.4f}")


def _roc_metrics(df):
    fpr, tpr, _ = roc_curve(df['true_label'], df['probability'])
    return fpr, tpr, auc(fpr, tpr)


def run_test_set_compare():
    print("\n" + "=" * 70)
    print("Figure 2F - Model comparison on held-out test set")
    print("=" * 70)
    cfg = CONFIG["input"]["test_set"]
    for p in [cfg["attic_result"], cfg["editpredict_result"], cfg["model_path"],
              cfg["pos_test_fa"], cfg["neg_test_fa"],
              cfg["pos_scores_csv"], cfg["neg_scores_csv"]]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    print("Loading ATTIC result ...")
    attic_df = _load_attic_test_set(cfg["attic_result"])
    print("Loading EditPredict result ...")
    ep_df = _load_editpredict_test_set(cfg["editpredict_result"])

    print("Running CNN prediction on test set ...")
    cnn_df = _predict_simplecnn(
        cfg["model_path"], cfg["pos_test_fa"], cfg["neg_test_fa"],
        cfg["pos_scores_csv"], cfg["neg_scores_csv"],
        cfg["global_min"], cfg["global_max"])

    out_png = os.path.join(TEST_DIR, "roc_curves.png")
    _plot_three_roc(ep_df, attic_df, cnn_df, out_png, cnn_label='CNN Model')


# ============================================================================
# Figure 2G  -  Real FASTA model comparison
# ============================================================================
def _extract_interval(header):
    h = header.lstrip('>').split()[0]
    m = re.match(r'(chr\w+):(\d+)-(\d+)', h)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None


def _n_ratio(seq):
    return seq.upper().count('N') / len(seq) if seq else 1.0


def _load_editpredict_real(edit_file, snp_file):
    edit_df = pd.read_csv(edit_file, sep='\t', header=0)
    edit_df['true_label'] = 1
    edit_df['YES_prob'] = (edit_df['YES_prob'].fillna(1 - edit_df['NO_prob'])
                           if 'YES_prob' in edit_df.columns
                           else 1 - edit_df['NO_prob'])
    edit_df['probability'] = edit_df['YES_prob']
    edit_df['Chromosome'] = edit_df['Chromosome'].astype(str)
    edit_df['Position']   = edit_df['Position'].astype(int)

    snp_df = pd.read_csv(snp_file, sep='\t', header=0)
    snp_df['true_label'] = 0
    snp_df['YES_prob'] = (snp_df['YES_prob'].fillna(1 - snp_df['NO_prob'])
                          if 'YES_prob' in snp_df.columns
                          else 1 - snp_df['NO_prob'])
    snp_df['probability'] = snp_df['YES_prob']
    snp_df['Chromosome'] = snp_df['Chromosome'].astype(str)
    snp_df['Position']   = snp_df['Position'].astype(int)

    return pd.concat([edit_df, snp_df], ignore_index=True)[
        ['true_label', 'probability', 'Chromosome', 'Position']]


def _load_attic_real(edit_file, snp_file):
    edit_df = pd.read_csv(edit_file, header=0)
    edit_df['true_label'] = 1
    edit_df['probability'] = edit_df['Probability'].astype(float) \
        if 'Probability' in edit_df.columns else edit_df['Label'].astype(float)
    edit_df['header_info'] = edit_df['Header Info']

    snp_df = pd.read_csv(snp_file, header=0)
    snp_df['true_label'] = 0
    snp_df['probability'] = snp_df['Probability'].astype(float) \
        if 'Probability' in snp_df.columns else snp_df['Label'].astype(float)
    snp_df['header_info'] = snp_df['Header Info']
    return pd.concat([edit_df, snp_df], ignore_index=True)[
        ['true_label', 'probability', 'header_info']]


def _predict_real_fa(model_path, edit_fa, snp_fa,
                     edit_scores_csv, snp_scores_csv,
                     global_min, global_max, max_n_ratio):
    print(f"Using device: {DEVICE}")
    edit_seqs, edit_headers, edit_len = read_fasta(edit_fa, with_headers=True)
    snp_seqs,  snp_headers,  snp_len  = read_fasta(snp_fa,  with_headers=True)
    if edit_len != snp_len:
        raise ValueError("Edit/SNP FASTA length mismatch")
    seq_len = edit_len - 2

    edit_raw = np.loadtxt(edit_scores_csv)
    snp_raw  = np.loadtxt(snp_scores_csv)

    def _filter(seqs, headers, scores):
        keep = [i for i, s in enumerate(seqs) if _n_ratio(s) <= max_n_ratio]
        print(f"N-filter kept {len(keep)}/{len(seqs)}")
        return [seqs[i] for i in keep], [headers[i] for i in keep], scores[keep]

    edit_seqs, edit_headers, edit_raw = _filter(edit_seqs, edit_headers, edit_raw)
    snp_seqs,  snp_headers,  snp_raw  = _filter(snp_seqs,  snp_headers,  snp_raw)

    print(f"[Normalization] Hardcoded range [{global_min:.4f}, {global_max:.4f}]")
    scale = global_max - global_min + 1e-8
    edit_scores = (edit_raw - global_min) / scale
    snp_scores  = (snp_raw  - global_min) / scale

    edit_t = sequences_to_tensor_oh([one_hot_3mer(s) for s in edit_seqs], DEVICE)
    snp_t  = sequences_to_tensor_oh([one_hot_3mer(s) for s in snp_seqs],  DEVICE)
    edit_s_t = torch.tensor(edit_scores, dtype=torch.float, device=DEVICE)
    snp_s_t  = torch.tensor(snp_scores,  dtype=torch.float, device=DEVICE)

    model = SimpleCNN(seq_len=seq_len, alpha=ALPHA).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"[Model] Loaded: {model_path}")

    with torch.no_grad():
        edit_probs = torch.sigmoid(model(edit_t, edit_s_t)).cpu().numpy()
        snp_probs  = torch.sigmoid(model(snp_t, snp_s_t)).cpu().numpy()

    edit_df = pd.DataFrame({'header': edit_headers, 'true_label': 1,
                            'probability': edit_probs})
    snp_df  = pd.DataFrame({'header': snp_headers, 'true_label': 0,
                            'probability': snp_probs})
    return pd.concat([edit_df, snp_df], ignore_index=True)[
        ['header', 'true_label', 'probability']]


def _filter_editpredict_by_headers(ep_df, cnn_headers):
    valid = set()
    for h in cnn_headers:
        m = re.match(r'(chr\w+:\d+-\d+)', h.lstrip('>').split()[0])
        if m:
            valid.add(m.group(1))

    def _interval(row):
        return f"chr{row['Chromosome']}:{int(row['Position'])-55}-{int(row['Position'])+55}"
    mask = ep_df.apply(_interval, axis=1).isin(valid)
    return ep_df[mask][['true_label', 'probability']]


def _filter_attic_by_headers(attic_df, cnn_headers):
    valid = set()
    for h in cnn_headers:
        m = re.match(r'(chr\w+:\d+-\d+)', h.lstrip('>').split()[0])
        if m:
            valid.add(m.group(1))

    def _interval(s):
        m = re.match(r'(chr\w+:\d+-\d+)', str(s))
        return m.group(1) if m else None
    mask = attic_df['header_info'].apply(lambda x: _interval(x) in valid)
    return attic_df[mask].drop(columns='header_info')


def run_real_fa_compare():
    print("\n" + "=" * 70)
    print("Figure 2G - Model comparison on external real FASTA set")
    print("=" * 70)
    cfg = CONFIG["input"]["real_fa"]
    for p in [cfg["editpredict_edit"], cfg["editpredict_snp"],
              cfg["attic_edit"], cfg["attic_snp"], cfg["model_path"],
              cfg["edit_fa"], cfg["snp_fa"],
              cfg["edit_scores_csv"], cfg["snp_scores_csv"]]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    print("Loading EditPredict / ATTIC raw data ...")
    ep_raw = _load_editpredict_real(cfg["editpredict_edit"], cfg["editpredict_snp"])
    attic_raw = _load_attic_real(cfg["attic_edit"], cfg["attic_snp"])

    print("Running CNN prediction on real FASTA ...")
    cnn_df = _predict_real_fa(cfg["model_path"], cfg["edit_fa"], cfg["snp_fa"],
                              cfg["edit_scores_csv"], cfg["snp_scores_csv"],
                              cfg["global_min"], cfg["global_max"],
                              cfg["max_n_ratio"])

    valid_headers = cnn_df['header'].tolist()
    print("Aligning EditPredict / ATTIC to the CNN kept set ...")
    ep_df    = _filter_editpredict_by_headers(ep_raw, valid_headers)
    attic_df = _filter_attic_by_headers(attic_raw, valid_headers)
    cnn_df   = cnn_df[['true_label', 'probability']]

    print(f"\nFiltered sizes: EditPredict={len(ep_df)}, "
          f"ATTIC={len(attic_df)}, CNN={len(cnn_df)}")

    out_png = os.path.join(REAL_DIR, "roc_curves.png")
    _plot_three_roc(ep_df, attic_df, cnn_df, out_png, cnn_label='EditFinder')


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Figure 2 pipeline (A/B/C/D/E/F/G)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_ROOT}")

    try:
        run_alpha_scan()        # -> Figure 2A/B (CSV)
        run_ablation()          # -> Figure 2C/D/E (PNG)
        run_test_set_compare()  # -> Figure 2F (PNG)
        run_real_fa_compare()   # -> Figure 2G (PNG)
        print("\n=== All Figure 2 panels completed ===")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()