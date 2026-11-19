#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 pipeline (C/D/E)

[Figure to output mapping]
    Figure 1C: final_epoch_loss.csv  (data source for plotting loss curve by excel)
    Figure 1D: ROC curve (Development + Test) of the median-AUC seed
    Figure 1E: Full-sequence attribution logo with top motif highlighted

[External data (must be prepared in advance; fill paths in CONFIG['input'])]
    pos_development_fa / neg_development_fa : Development FASTA files (positive / negative)
    pos_test_fa  / neg_test_fa  : Test FASTA files (positive / negative)
    Python packages: torch, numpy, pandas, matplotlib, seaborn, scikit-learn,
                     biopython, captum, logomaker

[Figure to file mapping]
    Figure 1C: <out_dir>/final_epoch_loss.csv
    Figure 1D: <out_dir>/final_roc_curve.png
    Figure 1E: <out_dir>/DeepLIFT/logo_highlight_alpha<alpha>.png
"""

import csv
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from Bio.Align import PairwiseAligner

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG  (replace every path below with your own before running)
# ============================================================================
CONFIG = {
    "input": {
        "pos_development_fa": "/path/to/Input_data/development.fa",
        "neg_development_fa": "/path/to/Input_data/development2.fa",
        "pos_test_fa":  "/path/to/Input_data/test.fa",
        "neg_test_fa":  "/path/to/Input_data/test2.fa",
    },
    "output": {
        "out_dir": "/path/to/Model_final",
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
    "development": {
        "num_epochs":    32,
        "batch_size":    128,
        "learning_rate": 1e-4,
        "weight_decay":  1e-5,
        "seeds": [42, 0, 1, 2, 7, 13, 314, 2718, 111, 222, 333, 444, 555, 666,
                  777, 888, 999, 123, 1234, 12345, 123456, 3407, 2020, 2021,
                  2022, 2023, 2024, 2025, 2026, 2027],
    },
    "align": {
        "match":      1,
        "mismatch":   0,
        "open_gap":   -10,
        "extend_gap": -0.1,
        "mode":       "global",
    },
    "deeplift": {
        "model_seed":      333,
        "baseline_source": "all_development",
        "explain_source":  "test",
        "num_explain":     500,
        "window_size":     9,
        "center_idx":      55,
        "positive_only":   True,
        "motif_window":    10,
        "search_radius":   55,
    },
}


# ============================================================================
# Derived paths and parameters
# ============================================================================
POS_DEVELOPMENT_FA = CONFIG["input"]["pos_development_fa"]
NEG_DEVELOPMENT_FA = CONFIG["input"]["neg_development_fa"]
POS_TEST_FA  = CONFIG["input"]["pos_test_fa"]
NEG_TEST_FA  = CONFIG["input"]["neg_test_fa"]

OUT_DIR      = CONFIG["output"]["out_dir"]
MODEL_DIR    = os.path.join(OUT_DIR, "final_model")
DEEPLIFT_DIR = os.path.join(OUT_DIR, "DeepLIFT")

FIG1D_PNG = os.path.join(OUT_DIR, "final_roc_curve.png")
FIG1E_PNG = os.path.join(DEEPLIFT_DIR,
                         f"logo_highlight_alpha{CONFIG['model']['alpha']}.png")

SEED_RESULTS_CSV = os.path.join(OUT_DIR, "final_seed_results.csv")
SEED_LOSS_CSV    = os.path.join(OUT_DIR, "final_seed_loss.csv")
EPOCH_LOSS_CSV   = os.path.join(OUT_DIR, "final_epoch_loss.csv")
META_TXT         = os.path.join(OUT_DIR, "final_meta.txt")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEEPLIFT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
ALPHA              = CONFIG["model"]["alpha"]
CONV1_OUT_CHANNELS = CONFIG["model"]["conv1_out_channels"]
CONV2_OUT_CHANNELS = CONFIG["model"]["conv2_out_channels"]
FC1_OUT_FEATURES   = CONFIG["model"]["fc1_out_features"]
FC2_OUT_FEATURES   = CONFIG["model"]["fc2_out_features"]
DROPOUT_CONV       = CONFIG["model"]["dropout_conv"]
DROPOUT_FC         = CONFIG["model"]["dropout_fc"]

# Training hyperparameters
NUM_EPOCHS    = CONFIG["development"]["num_epochs"]
SEEDS         = CONFIG["development"]["seeds"]
BATCH_SIZE    = CONFIG["development"]["batch_size"]
LEARNING_RATE = CONFIG["development"]["learning_rate"]
WEIGHT_DECAY  = CONFIG["development"]["weight_decay"]

# Alignment parameters
ALIGN_MATCH      = CONFIG["align"]["match"]
ALIGN_MISMATCH   = CONFIG["align"]["mismatch"]
ALIGN_OPEN_GAP   = CONFIG["align"]["open_gap"]
ALIGN_EXTEND_GAP = CONFIG["align"]["extend_gap"]
ALIGN_MODE       = CONFIG["align"]["mode"]


# ============================================================================
# Utilities
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        seq = ''
        for line in file:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ''
                continue
            seq += line.strip()
        if seq:
            sequences.append(seq)
    return sequences


def one_hot_encode(seq):
    combinations = [''.join(n1 + n2 + n3) for n1 in 'ATGC' for n2 in 'ATGC' for n3 in 'ATGC']
    encoding = {comb: [1 if comb == triple else 0 for triple in combinations]
                for comb in combinations}
    three_mers = [seq[i:i + 3] for i in range(len(seq) - 2)]
    return [encoding.get(mer, [0] * 64) for mer in three_mers]


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


def precompute_pair_scores(query_seqs, ref_seqs, log_interval=1000):
    aligner = make_aligner()
    n_q = len(query_seqs)
    n_r = len(ref_seqs)
    M = np.full((n_q, n_r), -np.inf, dtype=np.float32)
    ref_norm = [t.strip().upper() for t in ref_seqs]

    print(f"Precomputing alignment scores: {n_q} queries x {n_r} refs = {n_q * n_r:,} alignments")
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
            print(f"  [{i + 1}/{n_q}] {rate:.1f} q/s, ETA {eta / 60:.1f} min", flush=True)
    print(f"Precompute done in {time.time() - t0:.1f}s")
    return M


def seqs_oh_to_tensor(oh_seqs, device):
    arr = np.stack([np.array(seq) if isinstance(seq[0], list) else np.zeros((len(seq), 64))
                    for seq in oh_seqs])
    return torch.tensor(arr, dtype=torch.float, device=device).permute(0, 2, 1)


def seqs_raw_to_tensor(raw_seqs):
    encoded = [one_hot_encode(seq) for seq in raw_seqs]
    arr = np.array(encoded, dtype=np.float32)
    return torch.tensor(arr).permute(0, 2, 1)


def metrics_from_probs(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    return acc, sens, spec, roc_auc, fpr, tpr


def mean_std_str(arr):
    return f"{np.mean(arr):.4f}±{np.std(arr):.4f}"


def evaluate_model_probs(model, loader, criterion=None):
    model.eval()
    y_true, y_probs = [], []
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for sequences, scores, labels in loader:
            sequences = sequences.to(DEVICE)
            scores    = scores.to(DEVICE)
            labels    = labels.to(DEVICE)

            outputs = model(sequences, scores)
            if criterion is not None:
                loss = criterion(outputs, labels)
                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_n    += bs

            probs = torch.sigmoid(outputs)
            y_true.extend(labels.tolist())
            y_probs.extend(probs.tolist())

    avg_loss = (total_loss / total_n) if total_n > 0 else float('nan')
    return np.array(y_true), np.array(y_probs), avg_loss


# ============================================================================
# Model
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, seq_len, alpha=ALPHA):
        super(SimpleCNN, self).__init__()
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
        x = self.fc3(x)
        return x.squeeze(1)


# ============================================================================
# Step 1: Training (Figure 1D + final_epoch_loss.csv for Figure 1C)
# ============================================================================
def run_training():
    print("\n" + "=" * 70)
    print("Development set training - Figure 1D and final_epoch_loss.csv (for Figure 1C)")
    print("=" * 70)
    print(f"Using device: {DEVICE}")

    # 1. Read sequences
    pos_development_sequences = read_fasta(POS_DEVELOPMENT_FA)
    neg_development_sequences = read_fasta(NEG_DEVELOPMENT_FA)
    pos_test_sequences  = read_fasta(POS_TEST_FA)
    neg_test_sequences  = read_fasta(NEG_TEST_FA)

    lengths = [len(pos_development_sequences[0]), len(neg_development_sequences[0]),
               len(pos_test_sequences[0]),  len(neg_test_sequences[0])]
    if not all(x == lengths[0] for x in lengths):
        raise ValueError("Sequences have different lengths.")
    seq_len = lengths[0] - 2

    N_pos      = len(pos_development_sequences)
    N_neg      = len(neg_development_sequences)
    N_pos_test = len(pos_test_sequences)
    N_neg_test = len(neg_test_sequences)
    N_development    = N_pos + N_neg

    print(f"Development positive: {N_pos}, negative: {N_neg}")
    print(f"Test positive       : {N_pos_test}, negative: {N_neg_test}")

    # 2. One-hot
    print("One-hot encoding sequences ...")
    pos_development_seq_oh = [one_hot_encode(s) for s in pos_development_sequences]
    neg_development_seq_oh = [one_hot_encode(s) for s in neg_development_sequences]
    pos_test_seq_oh  = [one_hot_encode(s) for s in pos_test_sequences]
    neg_test_seq_oh  = [one_hot_encode(s) for s in neg_test_sequences]

    all_development_seq_oh = pos_development_seq_oh + neg_development_seq_oh
    all_test_seq_oh  = pos_test_seq_oh + neg_test_seq_oh
    all_development_labels = [1] * N_pos + [0] * N_neg
    all_test_labels  = [1] * N_pos_test + [0] * N_neg_test

    # 3. Pair scores
    all_development_raw  = pos_development_sequences + neg_development_sequences
    all_test_raw   = pos_test_sequences + neg_test_sequences
    all_query_seqs = all_development_raw + all_test_raw
    ref_seqs       = pos_development_sequences
    pair_scores    = precompute_pair_scores(all_query_seqs, ref_seqs)

    # 4. Global score normalization
    all_pos_indices = np.arange(N_pos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        global_max_scores = np.max(pair_scores[:, all_pos_indices], axis=1)
    global_max_scores = np.where(np.isfinite(global_max_scores),
                                 global_max_scores, -1.0).astype(np.float64)

    development_scores_raw = global_max_scores[:N_development]
    development_valid      = development_scores_raw[development_scores_raw >= 0]
    if development_valid.size == 0:
        raise RuntimeError("No valid alignment score in Development set; cannot normalize.")

    global_min = float(development_valid.min())
    global_max = float(development_valid.max())
    scale      = (global_max - global_min) + 1e-8

    development_scores_norm = (development_scores_raw - global_min) / scale
    test_scores_raw   = global_max_scores[N_development:]
    test_scores_norm  = (test_scores_raw - global_min) / scale

    print(f"[Global score normalization] min={global_min:.4f}, max={global_max:.4f}, "
          f"scale={scale:.4f}")
    print(f"  valid Development scores: {development_valid.size}/{N_development}")

    # 5. DataLoaders
    X_development_t = seqs_oh_to_tensor(all_development_seq_oh, DEVICE)
    y_development_t = torch.tensor(all_development_labels, dtype=torch.float, device=DEVICE)
    s_development_t = torch.tensor(development_scores_norm, dtype=torch.float, device=DEVICE)

    X_test_t  = seqs_oh_to_tensor(all_test_seq_oh, DEVICE)
    y_test_t  = torch.tensor(all_test_labels, dtype=torch.float, device=DEVICE)
    s_test_t  = torch.tensor(test_scores_norm, dtype=torch.float, device=DEVICE)

    development_loader = DataLoader(TensorDataset(X_development_t, s_development_t, y_development_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_t,  s_test_t,  y_test_t),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 6. Multi-seed training
    criterion = nn.BCEWithLogitsLoss()
    per_seed_results = []
    per_seed_epoch_losses = {}

    for seed in SEEDS:
        print(f"\n===== Training with seed={seed}  ({NUM_EPOCHS} epochs) =====")
        set_seed(seed)

        model = SimpleCNN(seq_len=seq_len, alpha=ALPHA).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        epoch_losses_this_seed = []

        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for xb, sb, yb in development_loader:
                optimizer.zero_grad()
                out = model(xb, sb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            avg_epoch_loss = epoch_loss / len(development_loader)
            epoch_losses_this_seed.append(avg_epoch_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:>2}/{NUM_EPOCHS} | "
                      f"loss={avg_epoch_loss:.4f}")

        per_seed_epoch_losses[seed] = epoch_losses_this_seed

        model_path = os.path.join(MODEL_DIR, f"final_model_seed{seed}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  Saved model -> {model_path}")

        y_tr_true, y_tr_probs, tr_loss = evaluate_model_probs(model, development_loader, criterion)
        tr_acc, tr_sens, tr_spec, tr_auc, tr_fpr, tr_tpr = \
            metrics_from_probs(y_tr_true, y_tr_probs)

        y_te_true, y_te_probs, te_loss = evaluate_model_probs(model, test_loader, criterion)
        te_acc, te_sens, te_spec, te_auc, te_fpr, te_tpr = \
            metrics_from_probs(y_te_true, y_te_probs)

        per_seed_results.append({
            'seed': seed,
            'development_acc':  tr_acc,  'development_sens': tr_sens,
            'development_spec': tr_spec, 'development_auc':  tr_auc,
            'development_loss': tr_loss,
            'test_acc':   te_acc,  'test_sens':  te_sens,
            'test_spec':  te_spec, 'test_auc':   te_auc,
            'test_loss':  te_loss,
            'development_fpr': tr_fpr, 'development_tpr': tr_tpr,
            'test_fpr':  te_fpr, 'test_tpr':  te_tpr,
        })

        print(f"  Development -> Acc={tr_acc:.4f}, SE={tr_sens:.4f}, "
              f"SP={tr_spec:.4f}, AUC={tr_auc:.4f}, Loss={tr_loss:.4f}")
        print(f"  Test        -> Acc={te_acc:.4f}, SE={te_sens:.4f}, "
              f"SP={te_spec:.4f}, AUC={te_auc:.4f}, Loss={te_loss:.4f}")

    # 7. Summary
    development_accs   = [r['development_acc']  for r in per_seed_results]
    development_senss  = [r['development_sens'] for r in per_seed_results]
    development_specs  = [r['development_spec'] for r in per_seed_results]
    development_aucs   = [r['development_auc']  for r in per_seed_results]
    development_losses = [r['development_loss'] for r in per_seed_results]

    test_accs    = [r['test_acc']   for r in per_seed_results]
    test_senss   = [r['test_sens']  for r in per_seed_results]
    test_specs   = [r['test_spec']  for r in per_seed_results]
    test_aucs    = [r['test_auc']   for r in per_seed_results]
    test_losses  = [r['test_loss']  for r in per_seed_results]

    print(f"\n===== Summary (n={len(SEEDS)} seeds) =====")
    print(f"  Development: Acc = {mean_std_str(development_accs)} | "
          f"SE = {mean_std_str(development_senss)} | "
          f"SP = {mean_std_str(development_specs)} | "
          f"AUC = {mean_std_str(development_aucs)} | "
          f"Loss = {mean_std_str(development_losses)}")
    print(f"  Test       : Acc = {mean_std_str(test_accs)} | "
          f"SE = {mean_std_str(test_senss)} | "
          f"SP = {mean_std_str(test_specs)} | "
          f"AUC = {mean_std_str(test_aucs)} | "
          f"Loss = {mean_std_str(test_losses)}")

    # 8. Save CSVs (final_epoch_loss.csv is the data source for Figure 1C)
    with open(SEED_RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Seed', 'Test_Acc', 'Test_Sens', 'Test_Spec', 'Test_AUC'])
        for r in per_seed_results:
            writer.writerow([r['seed'],
                             f"{r['test_acc']:.4f}",
                             f"{r['test_sens']:.4f}",
                             f"{r['test_spec']:.4f}",
                             f"{r['test_auc']:.4f}"])
        writer.writerow(['Mean±Std',
                         mean_std_str(test_accs),
                         mean_std_str(test_senss),
                         mean_std_str(test_specs),
                         mean_std_str(test_aucs)])
    print(f"Seed results saved to {SEED_RESULTS_CSV}")

    with open(SEED_LOSS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Seed', 'Development_Loss', 'Test_Loss'])
        for r in per_seed_results:
            writer.writerow([r['seed'],
                             f"{r['development_loss']:.4f}",
                             f"{r['test_loss']:.4f}"])
        writer.writerow(['Mean±Std',
                         mean_std_str(development_losses),
                         mean_std_str(test_losses)])
    print(f"Seed loss (final) saved to {SEED_LOSS_CSV}")

    # ★ final_epoch_loss.csv - data source for Figure 1C
    with open(EPOCH_LOSS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Seed', 'Epoch', 'Development_Loss'])
        for seed in SEEDS:
            for ep, l in enumerate(per_seed_epoch_losses[seed], start=1):
                writer.writerow([seed, ep, f"{l:.4f}"])
    print(f"Epoch-by-epoch Development loss saved to {EPOCH_LOSS_CSV}")

    with open(META_TXT, 'w') as f:
        f.write(f"alpha = {ALPHA}\n")
        f.write(f"seq_len = {seq_len}\n")
        f.write(f"num_epochs = {NUM_EPOCHS}\n")
        f.write(f"seeds = {SEEDS}\n")
        f.write(f"norm_scope = GLOBAL (all Development samples)\n")
        f.write(f"global_min = {global_min:.6f}\n")
        f.write(f"global_max = {global_max:.6f}\n")
        f.write(f"n_development_pos = {N_pos}\n")
        f.write(f"n_development_neg = {N_neg}\n")
        f.write(f"n_test_pos  = {N_pos_test}\n")
        f.write(f"n_test_neg  = {N_neg_test}\n")
        f.write(f"model_dir   = {MODEL_DIR}\n")
    print(f"Meta saved to {META_TXT}")

    # 9. Figure 1D: ROC of median-AUC seed (Development + Test)
    test_aucs_arr = np.array(test_aucs)
    sorted_idx    = np.argsort(test_aucs_arr)
    plot_idx      = int(sorted_idx[len(sorted_idx) // 2])
    plot_seed     = SEEDS[plot_idx]
    plot_result   = per_seed_results[plot_idx]

    plot_seed_development_auc = plot_result['development_auc']
    plot_seed_test_auc  = plot_result['test_auc']

    print(f"\n[Plot 1D] median-AUC seed = {plot_seed} "
          f"(Development AUC = {plot_seed_development_auc:.4f}, "
          f"test AUC = {plot_seed_test_auc:.4f}, "
          f"test mean±std = {mean_std_str(test_aucs)})")

    plt.rcParams['font.size'] = 9
    plt.figure(figsize=(7.2, 6.4), dpi=300)

    plt.plot(
        plot_result['development_fpr'], plot_result['development_tpr'],
        color='#1f77b4', lw=2.5, linestyle='-',
        label=(f"Development ROC (seed={plot_seed})\n"
               f"AUC = {plot_seed_development_auc:.4f}\n"
               f"[mean±std = {mean_std_str(development_aucs)}]"),
    )
    plt.plot(
        plot_result['test_fpr'], plot_result['test_tpr'],
        color='red', lw=2.5, linestyle='-',
        label=(f"Test ROC (seed={plot_seed})\n"
               f"AUC = {plot_seed_test_auc:.4f}\n"
               f"[mean±std = {mean_std_str(test_aucs)}]"),
    )

    metrics_text = (
        "All seeds (n = {n})\n"
        "──────────────────\n"
        "Development:\n"
        "  ACC = {tr_acc}\n"
        "  SE  = {tr_se}\n"
        "  SP  = {tr_sp}\n"
        "  AUC = {tr_auc}\n"
        "Test:\n"
        "  ACC = {te_acc}\n"
        "  SE  = {te_se}\n"
        "  SP  = {te_sp}\n"
        "  AUC = {te_auc}"
    ).format(
        n      = len(SEEDS),
        tr_acc = mean_std_str(development_accs),
        tr_se  = mean_std_str(development_senss),
        tr_sp  = mean_std_str(development_specs),
        tr_auc = mean_std_str(development_aucs),
        te_acc = mean_std_str(test_accs),
        te_se  = mean_std_str(test_senss),
        te_sp  = mean_std_str(test_specs),
        te_auc = mean_std_str(test_aucs),
    )

    plt.text(0.02, 0.02, metrics_text,
             transform=plt.gca().transAxes,
             fontsize=7.5, va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='white', edgecolor='gray', alpha=0.9))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title(f'Final Model — ROC (median-AUC seed = {plot_seed})',
              fontsize=10, fontweight='bold')
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG1D_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved Fig 1D] {FIG1D_PNG}")


# ============================================================================
# Step 2: Figure 1E — DeepLIFT full-sequence logo with top motif highlighted
# ============================================================================
def run_deeplift_logo():
    print("\n" + "=" * 70)
    print("Figure 1E - Full-sequence attribution logo with top motif")
    print("=" * 70)
    from captum.attr import DeepLift
    from logomaker import Logo

    cfg_dl = CONFIG["deeplift"]
    model_seed      = cfg_dl["model_seed"]
    baseline_source = cfg_dl["baseline_source"]
    explain_source  = cfg_dl["explain_source"]
    NUM_EXPLAIN     = cfg_dl["num_explain"]
    WINDOW_SIZE     = cfg_dl["window_size"]
    CENTER_IDX      = cfg_dl["center_idx"]
    POSITIVE_ONLY   = cfg_dl["positive_only"]
    motif_window    = cfg_dl["motif_window"]
    search_radius   = cfg_dl["search_radius"]
    INFERENCE_ALPHA = ALPHA

    model_path = os.path.join(MODEL_DIR, f"final_model_seed{model_seed}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run training first.")

    # 1. Load sequences
    print("Loading sequences ...")
    pos_development_seq_raw = read_fasta(POS_DEVELOPMENT_FA)
    neg_development_seq_raw = read_fasta(NEG_DEVELOPMENT_FA)
    pos_test_seq_raw  = read_fasta(POS_TEST_FA)
    neg_test_seq_raw  = read_fasta(NEG_TEST_FA)

    N_pos      = len(pos_development_seq_raw)
    N_neg      = len(neg_development_seq_raw)
    N_pos_test = len(pos_test_seq_raw)
    N_neg_test = len(neg_test_seq_raw)
    N_development    = N_pos + N_neg
    N_test     = N_pos_test + N_neg_test

    pos_len = len(pos_development_seq_raw[0])
    assert all(len(s) == pos_len for s in
               pos_development_seq_raw + neg_development_seq_raw +
               pos_test_seq_raw + neg_test_seq_raw), "Sequence length mismatch."
    SEQ_LEN = pos_len
    N_3MERS = pos_len - 2
    print(f"SEQ_LEN={SEQ_LEN}, N_3MERS={N_3MERS}, N_development={N_development}, N_test={N_test}")

    development_seq_raw = pos_development_seq_raw + neg_development_seq_raw
    development_seq_tensor = seqs_raw_to_tensor(development_seq_raw)

    # 2. Dynamic pair scores (matching training)
    print("Computing alignment scores dynamically ...")
    all_query_seqs = (pos_development_seq_raw + neg_development_seq_raw +
                      pos_test_seq_raw + neg_test_seq_raw)
    ref_seqs = pos_development_seq_raw
    pair_scores = precompute_pair_scores(all_query_seqs, ref_seqs)

    all_pos_indices = np.arange(N_pos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        global_max_scores = np.max(pair_scores[:, all_pos_indices], axis=1)
    global_max_scores = np.where(np.isfinite(global_max_scores),
                                 global_max_scores, -1.0).astype(np.float64)

    development_scores_raw = global_max_scores[:N_development]
    development_valid      = development_scores_raw[development_scores_raw >= 0]
    assert development_valid.size > 0, "No valid score in Development set for normalization."

    mn    = float(development_valid.min())
    mx    = float(development_valid.max())
    scale = (mx - mn) + 1e-8

    development_scores_norm = (development_scores_raw - mn) / scale
    test_scores_raw   = global_max_scores[N_development:]
    test_scores_norm  = (test_scores_raw - mn) / scale

    development_scores_tensor = torch.tensor(development_scores_norm, dtype=torch.float32)
    test_scores_tensor  = torch.tensor(test_scores_norm,  dtype=torch.float32)

    print(f"[Global score normalization] min={mn:.4f}, max={mx:.4f}, scale={scale:.4f}")

    # 3. Baseline
    if baseline_source == "all_development":
        baseline_seq   = development_seq_tensor.mean(dim=0, keepdim=True).to(DEVICE)
        baseline_score = development_scores_tensor.mean().unsqueeze(0).to(DEVICE)
    else:
        raise ValueError(f"Unknown BASELINE_SOURCE: {baseline_source}")

    print(f"Baseline source: {baseline_source}, baseline_score = {baseline_score.item():.6f}")

    # 4. Explain set
    if explain_source == "test":
        explain_seq_tensor    = seqs_raw_to_tensor(pos_test_seq_raw).to(DEVICE)
        explain_scores_tensor = test_scores_tensor[:N_pos_test].to(DEVICE)
        explain_raw_seq       = pos_test_seq_raw
    elif explain_source == "development_pos":
        explain_seq_tensor    = development_seq_tensor[:N_pos].to(DEVICE)
        explain_scores_tensor = development_scores_tensor[:N_pos].to(DEVICE)
        explain_raw_seq       = pos_development_seq_raw
    else:
        raise ValueError(f"Unknown EXPLAIN_SOURCE: {explain_source}")

    print(f"Explain source: {explain_source}, available = {len(explain_raw_seq)}")

    # 5. Load model
    print(f"Loading pretrained model from {model_path} ...")
    model = SimpleCNN(seq_len=N_3MERS, alpha=ALPHA).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model.alpha = INFERENCE_ALPHA
    print(f"Model alpha = {model.alpha}")

    # 6. DeepLIFT
    num_explain = min(NUM_EXPLAIN, len(explain_raw_seq))
    explain_seq     = explain_seq_tensor[:num_explain]
    explain_scores  = explain_scores_tensor[:num_explain]
    explain_raw_seq = explain_raw_seq[:num_explain]

    baseline_seqs_expanded   = baseline_seq.repeat(num_explain, 1, 1)
    baseline_scores_expanded = baseline_score.repeat(num_explain)

    print(f"Running DeepLIFT on {num_explain} samples ...")
    dl = DeepLift(model)
    attr = dl.attribute(inputs=(explain_seq, explain_scores),
                        baselines=(baseline_seqs_expanded, baseline_scores_expanded))
    seq_attr   = attr[0].detach().cpu().numpy()
    score_attr = attr[1].detach().cpu().numpy()

    print(f"Score attribution abs mean:    {np.abs(score_attr).mean():.6f}")
    print(f"Sequence attribution abs mean: {np.abs(seq_attr).mean():.6f}")

    # 7. Completeness check
    with torch.no_grad():
        out_input = model(explain_seq, explain_scores).detach().cpu().numpy()
        out_base  = model(baseline_seqs_expanded, baseline_scores_expanded).detach().cpu().numpy()
    delta_output = out_input - out_base
    total_attr   = seq_attr.sum(axis=(1, 2)) + score_attr
    errors       = total_attr - delta_output

    print("=" * 60)
    print("DeepLIFT Completeness Check")
    print(f"  Mean |error|        : {np.mean(np.abs(errors)):.6e}")
    print(f"  Max  |error|        : {np.max(np.abs(errors)):.6e}")
    print(f"  Mean |delta_output| : {np.mean(np.abs(delta_output)):.6e}")
    print("=" * 60)

    with open(os.path.join(DEEPLIFT_DIR, "completeness_check.txt"), "w") as f:
        f.write(f"model_path={model_path}\n")
        f.write(f"baseline_source={baseline_source}\n")
        f.write(f"explain_source={explain_source}\n")
        f.write(f"inference_alpha={INFERENCE_ALPHA}\n")
        f.write(f"num_explain={num_explain}\n")
        f.write(f"score_norm_min={mn:.6f}\n")
        f.write(f"score_norm_max={mx:.6f}\n")
        f.write(f"mean_abs_error={np.mean(np.abs(errors)):.6e}\n")
        f.write(f"max_abs_error={np.max(np.abs(errors)):.6e}\n")
        f.write(f"mean_abs_delta_output={np.mean(np.abs(delta_output)):.6e}\n")

    # 8. 3mer -> base attribution
    print("Converting 3mer attribution to single base level ...")
    all_base_scores = []
    for i in range(num_explain):
        inp_np = explain_seq[i].detach().cpu().numpy()
        triplet_contrib = (seq_attr[i] * inp_np).sum(axis=0)
        if POSITIVE_ONLY:
            triplet_contrib = np.maximum(triplet_contrib, 0.0)

        base_scores = np.zeros(SEQ_LEN)
        base_cnt    = np.zeros(SEQ_LEN)
        for j in range(N_3MERS):
            val = triplet_contrib[j]
            base_scores[j]     += val
            base_scores[j + 1] += val
            base_scores[j + 2] += val
            base_cnt[j]     += 1
            base_cnt[j + 1] += 1
            base_cnt[j + 2] += 1
        mask = base_cnt > 0
        base_scores[mask] = base_scores[mask] / base_cnt[mask]
        all_base_scores.append(base_scores)
    all_base_scores = np.array(all_base_scores)

    # 9. PWM
    nuc = ['A', 'C', 'G', 'T']
    pwm = np.zeros((SEQ_LEN, 4))
    for seq, contrib in zip(explain_raw_seq, all_base_scores):
        for pos, b in enumerate(seq):
            if b in nuc:
                pwm[pos, nuc.index(b)] += contrib[pos]

    # 10. Top motif window search
    print("Locating top motif window ...")
    start_min = max(0, CENTER_IDX - search_radius)
    start_max = min(SEQ_LEN - motif_window, CENTER_IDX + search_radius) + 1

    window_avg_scores = np.zeros(start_max - start_min)
    for scores in all_base_scores:
        for start in range(start_min, start_max):
            end = start + motif_window
            window_avg_scores[start - start_min] += np.mean(scores[start:end])

    window_avg_scores /= len(all_base_scores)
    best_start = start_min + int(np.argmax(window_avg_scores))
    best_end   = best_start + motif_window

    positions      = np.arange(SEQ_LEN) - CENTER_IDX
    relative_start = best_start - CENTER_IDX
    relative_end   = best_end   - CENTER_IDX - 1
    print(f"Top motif window: abs {best_start}-{best_end} "
          f"(relative {relative_start} to {relative_end})")

    # 11. Draw full logo with highlight
    plt.figure(figsize=(20, 5))
    logo_hl = Logo(pd.DataFrame(pwm, columns=nuc),
                   color_scheme={'A': 'green', 'C': 'blue',
                                 'G': 'orange', 'T': 'red'})
    logo_hl.style_spines(visible=False)
    logo_hl.style_spines(spines=['left', 'bottom'], visible=True)
    logo_hl.ax.set_ylabel("Total Contribution")
    logo_hl.ax.set_xlabel("Relative Position")
    logo_hl.ax.set_title(
        f"Full Sequence Logo with Top Motif Highlight "
        f"(seed{model_seed}, explain={explain_source}, alpha={INFERENCE_ALPHA})")

    plt.xticks(np.arange(0, SEQ_LEN, 5), positions[::5])
    plt.axvline(x=CENTER_IDX, color='red', linestyle='--', alpha=0.7)
    plt.axvspan(best_start, best_end, color='yellow', alpha=0.3)
    plt.text((best_start + best_end) / 2, plt.ylim()[1] * 0.9,
             f'Top Motif ({relative_start} to {relative_end})',
             ha='center', color='darkorange', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG1E_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved Fig 1E] {FIG1E_PNG}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Figure 1 pipeline (C/D/E)")
    print("=" * 70)
    print(f"Device   : {DEVICE}")
    print(f"Output   : {OUT_DIR}")

    try:
        run_training()       # -> Figure 1D + final_epoch_loss.csv (for Fig 1C)
        run_deeplift_logo()  # -> Figure 1E
        print("\n=== All Figure 1 panels completed ===")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()