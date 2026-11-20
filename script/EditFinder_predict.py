"""
Usage
python EditFinder_predict.py \
    --fasta "/path/to/dataset.fa" \
    --scores "/path/to/alignment_scores.csv" \
    --model "/path/to/final_model_seed12345.pth" \
    --output_dir /path/to/output \
    --batch_size 1042 \
    --threshold 0.5
"""

import argparse
import os
import re
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing

ALPHA = 279
CONV1_OUT_CHANNELS = 128
CONV2_OUT_CHANNELS = 256
FC1_OUT_FEATURES = 256
FC2_OUT_FEATURES = 128
DROPOUT_CONV = 0.4
DROPOUT_FC = 0.7

GLOBAL_MIN = 41.0
GLOBAL_MAX = 105.0


class SimpleCNN(nn.Module):
    def __init__(self, seq_len, alpha=ALPHA):
        super(SimpleCNN, self).__init__()
        self.seq_len = seq_len
        self.alpha = alpha

        self.conv1 = nn.Conv1d(64, CONV1_OUT_CHANNELS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(CONV1_OUT_CHANNELS)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(DROPOUT_CONV)

        self.conv2 = nn.Conv1d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=3, padding=1)
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
        weighted_scores = scores * self.alpha
        weighted_scores = weighted_scores.unsqueeze(1)
        x = torch.cat((x, weighted_scores), dim=1)
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.dropout5(self.relu5(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze(1)


combinations = [''.join(n1 + n2 + n3) for n1 in 'ATGC' for n2 in 'ATGC' for n3 in 'ATGC']
encoding_dict = {comb: idx for idx, comb in enumerate(combinations)}


def encode_3mer(seq):
    L = len(seq)
    indices = np.array([encoding_dict.get(seq[i:i + 3], -1) for i in range(L - 2)], dtype=np.int64)
    onehot = np.zeros((L - 2, 64), dtype=np.float32)
    valid = indices >= 0
    onehot[valid, indices[valid]] = 1.0
    return onehot


def normalize_scores(scores, gmin, gmax):
    return (scores - gmin) / (gmax - gmin + 1e-8)


def parse_header(header):
    """
    header: '>chr1:10000-10111(+)'
    return (chrom, start, end, strand)
    """
    m = re.match(r'>(chr\w+):(\d+)-(\d+)\(([+-])\)', header)
    if not m:
        m = re.match(r'>(chr\w+):(\d+)-(\d+)', header)
        if m:
            return m.group(1), m.group(2), m.group(3), '+'
        raise ValueError(f"Unable to parse header: {header}")
    return m.group(1), m.group(2), m.group(3), m.group(4)


def fasta_score_iterator(fasta_path, scores_path):
    with open(fasta_path) as f_fa, open(scores_path) as f_sc:
        header = None
        seq = ''
        for line in f_fa:
            if line.startswith('>'):
                if header is not None:
                    try:
                        score = float(next(f_sc).strip())
                    except StopIteration:
                        break
                    yield header, seq, score
                header = line.strip()
                seq = ''
            else:
                seq += line.strip()
        if header is not None:
            try:
                score = float(next(f_sc).strip())
            except StopIteration:
                pass
            else:
                yield header, seq, score


def get_seq_length(fasta_path):
    with open(fasta_path) as f:
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq:
                    break
                continue
            seq += line.strip()
        return len(seq)


def count_sequences(fasta_path):
    return sum(1 for line in open(fasta_path) if line.startswith('>'))


def main():
    parser = argparse.ArgumentParser(description='EditFinder predict, output:edit_predict.bed and snp_predict.bed')
    parser.add_argument('--fasta', required=True, help='The FASTA file to be predicted')
    parser.add_argument('--scores', required=True, help='The corresponding alignment score file (with one score per line)')
    parser.add_argument('--model', required=True, help='Pre-trained model path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--num_threads', type=int, default=None, help='The number of CPU threads is set to use 20% of the CPU cores by default.')
    parser.add_argument('--threshold', type=float, default=0.5, help='The probability threshold for determining "edit" (default value 0.5)')
    args = parser.parse_args()

    if args.num_threads is None:
        num_cpus = multiprocessing.cpu_count()
        args.num_threads = max(1, int(num_cpus * 0.2))
    torch.set_num_threads(args.num_threads)
    print(f"Using {args.num_threads} CPU threads for inference")

    global_min = GLOBAL_MIN
    global_max = GLOBAL_MAX
    print(f"[Hardcoded] Normalization range (from main experiment): "
          f"[{global_min:.4f}, {global_max:.4f}]")

    seq_len_orig = get_seq_length(args.fasta)
    seq_len = seq_len_orig - 2
    print(f"Original seq length: {seq_len_orig}, conv input length: {seq_len}")

    device = torch.device('cpu')
    model = SimpleCNN(seq_len=seq_len, alpha=ALPHA).to(device)
    state_dict = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded successfully (alpha={ALPHA}) from {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)
    edit_bed = os.path.join(args.output_dir, 'edit_predictions.bed')
    snp_bed = os.path.join(args.output_dir, 'snp_predictions.bed')

    total_seqs = count_sequences(args.fasta)
    print(f"Total sequences in FASTA: {total_seqs}")

    data_iter = fasta_score_iterator(args.fasta, args.scores)

    progress = tqdm(total=total_seqs, desc="Predicting", unit="seq")

    with open(edit_bed, 'a') as f_edit, open(snp_bed, 'w') as f_snp:
        batch_headers = []
        batch_seqs = []
        batch_scores = []

        for header, seq, score in data_iter:
            batch_headers.append(header)
            batch_seqs.append(seq)
            batch_scores.append(score)

            if len(batch_headers) >= args.batch_size:
                onehot_batch = [encode_3mer(s) for s in batch_seqs]
                seq_tensor = torch.from_numpy(np.stack(onehot_batch)).permute(0, 2, 1)
                score_arr = np.array(batch_scores, dtype=np.float32)
                score_norm = normalize_scores(score_arr, global_min, global_max)
                score_tensor = torch.from_numpy(score_norm)

                with torch.no_grad():
                    outputs = model(seq_tensor, score_tensor)
                    probs = torch.sigmoid(outputs).numpy()

                for h, p in zip(batch_headers, probs):
                    chrom, start, end, strand = parse_header(h)
                    name = 'edit' if p >= args.threshold else 'snp'
                    line = f"{chrom}\t{start}\t{end}\t{name}\t{p:.6f}\t{strand}\n"
                    if name == 'edit':
                        f_edit.write(line)
                    else:
                        f_snp.write(line)

                f_edit.flush()
                f_snp.flush()
                progress.update(len(batch_headers))
                batch_headers = []
                batch_seqs = []
                batch_scores = []

        if batch_headers:
            onehot_batch = [encode_3mer(s) for s in batch_seqs]
            seq_tensor = torch.from_numpy(np.stack(onehot_batch)).permute(0, 2, 1)
            score_arr = np.array(batch_scores, dtype=np.float32)
            score_norm = normalize_scores(score_arr, global_min, global_max)
            score_tensor = torch.from_numpy(score_norm)
            with torch.no_grad():
                outputs = model(seq_tensor, score_tensor)
                probs = torch.sigmoid(outputs).numpy()
            for h, p in zip(batch_headers, probs):
                chrom, start, end, strand = parse_header(h)
                name = 'edit' if p >= args.threshold else 'snp'
                line = f"{chrom}\t{start}\t{end}\t{name}\t{p:.6f}\t{strand}\n"
                if name == 'edit':
                    f_edit.write(line)
                else:
                    f_snp.write(line)
            progress.update(len(batch_headers))

    progress.close()
    print("Prediction completed.")


if __name__ == '__main__':
    main()
