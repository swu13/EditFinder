#!/usr/bin/env python3
"""
Usage: python NW_pair_seq_alignment.py <INPUT_FA> <OUT_DIR> <DEVELOPMENT_FA>
"""

import os
import sys
import time
import subprocess
import pickle
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from Bio.Align import PairwiseAligner

import multiprocessing as mp
CPU_COUNT       = mp.cpu_count()
NUM_WORKERS     = max(1, int(CPU_COUNT * 0.20))
BATCH_SIZE      = 5000
MAX_PENDING     = NUM_WORKERS * 2
CKPT_INTERVAL   = 300


def save_ckpt(path, obj):
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def load_ckpt(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def count_lines(path):
    r = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
    if r.returncode == 0:
        return int(r.stdout.strip().split()[0])
    return 0


def count_fasta_seqs(path):
    r = subprocess.run(f"grep -c '^>' '{path}'", shell=True,
                       capture_output=True, text=True)
    if r.returncode == 0:
        return int(r.stdout.strip())
    return 0


def load_development_sequences(fasta_path):
    """Return [(header, sequence), ...]"""
    development = []
    header = None
    seq = ''
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    development.append((header, seq.upper()))
                header = line
                seq = ''
            else:
                seq += line
        if header is not None:
            development.append((header, seq.upper()))
    return development


def stream_fasta_batches(fasta_path, batch_size, skip_seqs=0):
    batch = []
    batch_idx = 0
    skipped = 0
    header = None
    seq = ''

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    if skipped < skip_seqs:
                        skipped += 1
                    else:
                        batch.append((header, seq.upper()))
                        if len(batch) >= batch_size:
                            yield batch_idx, batch
                            batch_idx += 1
                            batch = []
                header = line
                seq = ''
            else:
                seq += line
        if header is not None:
            if skipped < skip_seqs:
                skipped += 1
            else:
                batch.append((header, seq.upper()))
    if batch:
        yield batch_idx, batch


def score_batch(args):
    batch_idx, batch, development = args
    aligner = PairwiseAligner()
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.1
    aligner.target_end_gap_score = 0
    aligner.query_end_gap_score = 0
    aligner.mode = 'global'

    scores = []
    try:
        for idx_q, (_, query_seq) in enumerate(batch):
            q = query_seq.strip().upper()
            if len(q) == 0:
                scores.append(-1)
                continue
            best = -9999999
            for idx_t, (_, target_seq) in enumerate(development):
                t = target_seq.strip().upper()
                if len(t) == 0:
                    continue
                if q == t:
                    continue
                s = aligner.score(q, t)
                if s > best:
                    best = s
            if best >= 0:
                scores.append(int(best))
            else:
                scores.append(-1)
        return batch_idx, scores, None
    except Exception as e:
        return batch_idx, [-1] * len(batch), str(e)


def main():
    if len(sys.argv) != 4:
        print("Usage: python {} <INPUT_FA> <OUT_DIR> <DEVELOPMENT_FA>".format(sys.argv[0]),
              file=sys.stderr)
        sys.exit(1)

    INPUT_FA        = sys.argv[1]
    OUT_DIR         = sys.argv[2]
    DEVELOPMENT_FA  = sys.argv[3]

    if not os.path.isfile(INPUT_FA):
        print(f"ERROR: INPUT_FA not found: {INPUT_FA}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(DEVELOPMENT_FA):
        print(f"ERROR: DEVELOPMENT_FA not found: {DEVELOPMENT_FA}", file=sys.stderr)
        sys.exit(1)

    TMP_DIR    = os.path.join(OUT_DIR, "NW_pair_seq_alignment")
    SCORE_FILE = os.path.join(OUT_DIR, "alignment_scores.csv")
    CKPT_FILE  = os.path.join(TMP_DIR, "progress.pkl")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    print(f"CPU: {CPU_COUNT}, workers: {NUM_WORKERS}", file=sys.stderr)
    print(f"Input FASTA       : {INPUT_FA}",       file=sys.stderr)
    print(f"Output directory  : {OUT_DIR}",        file=sys.stderr)
    print(f"Development FASTA : {DEVELOPMENT_FA}", file=sys.stderr)

    print("Loading development sequences...", file=sys.stderr)
    development_seqs = load_development_sequences(DEVELOPMENT_FA)
    development_num = len(development_seqs)
    print(f"Development sequences: {development_num:,}", file=sys.stderr)

    already_done = count_lines(SCORE_FILE) if os.path.exists(SCORE_FILE) else 0
    print(f"Already processed: {already_done:,}", file=sys.stderr)

    score_fh = open(SCORE_FILE, 'a', buffering=1 << 20)

    pending: dict[int, tuple] = {}
    next_write_idx = 0
    total_written  = already_done
    last_ckpt_time = time.time()
    t_start        = time.time()

    def flush_pending():
        nonlocal next_write_idx, total_written
        while next_write_idx in pending:
            seq_count, fut = pending[next_write_idx]
            if not fut.done():
                break
            batch_idx_r, scores, err = fut.result()
            if err:
                print(f"  [error] batch {next_write_idx}: {err}", file=sys.stderr)
                score_fh.write('\n'.join(['-1'] * seq_count) + '\n')
            else:
                score_fh.write('\n'.join(str(s) for s in scores) + '\n')
                score_fh.flush()
                total_written += seq_count
                print(f"  batch {next_write_idx:04d}: {seq_count} seqs scored", file=sys.stderr)
            del pending[next_write_idx]
            next_write_idx += 1

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for batch_idx, records in stream_fasta_batches(
                INPUT_FA, BATCH_SIZE, skip_seqs=already_done):
            n = len(records)
            if n == 0:
                continue

            fut = executor.submit(score_batch, (batch_idx, records, development_seqs))
            pending[batch_idx] = (n, fut)

            while len(pending) >= MAX_PENDING:
                flush_pending()
                if len(pending) >= MAX_PENDING:
                    time.sleep(0.2)

            now = time.time()
            if now - last_ckpt_time > CKPT_INTERVAL:
                flush_pending()
                save_ckpt(CKPT_FILE, {
                    'total_written': total_written,
                    'next_write_idx': next_write_idx,
                    'timestamp': datetime.now().isoformat()
                })
                elapsed = now - t_start
                rate = (total_written - already_done) / elapsed if elapsed > 0 else 0
                print(f"  [progress] written={total_written:,}, rate={rate:.1f} seq/s",
                      file=sys.stderr)
                last_ckpt_time = now

        print("All batches submitted, waiting for completion...", file=sys.stderr)
        while pending:
            flush_pending()
            if pending:
                time.sleep(0.2)

    score_fh.close()

    total_time = time.time() - t_start
    final_count = count_lines(SCORE_FILE)
    print(f"\nDone in {total_time/3600:.2f}h", file=sys.stderr)
    print(f"Score file: {SCORE_FILE}  ({final_count:,} lines)", file=sys.stderr)

    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)


if __name__ == "__main__":
    main()