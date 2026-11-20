# EditFinder

A-to-I RNA editing detection model to distinguish genuine RNA editing events from nucleotide changes arising from genomic SNPs

EditFinder is a two-stage pipeline wrapped into a single command:

'''
input.fa ─► [Stage 1: alignment scoring] ─► alignment_scores.csv
                                                     │
                                                     ▼
                                          [Stage 2: EditFinder_predict]
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          ▼                                                     ▼
                 edit_predictions.bed                                 snp_predictions.bed
'''

- Stage 1 ('NW_pair_seq_alignment.py') — computes a pairwise alignment score
  between each query sequence and a development (training) set, using ClustalW2-equivalent
  parameters ('match=1, mismatch=0, open_gap=-10, extend_gap=-0.1, end_gap=0').
- Stage 2 ('EditFinder_predict.py') — a 1D-CNN takes the 3-mer one-hot encoded sequence
  plus the normalized alignment score and classifies each locus as 'edit' or 'snp'.

---

## Environment

conda create -n EditFinder python=3.10 -y
conda activate editfinder
python -m pip install biopython torch numpy tqdm

---

## Usage

### One-shot

'''
bash run_editfinder.sh <INPUT_FA> <OUT_DIR> <MODEL_PATH> <DEVELOPMENT_FA>
'''

| argument         | description                                                        |
|------------------|--------------------------------------------------------------------|
| 'INPUT_FA'       | Query FASTA — one record = one candidate editing site              |
| 'OUT_DIR'        | Output directory (scores + BED files are written here)             |
| 'MODEL_PATH'     | Path to the '.pth' checkpoint                                      |
| 'DEVELOPMENT_FA' | Development FASTA used by Stage 1 scoring               |

Example

'''
bash run_editfinder.sh \
    /data/ref_editing/A_sites_centerNosnp.fa \
    /data/output/A_sites \
    models/final_model_seed12345.pth \
    /data_d/ZYan/data/Input_data/111/development.fa
'''

### Step-by-step

'''
# Stage 1 – alignment scoring
python NW_pair_seq_alignment.py <INPUT_FA> <OUT_DIR> <DEVELOPMENT_FA>

# Stage 2 – CNN prediction
python EditFinder_predict.py \
    --fasta       <INPUT_FA> \
    --scores      <OUT_DIR>/alignment_scores.csv \
    --model       <MODEL_PATH> \
    --output_dir  <OUT_DIR>/predictions
'''

'EditFinder_predict.py' options:

| option          | default | description                          |
|-----------------|---------|--------------------------------------|
| '--fasta'       | required | query FASTA (same as Stage 1)        |
| '--scores'      | required | 'alignment_scores.csv' from Stage 1  |
| '--model'       | required | '.pth' checkpoint            |
| '--output_dir'  | required | where BED files are written         |
| '--threshold'   | 0.5     | probability cutoff for 'edit'        |
| '--batch_size'  | 1024    | inference batch size                 |
| '--num_threads' | 20 % CPU | CPU threads for torch               |

---

## Input format

### Query FASTA ('INPUT_FA')

'''
>chr1:10000-10111(+)
ACGT...            # 111 bp
>chr2:25000-25111(-)
ACGT...
'''
---

## Output

| file | description |
|---|---|
| '<OUT_DIR>/alignment_scores.csv' | one integer score per input sequence, in input order (intermediate, Stage 1) |
| '<OUT_DIR>/predictions/edit_predictions.bed' | loci classified as RNA editing |
| '<OUT_DIR>/predictions/snp_predictions.bed'  | loci classified as genomic SNP |

BED files are standard 6-column format:

| col | name   | description                              |
|-----|--------|------------------------------------------|
| 1   | chrom  | e.g. 'chr1'                              |
| 2   | start  | 0-based start                            |
| 3   | end    | 1-based end                              |
| 4   | name   | 'edit' or 'snp'                          |
| 5   | score  | predicted probability (0–1, 6 decimals)  |
| 6   | strand | '+' / '-'                                |

---

## Notes

1. Resumable. if a run is interrupted, re-running the same command resumes from the last written line ('alignment_scores.csv' line count). Do not delete 'OUT_DIR' if you want to resume.
2. Sequence length must match training length (111 bp).
3. Parallelism. Stage 1 uses 'ProcessPoolExecutor' at 20 % of CPU cores; Stage 2 uses torch CPU threads at 20 % of CPU cores. Both are configurable in the source.

---