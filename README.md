# circIRESMap

Predicting circRNA IRES sites using CNN + DAMSR + BiGRU with Balanced Bagging.

## Model Architecture

- **Shared CNN**: Extracts local sequence features (motifs)
- **Branch 1 - DAMSR**: Multi-scale retention mechanism with distance-decay attention
- **Branch 2 - BiGRU**: Bidirectional recurrent accumulation for implicit global context
- **Balanced Bagging**: Ensemble of N models, each trained on all positive samples + 1/N negative samples

## Project Structure

```
circIRESMap/
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and device configuration
│   ├── data_utils.py      # Data loading, preprocessing, result output
│   ├── model.py           # Model definitions (FocalLoss, DAMSR, CNNDAMSRBiGRUModel)
│   ├── train.py           # Training loops (CV and final test)
│   └── main.py            # Entry point
├── data/                  # Place circRNA FASTA files here
│   ├── circRNA_seq.fasta
│   ├── circRNA_train.fasta
│   ├── circRNA_test.fasta
│   └── circRNA_id_mapping.tsv
├── results/               # Output directory
└── requirements.txt
```

## Data Format

Place the following files in `data/` directory:
- `circRNA_seq.fasta` - circRNA sequences
- `circRNA_train.fasta` - Training set annotations
- `circRNA_test.fasta` - Test set annotations
- `circRNA_id_mapping.tsv` - ID mapping table

## Usage

```bash
# 5-fold cross-validation
python -m src.main --cv

# Train final model and evaluate on independent test set
python -m src.main --final-test
```

## Optimized Hyperparameters

| Parameter | Value | Search Stage |
|-----------|-------|-------------|
| Learning rate | 0.001 | Stage 1 |
| Focal gamma | 1.5 | Stage 1 |
| Focal alpha | 0.5 | Stage 1 |
| Window size | 151 | Stage 2 |
| N bags | 9 | Stage 2 |
| Batch size | 128 | Stage 2 |
| Num filters | 128 | Stage 3 |
| BiGRU hidden size | 128 | Stage 3 |
| BiGRU num layers | 2 | Stage 3 |
| DAMSR layers | 1 | Stage 4 |
| DAMSR num heads | 2 | Stage 4 |
| Pool gamma | 0.97 | Stage 4 |

## Seed

Random seed is fixed at 42 for reproducibility.
