#!/usr/bin/env python3
"""
circIRESMap: Predicting circRNA IRES sites using CNN + DAMSR + BiGRU with Balanced Bagging.

Usage:
    python -m src.main --cv           Run 5-fold cross-validation with best hyperparameters
    python -m src.main --final-test   Train final model and evaluate on independent test set
"""

import sys
import os

from .config import (
    set_random_seed, get_runtime_device, print_gpu_info,
    FASTA_PREFIX, N_FOLDS, RANDOM_SEED, USE_CIRCULAR, OUTPUT_DIR,
    BEST_LR, BEST_FOCAL_GAMMA, BEST_FOCAL_ALPHA,
    BEST_WINDOW_SIZE, BEST_N_BAGS, BEST_BATCH_SIZE,
    BEST_NUM_FILTERS, BEST_BIGRU_HIDDEN_SIZE, BEST_BIGRU_NUM_LAYERS,
    BEST_DAMSR_LAYERS, BEST_DAMSR_NUM_HEADS, BEST_CONTEXT_POOL_GAMMA,
    MAX_EPOCHS, EARLY_STOP_PATIENCE
)
from .data_utils import load_data_from_fasta, create_kfold_splits
from .train import train_with_cross_validation, train_final_model_and_test


def main():
    """Main entry point"""
    set_random_seed(RANDOM_SEED)
    
    print("\n" + "=" * 80)
    print("circIRESMap - CNN + DAMSR + Balanced Bagging")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  FASTA prefix: {FASTA_PREFIX}")
    print(f"  Cross-validation folds: {N_FOLDS}")
    print(f"  Window size: {BEST_WINDOW_SIZE}bp")
    print(f"  Circular extraction: {'Yes' if USE_CIRCULAR else 'No'}")
    print(f"  Model: CNN + DAMSR (NeurIPS 2023) + BiGRU")
    print(f"  DAMSR: multi-scale retention (short+middle+long term)")
    print(f"  Bagging: {BEST_N_BAGS} models")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"\nOptimized hyperparameters:")
    print(f"  lr={BEST_LR}, focal_gamma={BEST_FOCAL_GAMMA}, focal_alpha={BEST_FOCAL_ALPHA}")
    print(f"  window_size={BEST_WINDOW_SIZE}, n_bags={BEST_N_BAGS}, batch_size={BEST_BATCH_SIZE}")
    print(f"  num_filters={BEST_NUM_FILTERS}, bigru_hidden={BEST_BIGRU_HIDDEN_SIZE}, bigru_layers={BEST_BIGRU_NUM_LAYERS}")
    print(f"  damsr_layers={BEST_DAMSR_LAYERS}, damsr_heads={BEST_DAMSR_NUM_HEADS}, pool_gamma={BEST_CONTEXT_POOL_GAMMA}")

    device = get_runtime_device()
    print_gpu_info(device)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sequences, train_labels, test_labels, id_mapping = load_data_from_fasta(FASTA_PREFIX)
    
    train_ids = list(train_labels.keys())
    folds = create_kfold_splits(train_ids, N_FOLDS, RANDOM_SEED)
    
    print("\n" + "=" * 80)
    print("Data preparation complete")
    print("=" * 80)
    
    print(f"\nTraining circRNAs: {len(train_labels)}")
    print(f"Test circRNAs: {len(test_labels)}")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--cv':
        print("\n" + "=" * 80)
        print("Mode: 5-fold cross-validation")
        print("=" * 80)
        
        try:
            cv_results = train_with_cross_validation(
                sequences=sequences,
                train_labels=train_labels,
                folds=folds,
                id_mapping=id_mapping,
                window_size=BEST_WINDOW_SIZE,
                use_circular=USE_CIRCULAR,
                output_dir=OUTPUT_DIR,
                lr=BEST_LR,
                focal_gamma=BEST_FOCAL_GAMMA,
                focal_alpha=BEST_FOCAL_ALPHA,
                max_epochs=MAX_EPOCHS,
                patience=EARLY_STOP_PATIENCE,
                run_name='final_cv',
                n_bags=BEST_N_BAGS,
                batch_size=BEST_BATCH_SIZE,
                num_filters=BEST_NUM_FILTERS,
                bigru_hidden_size=BEST_BIGRU_HIDDEN_SIZE,
                bigru_num_layers=BEST_BIGRU_NUM_LAYERS,
                damsr_layers=BEST_DAMSR_LAYERS,
                damsr_num_heads=BEST_DAMSR_NUM_HEADS,
                context_pool_gamma=BEST_CONTEXT_POOL_GAMMA
            )
            
            print(f"\n{'='*80}")
            print("Cross-validation complete!")
            print(f"  Mean AUPR: {cv_results['aupr'].mean():.4f} +/- {cv_results['aupr'].std():.4f}")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--final-test':
        print("\n" + "=" * 80)
        print("Mode: Final training + independent test")
        print("=" * 80)
        
        try:
            test_results = train_final_model_and_test(
                sequences, train_labels, test_labels, id_mapping,
                window_size=BEST_WINDOW_SIZE,
                use_circular=USE_CIRCULAR,
                n_bags=BEST_N_BAGS,
                batch_size=BEST_BATCH_SIZE,
                lr=BEST_LR,
                focal_gamma=BEST_FOCAL_GAMMA,
                focal_alpha=BEST_FOCAL_ALPHA,
                num_filters=BEST_NUM_FILTERS,
                bigru_hidden_size=BEST_BIGRU_HIDDEN_SIZE,
                bigru_num_layers=BEST_BIGRU_NUM_LAYERS,
                damsr_layers=BEST_DAMSR_LAYERS,
                damsr_num_heads=BEST_DAMSR_NUM_HEADS,
                context_pool_gamma=BEST_CONTEXT_POOL_GAMMA,
                max_epochs=MAX_EPOCHS,
                patience=EARLY_STOP_PATIENCE
            )
            
            print(f"\n{'='*80}")
            print("Final training complete!")
            print(f"  Independent test AUPR: {test_results['aupr']:.4f}")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("\nUsage:")
        print("  python -m src.main --cv           Run 5-fold cross-validation")
        print("  python -m src.main --final-test    Train final model and test")
        
        return sequences, train_labels, test_labels, id_mapping, folds


if __name__ == '__main__':
    main()
