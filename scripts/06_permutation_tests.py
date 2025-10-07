# ===========================
# Imports
# ===========================

import pickle
import sys
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from mne.decoding import Scaler
import os

script_dir = Path(__file__).parent  # directory of train_models.py
sys.path.append(str(script_dir.parent / 'helpers'))
from helpers import permutation_test

# ===========================
# Participant and Parameters
# ===========================

# participant and processing parameters
participants_info = [
    {'Participant': '01', 'Session': '01', 'Task': 'RotationTask'},
    {'Participant': '02', 'Session': '01', 'Task': 'RotationTask'},
    {'Participant': '03', 'Session': '01', 'Task': 'RotationTask'}
]
all_fs = [128]
h_freqs = [
    45, 
    100
    ]
icas = [
    False, 
    True
    ]
targets_y = [
    'selection',
    'decision'
    ]
config_types = ['long-window']

# permutation parameters
N = 10000 # permutations
metrics = ['balanced_accuracy', 'pr-auc'] # which metric

# ===========================
# Permutation Tests
# ===========================

script_dir = Path(__file__).parent

for info in participants_info:
    for config_type in config_types:
        for target in targets_y:
            for ica in icas:
                for fs in all_fs:
                    for h_freq in h_freqs:
                        print(f"\nPermutating participant={info['Participant']}, target={target}, config={config_type}, fs={fs}, h_freq={h_freq}, ica={ica}\n")

                        # paths
                        base_path = script_dir.parent / "data" / f"sub-{info['Participant']}/"
                        model_dir = base_path / "models" / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}" / config_type
                        folds_dir = model_dir / "folds"

                        # load data
                        X = np.load(base_path / "training_data" / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}" / config_type / "X.npy")
                        y = np.load(base_path / "training_data" / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}" / config_type / "y.npy")
                        
                        # ===========================
                        # Original Model
                        # ===========================
                        
                        # containers
                        obs_stats = [[] for _ in range(len(metrics))]
                        null_stats = [[] for _ in range(len(metrics))]
                        p_values = [[] for _ in range(len(metrics))]
                        
                        # load splits
                        splits_path = model_dir / "splits.pkl"
                        with open(splits_path, "rb") as f:
                                splits = pickle.load(f)
                                
                        # load model        
                        model_path = model_dir / "model-stratify-False.h5"
                        model = load_model(model_path)

                        # extract data
                        train_mask, test_mask, val_mask = splits
                        X_te, y_te = X[test_mask], y[test_mask]

                        # normalize
                        scaler_path = model_dir / "scaler.pkl"
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)
                        X_te = scaler.transform(X_te)

                        # predict
                        y_score = model.predict(X_te, verbose=2)[:, 1]
                        
                        # permutate
                        for idx, metric in enumerate(metrics):
                                obs_stat, null_stat, p_value = permutation_test(y_score, y_te, metric=metric, N=N)
                                obs_stats[idx].append(obs_stat)
                                null_stats[idx].append(null_stat)
                                p_values[idx].append(p_value)

                        # ===========================
                        # Folds
                        # ===========================
                        
                        # containers
                        fold_obs_stats = [[] for _ in range(len(metrics))]
                        fold_null_stats = [[] for _ in range(len(metrics))]
                        fold_p_values = [[] for _ in range(len(metrics))]
                        y_te_aggregated = []
                        y_te_score_aggregated = []
                        fold_te_sizes = []
                        
                        # load splits
                        fold_splits_path = model_dir / "fold_splits.pkl"
                        if fold_splits_path.exists():
                            with open(fold_splits_path, "rb") as f:
                                fold_splits = pickle.load(f)
        
                        # permutate folds
                        for fold_idx, (train_mask, test_mask) in enumerate(fold_splits):
                            print(f"Evaluating fold {fold_idx}")

                            # load
                            fold_model_path = folds_dir / f"model_fold{fold_idx}.h5"
                            model_fold = load_model(fold_model_path)

                            # extract test
                            X_te, X_tr, y_tr, y_te = X[test_mask], X[train_mask], y[train_mask], y[test_mask]
                            
                            # normalize (this was done training the model)
                            scaler = Scaler(scalings='mean') 
                            X_tr = scaler.fit_transform(X_tr)
                            X_te = scaler.transform(X_te) 

                            # predict
                            y_score = model_fold.predict(X_te, verbose=2)[:, 1]
                            
                            # aggregate
                            y_te_aggregated.append(y_te)
                            y_te_score_aggregated.append(y_score)
                            fold_te_sizes.append(len(y_te))

                            for idx, metric in enumerate(metrics):
                                fold_obs_stat, fold_null_stat, fold_p_value = permutation_test(y_score, y_te, metric=metric, N=N)
                                fold_obs_stats[idx].append(fold_obs_stat)
                                fold_null_stats[idx].append(fold_null_stat)
                                fold_p_values[idx].append(fold_p_value)
                        
                        # aggregated score
                        y_te_aggregated = np.concatenate(y_te_aggregated, axis=0)
                        y_te_score_aggregated = np.concatenate(y_te_score_aggregated, axis=0)

                        fold_agg_obs_stats = []
                        fold_agg_null_stats = []
                        fold_agg_p_values = []

                        for metric in metrics:
                            obs_stat, null_stat, p_value = permutation_test(
                                y_te_score_aggregated,
                                y_te_aggregated,
                                fold_te_size=fold_te_sizes, 
                                metric=metric,
                                N=N
                            )
                            fold_agg_obs_stats.append(obs_stat)
                            fold_agg_null_stats.append(null_stat)
                            fold_agg_p_values.append(p_value)
                            
                        # ===========================
                        # Save
                        # ===========================
                                                        
                        perm_results = {
                            'metrics': metrics,
                            'N': N,
                            'original': {
                                'obs_stats': obs_stats,
                                'null_stats': null_stats,
                                'p_values': p_values
                            },
                            'folds': {
                                'indices': list(range(len(fold_splits))),
                                'obs_stats': fold_obs_stats,
                                'null_stats': fold_null_stats,
                                'p_values': fold_p_values
                            },
                            'folds_aggregated': {
                                'obs_stats': fold_agg_obs_stats,
                                'null_stats': fold_agg_null_stats,
                                'p_values': fold_agg_p_values
                            }
                        }
                        
                        with open(model_dir / "perm_results.pkl", "wb") as f:
                            pickle.dump(perm_results, f)
                            
                        print(f"\nPermutation results saved.")

                        