# ===========================
# Imports
# ===========================

import sys
import numpy as np
from pathlib import Path
import pickle
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.optimizers import Adam
import gc
from keras import backend as K
from mne.decoding import Scaler

script_dir = Path(__file__).parent  # directory of train_models.py
sys.path.append(str(script_dir.parent / 'helpers'))
sys.path.append(str(script_dir.parent / 'EEGModels'))
from EEGModels import EEGNet
from helpers import split_by_trial

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

# training data parameters
stratify = False

# training parameters
random_state = 42
epochs_decision_selection = [100, 150]
lr = 1e-3
cross_validation = True
n_folds = 10

# enable GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU found, training will use CPU.")
else:
    print(f"GPUs available: {physical_devices}")

# grab gpu as needed, not preallocated
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True) 

# ===========================
# Train
# ===========================

for info in participants_info:
    for config_type in config_types:
        for target in targets_y:
            for ica in icas:
                for fs in all_fs:
                    for h_freq in h_freqs:
                        # parameters 
                        id = info['Participant']
                        session = info['Session']
                        task = info['Task']
                        if target == 'decision':
                            epochs = epochs_decision_selection[0]
                        else: 
                            epochs = epochs_decision_selection[1]
                            
                        # paths
                        base_path = script_dir.parent / "data" / f"sub-{id}/"
                        data_dir = base_path / "training_data" / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}" / config_type
                        
                        model_dir = base_path / "models" 
                        model_dir.mkdir(exist_ok=True)
                        
                        model_para_dir = model_dir / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}"
                        model_para_dir.mkdir(exist_ok=True)

                        save_dir = model_para_dir / config_type
                        save_dir.mkdir(exist_ok=True)
                        
                        print("")
                        print(f"Training participant={id}, config_type={config_type}, target={target}, fs={fs}, h_freq={h_freq}, ica={ica}")
                        print("")
                        
                        # load data
                        X = np.load(data_dir / "X.npy")
                        y = np.load(data_dir / "y.npy")
            
                        with open(data_dir / "data_info.pkl", 'rb') as f:
                            data_info = pickle.load(f)
                                                    
                        # split train_test
                        train_mask, test_mask, val_mask = split_by_trial(y=y, trial_label=data_info['trials'], stratify=stratify, random_state=random_state)
                        X_train = X[train_mask]
                        y_train = y[train_mask]
                        
                        # save splits 
                        splits = (train_mask, test_mask, val_mask)
                        with open(save_dir / "splits.pkl", "wb") as f:
                            pickle.dump(splits, f)
                        
                        # normalize
                        scaler = Scaler(scalings='mean') 
                        X_train = scaler.fit_transform(X_train) 
                        
                        with open(save_dir / "scaler.pkl", "wb") as f:
                            pickle.dump(scaler, f)
                            
                        # compute class weights
                        classes = np.unique(y_train)
                        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                        class_weight_dict = dict(zip(classes, class_weights))
                        
                        # garbage collect
                        gc.collect()
                        
                        # specify model, hyperparameters and compile
                        model = EEGNet(
                            nb_classes = 2, 
                            Chans = X_train.shape[1], 
                            Samples = X_train.shape[2]
                        )

                        optimizer = Adam(learning_rate=lr)

                        model.compile(
                            loss = 'sparse_categorical_crossentropy',
                            metrics=['accuracy'],
                            optimizer = optimizer)

                        # train
                        history = None
                        try:
                            history = model.fit(
                                X_train,
                                y_train,
                                epochs=epochs,
                                class_weight=class_weight_dict,
                                verbose=0
                            )
                        except KeyboardInterrupt:
                            print("\nTraining interrupted.")
                        finally:
                            if history is not None:
                                with open(save_dir / "history.pkl", "wb") as f:
                                    pickle.dump(history.history, f)
                                print("History saved.")
                            
                            model.save(str(save_dir / "model.h5"))    
                            print("Model saved.")
                            
                            # cross validation
                            if cross_validation:
                                print()
                                print("Cross validation.")
                                print()
                                
                                fold_model_dir = save_dir / "folds"
                                fold_model_dir.mkdir(exist_ok=True)
                                
                                # split train_test
                                fold_splits = split_by_trial(y=y, trial_label=data_info['trials'], n_splits=n_folds, stratify=stratify, random_state=random_state)
                                                
                                with open(save_dir / "fold_splits.pkl", "wb") as f:
                                    pickle.dump(fold_splits, f)
                                
                                for fold_idx, (train_mask, test_mask) in enumerate(fold_splits):
                                    # split train_test                              
                                    X_tr, y_tr = X[train_mask], y[train_mask]
                                    X_te, y_te = X[test_mask], y[test_mask]
                                    
                                    # normalize
                                    scaler = Scaler(scalings='mean') 
                                    X_tr = scaler.fit_transform(X_tr) 
                                    X_te = scaler.transform(X_te)
                                    
                                    # class weights
                                    classes = np.unique(y_tr)
                                    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
                                    class_weight_dict = dict(zip(classes, class_weights))
                                    
                                    # create model per fold
                                    model_fold = EEGNet(nb_classes=2, Chans=X_tr.shape[1], Samples=X_tr.shape[2])
                                    optimizer = Adam(learning_rate=lr)
                                    model_fold.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                                    
                                    model_fold.fit(X_tr, y_tr, epochs=epochs, class_weight=class_weight_dict, verbose=0)
                                    
                                    # save model_fold
                                    model_fold.save(fold_model_dir / f"model_fold{fold_idx}.h5") 
                                    
                                    # free GPU memory
                                    K.clear_session()
                                    gc.collect()
                                    
                        K.clear_session()
                        
                        print("Successfully terminated.")
