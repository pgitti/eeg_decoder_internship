# ===========================
# Imports
# ===========================

import sys
from pathlib import Path
import numpy as np
import pickle

sys.path.append('helpers')
from helpers import load_data, extract_training_data

# ===========================
# Participant and Parameters
# ===========================

# participant and signal processing parameters
participants_info = [
    {'Participant': '01', 'Session': '01', 'Task': 'RotationTask'},
    {'Participant': '02', 'Session': '01', 'Task': 'RotationTask'},
    {'Participant': '03', 'Session': '01', 'Task': 'RotationTask'}
]

all_fs = [128]
h_freqs = [45, 100]
icas = [False, True]
targets_y = ['selection', 'decision']
random_state = 42
participant_omit_trials = [[144, 145, 146, 147, 148, 149], None, None]

# windowing and selection parameters
data_config = [    
    {'type': 'long-window', 
    # eeg window specifications
    'window_ms': 500,
    'step_ms': 50,
    'min_fixation_ms': 50, # below is considered no fixation
    # selection corrections
    'use_window': True, 
    'selection_window_ms': 250,
    'omit_brief_fixations_per_trial': False, 
    'min_fixation_per_trial_ms': 2000, # below that is not considered a no_decision
    'flag_no_decisions_false': False,
    # train test split
    'test_size': 0.15,
    'val_size': None
    },
    
    # {'type': 'medium-window', 
    # # eeg window specifications
    # 'window_ms': 300,
    # 'step_ms': 50,
    # 'min_fixation_ms': 50, # below is considered no fixation
    # # selection corrections
    # 'use_window': True, 
    # 'selection_window_ms': 250,
    # 'omit_brief_fixations_per_trial': False, 
    # 'min_fixation_per_trial_ms': 1000, # below that is not considered a no_decision
    # 'flag_no_decisions_false': False,
    # # train test split
    # 'test_size': 0.15,
    # 'val_size': None
    # }
]
# ===========================
# Data Loading and Saving
# ===========================

for info, omit_trials in zip(participants_info, participant_omit_trials):
    # specify paths
    id = info['Participant']
    session = info['Session']
    task = info['Task']

    # Get project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    base_path = project_root / f"data/sub-{id}/"
    data_dir = base_path / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True) 
    
    for config in data_config:
        for target in targets_y:
            for fs in all_fs:
                for h_freq in h_freqs:
                    for ica in icas:
                        print("")
                        print(f"Processing id={id}, target={target}, sampling rate={fs}, h_freq cut-off={h_freq}, ica={ica}, config={config}.")
                        print("")
                        
                        save_dir = data_dir / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}" / config['type']
                        save_dir.mkdir(parents=True, exist_ok=True) 
                                        
                        # create data
                        data = load_data(info, fs, h_freq, ica)
                        training_data, data_info = extract_training_data(
                            data,
                            target,
                            config['window_ms'],
                            config['step_ms'],
                            config['use_window'],
                            config['selection_window_ms'],
                            config['omit_brief_fixations_per_trial'],
                            config['min_fixation_per_trial_ms'],
                            config['flag_no_decisions_false'],
                            config['min_fixation_ms']
                            )
                        
                        # omit_trials?
                        if omit_trials is not None:
                            omit_mask = ~np.isin(data_info['trials'], omit_trials)
                        else:
                            omit_mask = np.ones_like(data_info['trials'], dtype=bool)

                        X, y = training_data[0][omit_mask], training_data[1][omit_mask]
                        data_info['shape'] = X.shape
                            
                        # save
                        np.save(save_dir / "X.npy", X)
                        np.save(save_dir / "y.npy", y)
                        
                        with open(save_dir / "data_info.pkl", 'wb') as f:
                            pickle.dump(data_info, f, protocol=pickle.HIGHEST_PROTOCOL)
                
      