# ===========================
# Imports
# ===========================

import mne
from pathlib import Path
import numpy as np

# ===========================
# Participant and Parameters
# ===========================

participants_info = [
    {'Participant': '01', 'Session': '01', 'Task': 'RotationTask'},
    {'Participant': '02', 'Session': '01', 'Task': 'RotationTask'},
    {'Participant': '03', 'Session': '01', 'Task': 'RotationTask'} 
]

all_fs = [128]
h_freqs = [45, 100]
random_state = 42

for info in participants_info:
    id = info['Participant']
    session = info['Session']
    task = info['Task']

    # path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    base_path = project_root / f"data/sub-{id}/ses-{session}/"
    
    for fs in all_fs:
        for h_freq in h_freqs:
            print("")
            print(f"Processing id={id}, sampling rate={fs}, h_freq cut-off={h_freq}.")
            print("")
            
            # read data
            raw = mne.io.read_raw_brainvision(base_path / "eeg.vhdr", preload=True, verbose=False)    

            # drop channels
            if "Extra151" in raw.ch_names: raw.drop_channels(["Extra151"])
            if "Extra152" in raw.ch_names: raw.drop_channels(["Extra152"])

            # filter
            raw.filter(l_freq=1, h_freq=h_freq, fir_design='firwin')

            # rename trigger map 
            trigger_map = {
                    'Stimulus/S 10': 'start',
                    'Stimulus/S 11': 'end',
                    'Stimulus/S100': 'fig0_select',
                    'Stimulus/S101': 'fig1_select',
                    'Stimulus/S102': 'fig2_select',
                    'Stimulus/S103': 'fig3_select',
                    'Stimulus/S104': 'fig4_select',
                    'Stimulus/S105': 'fig5_select'
                }

            raw.annotations.rename(trigger_map)

            # drop unused triggers (make epoch calculations faster)
            keep_triggers = list(trigger_map.values())

            mask = [trigger in keep_triggers for trigger in raw.annotations.description]
            keep_annotations = raw.annotations[mask]
            raw.set_annotations(keep_annotations)

            # # crop (discard pre first and post last trial)
            # starts = [ann['onset'] for ann in raw.annotations if ann['description'] == 'start']
            # start_time = starts[0]

            # ends = [ann['onset'] for ann in raw.annotations if ann['description'] == 'end']
            # end_time = ends[-1]

            # raw = raw.crop(tmin=float(start_time), tmax=float(end_time) + 1) # +1s for windowing that extends over the end of trial

            # # update annotations
            # raw.annotations.onset = raw.annotations.onset - start_time 
            # if raw.annotations.description[0] != 'start':
            #     raw.annotations.description = np.insert(raw.annotations.description, 0, "start")
            #     raw.annotations.onset = np.insert(raw.annotations.onset, 0, 0.0)

            # resample
            raw.resample(sfreq=fs)
            
            # rereference
            raw.set_eeg_reference('average', projection=False)

            # define and set the montage (electrode layout)
            raw.rename_channels({'O9':'I1','O10':'I2' })

            dig_montage = mne.channels.make_standard_montage("standard_1005", head_size=0.095)
            _ = raw.set_montage(dig_montage)
            
            # ica
            print("")
            print("Computing ICA. This will take a while.")
            print("")
            ica = mne.preprocessing.ICA(
                n_components=125,
                max_iter="auto",
                random_state=random_state)
            ica.fit(raw)
            
            # save data
            print("Saving data.")
            print("")
            
            ica_dir = base_path.parent / "ica"
            ica_dir.mkdir(exist_ok=True)
            ica.save(ica_dir / f"{fs}hz_1-{h_freq}_ica.fif", overwrite=True)
            
            pre_dir = base_path.parent / "preprocessed"
            pre_dir.mkdir(exist_ok=True)
            raw.save(pre_dir / f"{fs}hz_1-{h_freq}_eeg_ica-False.fif", overwrite=True)