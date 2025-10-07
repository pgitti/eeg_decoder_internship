# ===========================
# Imports
# ===========================
from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import mne
from IPython.display import display
from IPython.core.display import HTML
import matplotlib.cm
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow.keras.models import load_model
import pickle
import seaborn as sns

# ===========================
# Training Data Exploratory
# ===========================

def load_data(info, fs, h_freq, ica):
    """
    Loads raw participant data (EEG + eye-tracking + answers).
    """   
    id = info['Participant']
    session = info['Session']
    n_trials_per_block = 30
    
    # file paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up from src/ to project root
    base_path = project_root / f"data/sub-{id}/ses-{session}/"
    
    # load data
    answers_file = pd.read_csv(base_path / "answers.csv")
    mouse_log = pd.read_csv(base_path / "mouse_log.csv", low_memory=False)
    image_boxes = _create_image_boxes()
    mouse_log['label'] = [_find_fixation_label(x, y, image_boxes) for x, y in zip(mouse_log['mouse x'], mouse_log['mouse y'])]
    fixation_data = eye_fixation_extraction(mouse_log)
    
    pre_dir = base_path.parent / "preprocessed"
    eeg = mne.io.read_raw_fif(pre_dir / f"{fs}hz_1-{h_freq}_eeg_ica-{ica}.fif", preload=True)
    
    onsets = eeg.annotations.onset
    triggers = eeg.annotations.description

    # process
    figure_checked = compute_figure_checked(fixation_data, n_trials_per_block)
    fixations_analysis = compute_fixations_analysis(fixation_data, n_trials_per_block)
    performance_df = create_performance_labels(answers_file, figure_checked)

    data = {
        'id': id,
        'fs': fs,
        'h_freq': h_freq,
        'eeg': eeg,
        'answers_file': answers_file,
        'onsets': onsets,
        'triggers': triggers,
        'figure_checked': figure_checked,
        'fixation_data': fixation_data,
        'fixations_analysis': fixations_analysis,
        'performance_df': performance_df,
        'mouse_log': mouse_log
    }
    return data

def build_selection_data(data):
    """
    Combines EEG and eye-tracking data ('load_data') to build selection data. 
    Can be used to check when keys were selected and where particpants were looking at that time to remove unintentional selections by participants.
    """
    # --- Extract raw components ---
    answers_file = data['answers_file'].copy()
    figure_checked = data['figure_checked'].copy()
    onsets = data['onsets'].copy()
    triggers = data['triggers'].copy()
    fixation_data = data['fixations_analysis'].copy()

    # --- Base selection_data from experiment log ---
    selection_data = pd.DataFrame({
        "s_after_experiment_start": onsets,
        "trigger": triggers
    })
    selection_data['trial_number'] = (selection_data['trigger'] == 'start').cumsum() - 1

    # --- Prepare fixation_data ---
    fixation_data['trial_number'] = ((fixation_data['trial_number'] + 1) + fixation_data['block_number'] * 30) - 1
    fixation_data['trigger'] = fixation_data.groupby('trial_number').cumcount().map(lambda i: f"fig{i}_select")
    fixation_data = fixation_data[['trial_number', 'trigger', 'Time_on_option_(in_seconds)']].drop_duplicates()

    # --- Prepare answers_file ---
    answers_file['trial_number'] = answers_file['trial_number'] + answers_file['block_number'] * 30
    answers_file = (
        answers_file
        .assign(
            selected_keys=answers_file['selected_keys'].str.split('|'),
            right_answers_keys=answers_file['right_answers_keys'].str.split('|')
        )
        .explode(['selected_keys', 'right_answers_keys'], ignore_index=True)
    )
    answers_file['trigger'] = answers_file.groupby('trial_number').cumcount().map(lambda i: f"fig{i}_select")
    answers_file = answers_file.drop(columns='block_number')

    # --- Prepare figure_checked ---
    figure_checked_melted = (
        figure_checked
        .melt(id_vars=['trial_number', 'block_number'], var_name='trigger', value_name='figure_checked')
        .assign(
            trigger=lambda df: df['trigger'].str.replace(r'figure_([0-9])', r'fig\1_select', regex=True),
            trial_number=lambda df: df['trial_number'] + df['block_number'] * 30
        )
        .drop(columns='block_number')
    )

    # --- Build complete trial-trigger grid ---
    triggers_full = ['start'] + [f"fig{i}_select" for i in range(6)] + ['end']
    trial_numbers = selection_data['trial_number'].unique().tolist()
    trial_trigger_grid = pd.MultiIndex.from_product(
        [trial_numbers, triggers_full],
        names=['trial_number', 'trigger']
    ).to_frame(index=False)

    # --- Merge all sources ---
    selection_data = (
        trial_trigger_grid
        .merge(selection_data, on=['trial_number', 'trigger'], how='left')
        .merge(answers_file, on=['trial_number', 'trigger'], how='left')
        .merge(figure_checked_melted, on=['trial_number', 'trigger'], how='left')
        .merge(fixation_data, on=['trial_number', 'trigger'], how='left')
    )

    # --- Convert and compute correctness ---
    selection_data['selected_keys'] = selection_data['selected_keys'].map({'selected': 1, 'not_selected': 0}).astype(float)
    selection_data['right_answers_keys'] = selection_data['right_answers_keys'].map({'True': 1, 'False': 0}).astype(float)

    selection_data['correct'] = np.where(
        selection_data[['selected_keys', 'right_answers_keys']].isna().any(axis=1),
        np.nan,
        (selection_data['selected_keys'] == selection_data['right_answers_keys']).astype(float)
    )

    # --- Ensure start/end have NaN correctness ---
    selection_data.loc[selection_data['trigger'].isin(['start', 'end']), 'correct'] = np.nan

    # --- Drop irrelevant columns if they exist ---
    cols_to_drop = [c for c in ['base_image', 'displayed_options', 'timestamp', 'false_positive', 'false_negative'] if c in selection_data]
    selection_data = selection_data.drop(columns=cols_to_drop).reset_index(drop=True)

    return selection_data

def correct_selections(
    selection_data, 
    use_window=True, 
    selection_window_s = 0.25, 
    omit_brief_fixations_per_trial = True, 
    min_fixation_per_trial_s = 2, 
    flag_no_decisions_false = True
    ):
    """
    Corrects unintentional selections by participants in selection_data. 
    If a figure was selected within the window at the start of trial and the same figure was selected within the window in the previous trial,
    an artificial selection is added at the end of the previous trial and the answer is updated. 
    If figure is gazed at for at least min_fixation_per_trial_ms and flag_no_decisions_false is True, correct_updated becomes False.
    """
    # initialize new columns
    selection_data['artificial'] = False
    selection_data['decision'] = np.nan
    selection_data['decision'] = selection_data['decision'].astype(object)
    selection_data.loc[
        (selection_data['trigger'] != 'start') & (selection_data['trigger'] != 'end') & (~selection_data['correct'].isna()),
        'decision'
    ] = 1.0

    # create updated columns as copies
    selection_data['selected_keys_updated'] = selection_data['selected_keys']
    selection_data['decision_updated'] = selection_data['decision']
    selection_data['correct_updated'] = selection_data['correct']
    selection_data['figure_checked_updated'] = selection_data['figure_checked']

    trials = selection_data['trial_number'].max()

    # 1) add artifical selection in previous trial, when selection was made both in previous and current trial within selection_window_s
    if use_window:
        selections = selection_data.loc[selection_data['trial_number'] == 0, 'trigger']
        time = selection_data.loc[selection_data['trial_number'] == 0, 's_after_experiment_start']

        for trial in range(1, trials + 1):
            selections_prev, time_prev = selections, time
            selections = selection_data.loc[selection_data['trial_number'] == trial, 'trigger']
            time = selection_data.loc[selection_data['trial_number'] == trial, 's_after_experiment_start']

            idx_in_window = (time.values - time.values[0]) < selection_window_s
            idx_in_window_prev = (time_prev.values[-1] - time_prev.values) < selection_window_s
            idx_shared = np.isin(selections_prev, selections[idx_in_window])
            change_selections = idx_in_window_prev & idx_shared & (selections_prev != 'end')

            if not change_selections.any():
                continue

            # rows needing artificial insertion
            change_idx = selections_prev[change_selections].index
            new_rows = selection_data.loc[change_idx].copy()
            new_rows['s_after_experiment_start'] = time_prev.values[-1] - 0.001
            new_rows['artificial'] = 1.0
            new_rows['decision'] = np.nan

            # insert artificial rows
            n_original = len(selection_data)
            selection_data = pd.concat([selection_data, new_rows], ignore_index=True)

            # update answers
            new_rows_idx = range(n_original, n_original + len(new_rows))
            update_prev_idx = selections_prev.index[np.isin(selections_prev, selection_data.loc[change_idx, 'trigger'])]
            update_idx = pd.Index(update_prev_idx).append(pd.Index(new_rows_idx))

            conditions = [
                selection_data.loc[update_idx, 'selected_keys_updated'] == 1.0,
                selection_data.loc[update_idx, 'selected_keys_updated'] == 0.0,
                selection_data.loc[update_idx, 'selected_keys_updated'].isna()
            ]
            choices_selected = [0.0, np.nan, 1.0]
            choices_decision = [1.0, 0.0, 1.0]

            selection_data.loc[update_idx, 'selected_keys_updated'] = np.select(conditions, choices_selected, default=np.nan)
            selection_data.loc[update_idx, 'decision_updated'] = np.select(conditions, choices_decision, default=np.nan)
            selection_data.loc[update_idx, 'correct_updated'] = (
                selection_data.loc[update_idx, 'selected_keys_updated'] == 
                selection_data.loc[update_idx, 'right_answers_keys']
            ).astype(float)

    # 2) masks
    brief_mask = (
        (selection_data['Time_on_option_(in_seconds)'] < min_fixation_per_trial_s) &
        (selection_data['trigger'] != 'start') &
        (selection_data['trigger'] != 'end')
    )

    long_mask = (
        (selection_data['Time_on_option_(in_seconds)'] >= min_fixation_per_trial_s) &
        (selection_data['trigger'] != 'start') &
        (selection_data['trigger'] != 'end')
    )
    no_decision_mask = long_mask & selection_data['correct_updated'].isna() 

    # 3) Short fixations = not checked
    selection_data.loc[brief_mask, 'figure_checked_updated'] = 0.0

    # 4) long fixations without selection = no-decision
    selection_data.loc[no_decision_mask, 'decision_updated'] = 0.0

    # 5) short fixation = fully excluded if omit_brief_fixations_per_trial
    if omit_brief_fixations_per_trial:
        selection_data.loc[brief_mask, ['decision_updated', 'correct_updated']] = [np.nan, np.nan]

    # 6) long fixation without decision = incorrect (override)
    if flag_no_decisions_false:
        selection_data.loc[no_decision_mask, ['correct_updated']] = 0.0

    # final sorting and column selection
    selection_data = selection_data.sort_values('s_after_experiment_start').reset_index(drop=True)
    columns = [
        'trial_number', 'trigger', 's_after_experiment_start',
        'selected_keys', 'selected_keys_updated', 'right_answers_keys',
        'Time_on_option_(in_seconds)', 'artificial', 'decision', 'decision_updated',
        'figure_checked', 'figure_checked_updated', 'correct', 'correct_updated'
    ]
    return selection_data[columns]

def compute_selection_statistics(corrected_selection_data):
    result = []
    artificial_trials = corrected_selection_data[(corrected_selection_data['artificial'] == True)]['trial_number'].unique()
    not_checked_but_trigger_trials = corrected_selection_data[(corrected_selection_data['figure_checked'] == False) & (~corrected_selection_data['selected_keys'].isna())]['trial_number'].unique()
    
    corrected_selection_data_unique = corrected_selection_data.drop_duplicates(subset = ['trigger', 'trial_number'])

    mask = (
        corrected_selection_data_unique['correct_updated'].notna() &
        (corrected_selection_data_unique['correct'] != corrected_selection_data_unique['correct_updated'])
    )
    
    n_selections = np.sum(corrected_selection_data_unique['correct_updated'].notna())
    n_selections_old = np.sum(corrected_selection_data_unique['correct'].notna())

    updated_trials = corrected_selection_data_unique.loc[mask, 'trial_number'].unique()
    
    checked = np.mean(corrected_selection_data_unique['figure_checked_updated'])
    accuracy = np.mean(corrected_selection_data_unique['correct_updated'])
    
    checked_old = np.mean(corrected_selection_data_unique['figure_checked'])
    accuracy_old = np.mean(corrected_selection_data_unique['correct'])

    result = {
        "artificial_trials": artificial_trials,
        "no_checked_trials": not_checked_but_trigger_trials,
        "updated_trials": updated_trials,
        "checked_updated": checked,
        "accuracy_updated": accuracy,
        "n_selections_updated": n_selections,
        "n_selections_old": n_selections_old,
        "checked_old": checked_old,
        "accuracy_old": accuracy_old
    }
    
    return result

def compute_figure_checked(df_fixations, n_trials_per_block):
    results = []
    for trial, group in df_fixations.groupby('trial_number'):
        # Determine block and relative trial number
        block_number = trial // n_trials_per_block
        trial_number = trial % n_trials_per_block

        present_labels = set(group['label'])
        label_presence = [label in present_labels for label in range(6)]

        row = [block_number, trial_number] + label_presence
        results.append(row)

    columns = ['block_number', 'trial_number'] + [f'figure_{i}' for i in range(6)]
    return pd.DataFrame(results, columns=columns)

def compute_fixations_analysis(df_fixations, n_trials_per_block, base_label=6):
    results = []

    for option_label in range(6):
        for trial, group in df_fixations.groupby('trial_number'):
            # Determine block and relative trial number
            block_number = trial // n_trials_per_block
            trial_number = trial % n_trials_per_block

            # Sort by time
            group = group.sort_values(by='timestamp').reset_index(drop=True)
            timestamps = group['timestamp'].tolist()
            durations = np.diff(timestamps + [timestamps[-1]])  # last one gets 0
            group['duration'] = durations

            labels = group['label'].tolist()

            # Back-and-Forth Count 
            compressed = [labels[0]]
            for l in labels[1:]:
                if l != compressed[-1]:
                    compressed.append(l)

            count = 0
            for i in range(1, len(compressed) - 1):
                if compressed[i] == base_label:
                    prev_label = compressed[i - 1]
                    next_label = compressed[i + 1]
                    if prev_label == option_label and next_label == option_label:
                        count += 1

            # Fixation Metrics 
            total_fixations = len(group)
            base_fixation_count = sum(group['label'] == base_label)
            option_fixation_count = sum(group['label'] == option_label)

            # Time to first base image fixation
            base_fixations = group[group['label'] == base_label]
            if not base_fixations.empty:
                base_first_time = base_fixations.iloc[0]['timestamp'] - group.iloc[0]['timestamp']
            else:
                base_first_time = np.nan

            # Time on option / base
            time_on_option = group.loc[group['label'] == option_label, 'duration'].sum()
            time_on_base = group.loc[group['label'] == base_label, 'duration'].sum()

            # Append Results 
            results.append({
                'block_number': block_number,
                'trial_number': trial_number,
                'figure_num': option_label,
                'Back_and_forth_count': count,
                'Total_fixations': total_fixations,
                'Base_fixation_count': base_fixation_count,
                'Option_fixation_count': option_fixation_count,
                'Base_first_fixation_time_(in_seconds)': base_first_time,
                'Time_on_option_(in_seconds)': time_on_option,
                'Time_on_base_(in_seconds)': time_on_base
            })

    return pd.DataFrame(results)

def create_performance_labels(answers_file, figure_checked):
    records = []

    for trial_idx in range(len(answers_file)):
        # Split pipe-delimited strings into lists
        selected_raw = answers_file.loc[trial_idx, 'selected_keys']
        correct_raw = answers_file.loc[trial_idx, 'right_answers_keys']

        selected = selected_raw.split('|') if isinstance(selected_raw, str) else ['None'] * 6
        correct = correct_raw.split('|') if isinstance(correct_raw, str) else ['False'] * 6

        for fig_idx in range(6):  # Always 6 figures
            selected_answer = selected[fig_idx] if fig_idx < len(selected) else 'None'
            correct_answer = correct[fig_idx] if fig_idx < len(correct) else 'False'
            looked = figure_checked.loc[trial_idx, f'figure_{fig_idx}']

            if selected_answer == 'selected':
                label = 'Correct +' if correct_answer == 'True' else 'False +'
            elif selected_answer == 'not_selected':
                label = 'False -' if correct_answer == 'True' else 'Correct -'
            elif looked:
                label = 'No decision'
            else:
                label = 'Not looked at'

            block_number = answers_file.loc[trial_idx, 'block_number']
            trial_number = answers_file.loc[trial_idx, 'trial_number']

            records.append({
                'block_number': block_number,
                'trial_number': trial_number,
                'figure_num': fig_idx,
                'performance_label': label
            })
            
    return pd.DataFrame(records)


# ===========================
# Eye-tracking functions
# ===========================

def _height2pix(x_h, y_h, screen_width_px=1920, screen_height_px=1080):

        x_limit = screen_width_px / screen_height_px * 0.5

        y_px =int((y_h + 0.5) *screen_height_px)
        x_px =int((x_h + x_limit) *screen_height_px)
        x_px =x_px-screen_width_px
        y_px = screen_height_px - y_px
        return  x_px,y_px

def _find_fixation_label(x, y, image_boxes):
    for idx, box in enumerate(image_boxes):
        if box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]:
            return idx
    return None

def _create_image_boxes():
    positions_norm = [(-0.5, 0), (0, 0), (0.5, 0), 
                  (-0.5, -0.35), (0, -0.35), (0.5, -0.35),
                  (0, 0.325)] #(0, 0.325) is the base figure
    
    img_size_norm = 0.25 * 1.15
    
    img_size_pix = _height2pix(0.25, 0.5-img_size_norm*1.07)[1]
    image_boxes = []

    for x_norm, y_norm in positions_norm:
        center_x, center_y = _height2pix(x_norm, y_norm)
        half = img_size_pix // 2
        box = {
            "x_min": center_x - half,
            "x_max": center_x + half,
            "y_min": center_y - half,
            "y_max": center_y + half
        }
        image_boxes.append(box)
    
    return image_boxes

def eye_fixation_extraction(mouse_log):
    """
    Creates eye fixation statistics based on mouse_log. E.g. How long a certain image was looked at, etc.
    """
    image_boxes = _create_image_boxes()
    
    start_end_indices = []
    current_start = None

    for idx, event in enumerate(mouse_log["event"]):
        if str(event).strip().upper() == "START":
            current_start = idx
        elif str(event).strip().upper() == "END" and current_start is not None:
            start_end_indices.append((current_start, idx))
            current_start = None
    
    fixation_rows = []

    for i in range(len(start_end_indices)):
        (start_idx, end_idx) = start_end_indices[i]

        segment = mouse_log.iloc[start_idx:end_idx+1].copy()

        segment['label'] = segment.apply(
            lambda row: _find_fixation_label(row['mouse x'], row['mouse y'], image_boxes), axis=1
        )

        segment = segment.dropna(subset=['label'])
        segment['label'] = segment['label'].astype(int)

        segment['trial_number'] = i

        fixation_rows.append(segment[['trial_number', 'timestamp', 'label']])


    all_fixations = pd.concat(fixation_rows, ignore_index=True)

    return all_fixations

# ===========================
# Extracting training data functions
# ===========================

def sync_eeg_and_eye_tracker(mouse_log: pd.DataFrame, onsets, triggers):
    
    # get start events and their time difference between mne (eeg) and eye-tracker
    starts_mne_onset = onsets[triggers == 'start']
    starts_eyetrack = mouse_log[mouse_log['event'] == "START"]['timestamp']
    start_diffs = starts_eyetrack - starts_mne_onset

    # find first start event in eye-tracker
    which_diff = np.cumsum(mouse_log['event'] == 'START') - 1
    start_idx = np.where(which_diff == 0)[0][0]
    which_diff = which_diff[which_diff != -1] # remove pre start diffs

    # pick start_diffs based on trial (same dimension as eye-tracker data)
    t_eyetracker = mouse_log['timestamp'].values[start_idx:]
    diffs = start_diffs.values[which_diff]

    mouse_log_synced = mouse_log[start_idx:].copy()
    mouse_log_synced['eeg_time'] = t_eyetracker - diffs
    return mouse_log_synced

def extract_window_times(fixation_data_synced, window_s, step_s, min_fixation_s = 0.001):
    """
    Extract window times (start + end) whenever a figure is fixated. Excludes fixations when they are shorter than `min_fixation_s`.
    """
    times = fixation_data_synced['eeg_time'].values
    triggers = fixation_data_synced['label'].values
    trials = fixation_data_synced['trial_number'].values
    
    # filter for min_fixation_s
    label_changes = np.concatenate(([True], triggers[1:] != triggers[:-1]))
    change_indices = np.where(label_changes)[0]
    start_indices = change_indices[:-1]
    end_indices = change_indices[1:] - 1

    start_indices = np.append(start_indices, change_indices[-1])
    end_indices = np.append(end_indices, len(triggers) - 1)

    start_times = times[start_indices]
    end_times = times[end_indices]
    durations = end_times - start_times

    valid_segments = durations >= min_fixation_s

    keep_idx = np.hstack([np.arange(start, end + 1) for keep, start, end in zip(valid_segments, start_indices, end_indices) if keep])

    times = times[keep_idx]
    triggers = triggers[keep_idx]
    trials = trials[keep_idx]
     
    n = len(times)

    start_times = []
    end_times = []
    out_triggers = []
    out_trials = []

    gaze_switches = np.concatenate(([0], np.where(np.diff(triggers) != 0)[0] + 1))

    for i in gaze_switches:
        start_time = times[i]
        start_trigger = triggers[i]
        current_trigger = start_trigger
        current_trial = trials[i]
        update_idx = i
        if start_trigger != 6:
            while current_trigger == start_trigger and update_idx + 1 < n:
                out_triggers.append(current_trigger)
                out_trials.append(current_trial)
                start_times.append(start_time)
                end_times.append(start_time + window_s)
                
                update_time = start_time + step_s
                update_idx = np.searchsorted(times, update_time) - 1
                current_trigger = triggers[update_idx]
                start_time = update_time
        
    start_times = np.array(start_times, dtype=float)
    end_times = np.array(end_times, dtype=float)
    out_triggers = np.array(out_triggers, dtype=int)
    out_trials = np.array(out_trials, dtype=int)
    
    return start_times, end_times, out_triggers, out_trials

def extract_training_data(data,
                          target = 'selection',
                          window_ms = 250,
                          step_ms = 50,
                          use_window = True,
                          selection_window_ms = 250,
                          omit_brief_fixations_per_trial = True,
                          min_fixation_per_trial_ms = 1000,
                          flag_no_decisions_false = True,
                          min_fixation_ms = 100
                          ):
    """
    Extracts training data (X, y) from EEG by windowing gaze switches (eye-tracking data). 
    Also applies correct_selections to correct for involuntary selections.
    """
    fixation_data = data['fixation_data']
    onsets = data['onsets']
    triggers = data['triggers']
    mouse_log = data['mouse_log']
    eeg = data['eeg']
    
    fs = data['eeg'].info['sfreq']
    lowpass = data['eeg'].info['lowpass']
    highpass = data['eeg'].info['highpass']
    
    # synchronize eeg and eye-tracker time
    mouse_log_synced = sync_eeg_and_eye_tracker(mouse_log, onsets, triggers)
    fixation_data_synced = fixation_data.merge(mouse_log_synced[['timestamp', 'eeg_time']])     
    
    # extract eeg window times
    start_times, end_times, triggers, trials = extract_window_times(fixation_data_synced, window_ms / 1000, step_ms / 1000, min_fixation_ms / 1000)
    
    # extract eeg windows
    times = eeg.times
    eeg_data = eeg.get_data()

    # extract selections
    selection_data = build_selection_data(data)

    # correct_selections
    selection_data = correct_selections(
        selection_data, 
        use_window=use_window,
        selection_window_s=selection_window_ms/1000,
        omit_brief_fixations_per_trial=omit_brief_fixations_per_trial,
        min_fixation_per_trial_s=min_fixation_per_trial_ms/1000,
        flag_no_decisions_false=flag_no_decisions_false 
        )

    # drop unused selections
    if target == 'selection':
        selection_data = selection_data[['trial_number', 'trigger', 'correct_updated']].dropna(subset=['correct_updated']).drop_duplicates(subset=["trial_number", "trigger"])
    else: 
        selection_data = selection_data[['trial_number', 'trigger', 'decision_updated']].dropna(subset=['decision_updated']).drop_duplicates(subset=["trial_number", "trigger"])
    selection_data["trigger"] = selection_data["trigger"].str.extract(r"fig(\d+)_select").astype(int)

    df_query = pd.DataFrame({
        "trial_number": trials, 
        "trigger": triggers
    })

    # extract y
    merged = df_query.merge(selection_data, on=["trial_number", "trigger"], how="left")
    if target == 'selection':
        na_idx = merged['correct_updated'].isna()
        y = merged['correct_updated'].values[np.array(~na_idx)]
    else:
        na_idx = merged['decision_updated'].isna()
        y = merged['decision_updated'].values[np.array(~na_idx)]
    y = y.astype(np.int32)

    # extract start_idxs
    start_idxs = (np.searchsorted(times, start_times) - 1)[~na_idx]
    end_idxs = (np.searchsorted(times, end_times) - 1)[~na_idx]

    expected_samples = int(round(window_ms / 1000 * fs))
    window_samples = end_idxs[0] - start_idxs[0]     
    if not (expected_samples - 1 <= window_samples <= expected_samples + 1):
        raise AssertionError(
            f"Expected ~{expected_samples} samples for {window_ms} ms, got {window_samples}"
        )
        
    # truncate to expected_samples (sometimes it might take one more sample)
    X = np.empty((len(start_idxs), eeg_data.shape[0], expected_samples), dtype=np.float32)
    for i, (start, end) in enumerate(zip(start_idxs, end_idxs)):
        window = eeg_data[:, start:end]
        X[i] = window[:, :expected_samples]
    
    # provide info
    info = {
        'id': data['id'],
        'trials': trials[~na_idx],
        'target': target,
        'fs': fs,
        'lowpass': lowpass, 
        'highpass': highpass,
        'window_ms': window_ms,
        'step_ms': step_ms,
        'shape': X.shape,
        # correct_selection
        'use_window':use_window,
        'selection_window_ms':selection_window_ms,
        'omit_brief_fixations_per_trial':omit_brief_fixations_per_trial,
        'min_fixation_per_trial_ms':min_fixation_per_trial_ms,
        'min_fixation_ms':min_fixation_ms,
        'flag_no_decisions_false':flag_no_decisions_false 
    }
    
    return (X, y), info

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np


def split_by_trial(y, trial_label, n_splits=1, val_size=None, test_size=0.15, 
                   stratify=False, random_state=42):
    """
    Split data into train/test (and optional folds) based on trial IDs. 
    Has a costume stratify method, where trials are assigned to quartiles based on class balance ratio
    (e.g. 0.9 ratio is assigned to 4, 0.4 -> 2) and then split trying to keep the same amount of quartiles in test and train set.
    """
    unique_trials = np.unique(trial_label)
    rng = np.random.default_rng(seed=42)
    shuffled_trials = rng.permutation(unique_trials)

    # stratification target
    if stratify:
        trial_frac = np.array([np.mean(y[trial_label == t]) for t in shuffled_trials])
        bins = [0, 0.5, 1.0]
        strat_labels = np.digitize(trial_frac, bins)
    else:
        strat_labels = None

    if n_splits == 1:
        # single train/test split
        trv_trials, test_trials = train_test_split(
            shuffled_trials,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_labels
        )
        
        if val_size is not None:
            test_size = val_size / (1.0 - test_size)
            
            if stratify:
                trv_mask = np.isin(shuffled_trials, trv_trials)
                strat_labels_trv = strat_labels[trv_mask]
            else:
                strat_labels_trv = None
            
            train_trials, val_trials = train_test_split(
                trv_trials,
                test_size=test_size,
                random_state=random_state,
                stratify=strat_labels_trv
            )
            
        else:
            train_trials = trv_trials
            
        train_mask = np.isin(trial_label, train_trials)
        val_mask = np.isin(trial_label, val_trials) if val_size is not None else None
        test_mask = np.isin(trial_label, test_trials)
        return train_mask, test_mask, val_mask
    else:
        # cross-validation folds
        if stratify:
            cv = StratifiedKFold(n_splits=n_splits)
            split_iter = cv.split(shuffled_trials, strat_labels)
        else:
            cv = KFold(n_splits=n_splits)
            split_iter = cv.split(shuffled_trials)

        fold_splits = []
        for train_idx, val_idx in split_iter:
            train_trials = shuffled_trials[train_idx]
            val_trials = shuffled_trials[val_idx]

            train_mask = np.isin(trial_label, train_trials)
            test_mask = np.isin(trial_label, val_trials)

            fold_splits.append((train_mask, test_mask))
        return fold_splits
                            
# ===========================
# Plotting functions
# ===========================

def plot_trajectory(data, id, trial, time_range, step):
    plt.close('all')  # close previous figures
    mouse_log = data[id]['mouse_log'].copy()
    mouse_log['trial'] = np.cumsum(mouse_log['event'] == 'START') - 1
    df_trial = mouse_log[mouse_log['trial'] == trial].copy()
    if df_trial.empty:
        print(f"No data found for trial {trial}")
        return

    df_trial = df_trial.iloc[::int(step)]
    t0, t_end = df_trial['timestamp'].iloc[0], df_trial['timestamp'].iloc[-1]
    df_trial['norm_time'] = (df_trial['timestamp'] - t0) / (t_end - t0)
    df_trial = df_trial[(df_trial['norm_time'] >= time_range[0]) & (df_trial['norm_time'] <= time_range[1])]
    if df_trial.empty:
        print(f"No data in time range {time_range} for trial {trial}")
        return

    labels = df_trial['label']
    unique_labels = np.sort(labels.dropna().unique())
    
    # Categorical colormap with one color per label + one for NaN
    base_cmap = matplotlib.cm.get_cmap('tab10')
    n_labels = len(unique_labels)
    colors = [base_cmap(i) for i in range(n_labels)]
    colors.append((0.5, 0.5, 0.5, 1))  # Gray for NaN

    cmap = ListedColormap(colors)

    # Map labels to color indices, NaN -> last index (gray)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    color_indices = labels.map(label_to_idx).fillna(n_labels).astype(int).to_numpy()

    # Norm and boundaries for discrete colorbar
    boundaries = np.arange(-0.5, n_labels + 1.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N) # type: ignore

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(df_trial["mouse x"], df_trial["mouse y"],
                    c=color_indices, cmap=cmap, norm=norm,
                    s=30)
    ax.plot(df_trial["mouse x"], df_trial["mouse y"], color='gray', alpha=0.5, linestyle='-')

    # Create colorbar with ticks centered on each color
    cbar = plt.colorbar(sc, ax=ax, boundaries=boundaries, ticks=np.arange(n_labels + 1))
    
    # Tick labels for each label + NaN
    tick_labels = [str(lbl) for lbl in unique_labels] + ['NaN (no label)']
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Fixation Label")

    # Draw image boxes (your function)
    for i, box in enumerate(_create_image_boxes()):
        width, height = box['x_max'] - box['x_min'], box['y_max'] - box['y_min']
        rect = patches.Rectangle(
            (box['x_min'], box['y_min']), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            box['x_min'] + width / 2,
            box['y_min'] + height / 2,
            str(i), color='red',
            ha='center', va='center', fontsize=12, weight='bold'
        )

    ax.set_title(f"Mouse Trajectory – Trial {trial}, Time {time_range[0]:.2f}–{time_range[1]:.2f}")
    ax.invert_yaxis()
    ax.grid(True)
    plt.show()

def plot_selection_data(data, id, trial, use_window, selection_window, omit_brief_fixations_per_trial, min_fixation_per_trial_ms, flag_no_decisions_false):
    """
    Plots selection_data for observing when selected keys were pressed.
    """
    plt.close('all')  # close previous figures
    plt.figure(figsize=(10, 4))
    
    selection_data = build_selection_data(data[id])
    # correct_triggers?
    df_global = correct_selections(selection_data, 
                                   use_window=use_window,
                                   selection_window_s=selection_window,
                                   omit_brief_fixations_per_trial=omit_brief_fixations_per_trial,
                                   min_fixation_per_trial_s=min_fixation_per_trial_ms,
                                   flag_no_decisions_false=flag_no_decisions_false)

    df = df_global[df_global['trial_number'] == trial].copy()

    trigger_order = ['start'] + [f'fig{i}_select' for i in range(6)] + ['end']
    df['trigger'] = pd.Categorical(df['trigger'], categories=trigger_order, ordered=True)
    df = df.sort_values('trigger')
    
    # Plot all real (non-artificial) triggers
    real_df = df[df['artificial'] != True]
    plt.scatter(
        real_df['s_after_experiment_start'],
        real_df['trigger'],
        s=80,
        c=real_df['correct_updated'].map({1.0: 'green', 0.0: 'red'}).fillna('gray'),
        label='Real'
    )
    
    # plot window
    if use_window:
        trial_start = df['s_after_experiment_start'].min()
        trial_end = df['s_after_experiment_start'].max()
        plt.axvspan(trial_start, trial_start + selection_window, color='yellow', alpha=0.1, label='Correction Window')
        plt.axvspan(trial_end - selection_window, trial_end, color='yellow', alpha=0.1, label='Correction Window')
    
        # Plot artificial triggers with asterisk marker
        artificial_df = df[df['artificial'] == True]
        plt.scatter(
            artificial_df['s_after_experiment_start'],
            artificial_df['trigger'],
            s=120,
            c=artificial_df['correct_updated'].map({1.0: 'green', 0.0: 'red'}).fillna('gray'),
            label='Artificial'
        )   

    plt.xlabel('Seconds after experiment start')
    plt.ylabel('Trigger')
    plt.title(f'Triggers for Trial {trial}')
    plt.grid(True)
    
    legend_elements = [
    Patch(facecolor='green', label='Correct (1)'),
    Patch(facecolor='red', label='Not correct (0)'),
    Patch(facecolor='yellow', alpha=0.1, label='Correction Window')
    ]
    plt.legend(handles=legend_elements)
    
    # check global statistics and trials of interest
    result = compute_selection_statistics(df_global)
        
    for key, value in result.items():
        print(f"{key}: {np.round(value, 2)}")
    
    plt.show()
    df_sorted = df.sort_values('s_after_experiment_start')
    #display(HTML(df_sorted.to_html()))
    

    # Scrollable HTML table (both vertical and horizontal)
    scrollable_table = f"""
    <div style="max-height:400px; max-width:100%; overflow-y:auto; overflow-x:auto; border:1px solid lightgray;">
        {df_sorted.to_html(index=False)}
    </div>
    """
    display(HTML(scrollable_table))

def plot_windows(data, id, trial, window_ms=1000, step_ms=250, min_fixation_ms=100):
    plt.close('all')  # close previous figures

    # get data
    selection_data = build_selection_data(data[id])
    fixation_data = data[id]['fixation_data']
    onsets = data[id]['onsets']
    triggers = data[id]['triggers']
    mouse_log = data[id]['mouse_log']
    mouse_log_synced = sync_eeg_and_eye_tracker(mouse_log, onsets, triggers)
    fixation_data_synced = fixation_data.merge(mouse_log_synced[['timestamp', 'eeg_time']])
    start_times, end_times, triggers, trials = extract_window_times(fixation_data_synced, window_ms / 1000, step_ms / 1000, min_fixation_ms / 1000)
    ref_start_times, _, _, _ = extract_window_times(fixation_data_synced, window_ms / 1000, step_ms / 1000, 0)

    # plot
    trial_mask = trials == trial

    # Unique triggers for the selected trial
    trial_triggers = triggers[trial_mask]
    unique_triggers = np.unique(trial_triggers)

    plt.figure(figsize=(15, 8))

    for i, trig in enumerate(unique_triggers):
        mask = trial_mask & (triggers == trig)
        y = i  # y position for this trigger
        # Add jitter to y for each window to avoid overlap
        n = np.sum(mask)
        y_jitter = y + (np.concatenate([np.arange(-n/4, n/4), np.arange(-n/4, n/4)])) * 0.01  # jitter in [-0.1, 0.1]
        for (s, e, yj) in zip(start_times[mask], end_times[mask], y_jitter):
            plt.vlines(s, yj - 0.1, yj + 0.1, color='b')
            # Optionally, add horizontal lines or end markers as needed
            plt.hlines(yj, s, e, color='gray', linewidth=1)
            plt.vlines(e, yj - 0.1, yj + 0.1, color='r')

    filtered_fixation = fixation_data_synced[fixation_data_synced['trial_number'] == trial]
    filtered_selection = selection_data[selection_data['trial_number'] == trial]
    filtered_selection["fig_id"] = filtered_selection["trigger"].str.extract(r"fig(\d+)_select")
    filtered_selection["fig_id"] = pd.to_numeric(filtered_selection["fig_id"], errors="coerce")
    filtered_selection = filtered_selection.dropna(subset=["fig_id"]).reset_index(drop=True)
    
    plt.plot(filtered_fixation['eeg_time'], filtered_fixation['label'], color='gray')
    plt.scatter(filtered_selection['s_after_experiment_start'], filtered_selection['fig_id'], color='green', s=100, zorder=5)
    plt.xlabel('Time (s)')
    plt.ylabel('Trigger')
    plt.title(f'Windows (n = {len(start_times[trial_mask])})  for trial {trial} - % filtered by min fixations = {len(start_times) / len(ref_start_times)})')
    plt.grid(axis='x')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='selection'),
        Line2D([0], [0], color='b', lw=2, label='start window'),
        Line2D([0], [0], color='r', lw=2, label='end window'),
        Line2D([0], [0], color='gray', lw=2, label='gaze (on labels)')
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()
    
def plot_roc_and_pr_curves(y_true, y_score):
    """
    Plots ROC and Precision-Recall curves given true labels and predicted probabilities.
    Avoids recomputing predictions inside.
    """

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    positive_rate = np.mean(y_true)  # baseline precision

    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.hlines(positive_rate, xmin=0, xmax=1, colors='k', linestyles='--', label="Baseline")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

    
# ===========================
# Evaluation
# ===========================

def evaluate_model(participants_info, participant: str, target: str, h_freq: int, ica: bool, fs: int,
                   config_type: str, metric: str):
    # Get participant info
    info = next(p for p in participants_info if p['Participant'] == participant)
    session = info['Session']
    task = info['Task']

    # Paths
    base_path = Path("..") / f"data/sub-{participant}/"
    data_dir = base_path / "training_data" / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}" / config_type
    model_dir = base_path / "models" / f"{target}-{fs}hz_1-{h_freq}_ica-{ica}" / config_type

    # Load data
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")

    with open(data_dir / "data_info.pkl", 'rb') as f:
        data_config = pickle.load(f)
        
    with open(model_dir / "perm_results.pkl", 'rb') as f:
        perm_results = pickle.load(f)

    # load splits
    splits_path = model_dir / "splits.pkl"
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)
            
    fold_splits_path = model_dir / "fold_splits.pkl"
    with open(fold_splits_path, "rb") as f:
        fold_splits = pickle.load(f)
    
    model = load_model(model_dir / 'model.h5')

    with open(model_dir / "history.pkl", "rb") as f:
        history = pickle.load(f)
    
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # extract data
    train_mask, test_mask, val_mask = splits
    X_te, y_te = X[test_mask], y[test_mask]
    
    # normalize
    X_te = scaler.transform(X_te)
    
    print("\nData Config:")
    for k, v in data_config.items():
        print(f"{k}: {v}")

    # --- Evaluate model ---
    result = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nTest Loss: {result[0]:.4f}")
    print(f"Test Accuracy: {result[1]:.4f}")

    # Predictions
    predicted_test = model.predict(X_te, verbose=0)
    y_test_pred = np.argmax(predicted_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_te, y_test_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC and PR curves
    y_score = predicted_test[:, 1]
    plot_roc_and_pr_curves(y_te, y_score)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_te, y_test_pred))

    # Training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()

    # --- Permutation Test Results ---
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    axes = axes.flatten()

    idx = perm_results["metrics"].index(metric)
    N = perm_results['N']

    # --- Original model ---
    if "original" in perm_results:
        obs_stats = perm_results["original"]["obs_stats"][idx]
        null_stats = perm_results["original"]["null_stats"][idx]
        p_value = np.mean(perm_results["original"]["p_values"][idx])
        
        axes[0].hist(null_stats, bins=50, alpha=0.7)
        axes[0].axvline(obs_stats, color='red', linestyle='dashed', linewidth=2)
        axes[0].set_xlabel(metric)
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Original Model\nN = {N}, p ≈ {p_value:.4f}")

    # --- Fold models ---
    if "folds" in perm_results:
        # aggregated   
        obs_stats = perm_results["folds_aggregated"]["obs_stats"][idx]
        null_stats = perm_results["folds_aggregated"]["null_stats"][idx]
        p_value = np.mean(perm_results["folds_aggregated"]["p_values"][idx])

        axes[-1].hist(null_stats, bins=50, alpha=0.7)
        axes[-1].axvline(obs_stats, color='red', linestyle='dashed', linewidth=2)
        axes[-1].set_xlabel(metric)
        axes[-1].set_ylabel("Frequency")
        axes[-1].set_title(f"Aggregated Model\nN = {N}, p ≈ {p_value:.4f}")
    
        # per fold
        for i, (obs_stat, null_stats) in enumerate(zip(perm_results["folds"]["obs_stats"][idx],
                                                        perm_results["folds"]["null_stats"][idx])):
            p_value = (1 + np.sum(null_stats >= obs_stat)) / (1 + len(null_stats))

            ax = axes[i+1]
            ax.hist(null_stats, bins=50, alpha=0.7)
            ax.axvline(obs_stat, color='red', linestyle='dashed', linewidth=2)
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Fold {i+1}\nN = {N}, p ≈ {p_value:.4f}")

    plt.tight_layout()
    plt.show()

def permutation_test(y_score, y_test, fold_te_size = None, metric = 'pr-auc', N = 10000):
    # select metric
    if metric == 'pr-auc':
        T_obs = average_precision_score(y_test, y_score)
        stat_fn = lambda y_true, y_pred_scores: average_precision_score(y_true, y_pred_scores)
    elif metric == 'roc-auc':
        T_obs = roc_auc_score(y_test, y_score)
        stat_fn = lambda y_true, y_pred_scores: roc_auc_score(y_true, y_pred_scores)
    elif metric == 'balanced_accuracy':
        y_pred = (y_score >= 0.5).astype(int)
        T_obs = balanced_accuracy_score(y_test, y_pred)
        stat_fn = lambda y_true, y_pred_scores: balanced_accuracy_score(y_true, (y_pred_scores >= 0.5).astype(int))
    else: 
        ValueError("metric must be 'pr-auc', 'roc-auc' or 'balanced_accuracy")
    
    # run permutations
    perm_stats = []
    for _ in range(N):       
        if fold_te_size is None:
            y_perm = np.random.permutation(y_test)
        # permutate within fold
        else:
            start = 0
            y_perm = np.empty_like(y_test)
            for size in fold_te_size:
                end = start + size
                y_perm[start:end] = np.random.permutation(y_test[start:end])
                start = end
            
        perm_stats.append(stat_fn(y_perm, y_score))
    perm_stats = np.array(perm_stats)
    
    p_value = (1 + np.sum(perm_stats >= T_obs)) / (1 + N)
    
    return T_obs, perm_stats, p_value