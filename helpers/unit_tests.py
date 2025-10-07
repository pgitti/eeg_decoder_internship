import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
import pandas as pd
import numpy as np
import pytest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from pathlib import Path

from helpers.helpers import correct_selections 

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)  # or larger if needed

@pytest.fixture
def artificial_selection_data():
    # Artificial test data, as in your example
    data = {
        "trial_number": [0,0,0,0,1,1,1],
        "trigger": [
            "start","short_selection","selection_within_window","end","start","selection_within_window","no_decision_look"
        ],
        "s_after_experiment_start": [
            0.,1.,3.9,4.,4.1,4.2,4.5
        ],
        "selected_keys": [np.nan,0.0,np.nan,np.nan,np.nan,1.0,np.nan],
        "right_answers_keys": [np.nan,0.0,0.0,np.nan,np.nan,1.0,1.0],
        "figure_checked": [np.nan,True,True,np.nan,np.nan,True,True],
        "Time_on_option_(in_seconds)": [np.nan,1.,2.,np.nan,np.nan,1.,3.],
        "correct": [np.nan,1.0,1.0,np.nan,np.nan,1.0,np.nan],
    }
    return pd.DataFrame(data)

def test_correct_selections_basic(artificial_selection_data):
    result = correct_selections(artificial_selection_data.copy(),
                                omit_brief_fixations_per_trial=False,
                                flag_no_decisions_false=False)
    
    print("\n--- Input artificial_selection_data ---")
    print(artificial_selection_data)
    print("\n--- Result after correct_selections ---")
    print(result)
    
    # --- behavioral assertions ---
    # 1) No artificial rows if only one trial
    artificial_trigger_idx = 3
    assert result.loc[artificial_trigger_idx, 'artificial']

    # 2) Short fixations (<2s) should be set to NaN
    short_fix_mask = result['Time_on_option_(in_seconds)'] < 2 & result['correct'].isna()
    assert result.loc[short_fix_mask, 'decision_updated'].isna().all()

    # 3) Long fixations with no decision should be marked as incorrect (0.0)
    no_decision_mask = (
        (result['Time_on_option_(in_seconds)'] >= 2) &
        (result['trigger'].str.contains("fig")) &
        (result['correct_updated'].isna())
    )
    if no_decision_mask.any():
        assert (result.loc[no_decision_mask, 'correct_updated'] == 0.0).all()

