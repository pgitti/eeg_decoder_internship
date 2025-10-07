# Internship Project – EEG Decoding

## 🔍 Project Overview

This is a 250-hour research internship project in the Neuroinformatics group at the University of Vienna. The goal of the project was to find an EEG correlate of when people are struggeling in a mental rotation task ([ref to project when published]).  

The repository includes exploring and preprocessing of the EEG data and training and evaluation of ML models. Notebooks were generally used for exploration of the data, while scripts implemented the insights created by the notebooks. Both notebooks and scripts are numbered to indicate their execution order.

## 📁 Folder Structure

- `data/` – Raw and processed data (not committed, because sensitive data)
- `helpers/` – Helper functions
- `notebooks/` – Jupyter notebooks for analysis
- `scripts/` – Python scripts

## 🛠️ Setup Instructions

1. Clone the repo:
```bash
git clone https://github.com/pgitti/eeg_decoder_project.git
cd eeg_decoder_project
```

2. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate eeg_decoder
# conda env remove -n eeg_decoder
```

3. Unit tests
```bash
pytest helpers/unit_tests.py -v -s
```
I have used this environment throughout the project, because switching between notebooks and scripts caused problems.

## 🛠️ Training on server

1. Enter VPN

2. Send data (on Windows):

```bash
wsl
rsync -av --progress \
      --exclude '.git/' \
      --exclude 'data/' \
      /path/to/eeg_decoder_project/ \
      [user]@[ip]:~/eeg_decoder_project/

# and back
rsync -av --progress \
      --exclude '.git/' \
      [user]@[ip]:~/eeg_decoder_project/data/ \
      /path/to/eeg_decoder_project/data/ 
```

3. Generate training data, train models on GPU and do permutation tests on server (faster):

```bash
ssh [user]@[ip]
cd eeg_decoder

conda env create -f environment_srv.yml
conda activate eeg_decoder

nohup python scripts/05_train_models.py > logs/training.out 2>&1 &

# when finished, run permutation test
nohup python scripts/06_permutation_tests.py > logs/permutation.out 2>&1 &

nvidia-smi # to see gpu processes + PID
kill [PID]
```

If something related to Tensorflow does not work, consult https://www.tensorflow.org/install/pip#linux_setup.

