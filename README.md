# DermFM-Zero

### D1 Reproduction 

Original Repo: https://github.com/SiyuanYan1/DermFM-Zero/tree/main#-quick-start

### Setup

```bash
git clone https://github.com/DanielJ56/D1-Reviewer.git
cd DermFM-Zero
conda create -n dermfm-zero python=3.9.20
conda activate dermfm-zero
pip install -r requirements.txt
```

### Run

Change line 10 in dermfmzero_feeding.py IMAGE_PATH to local path of images

```bash
python dermfmzero_feeding.py
```

Outputs results to Master_Results.csv. Upon resuming, skips rows in Masters_Results.csv that are already filled.