import os

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/submissions', exist_ok=True)
os.makedirs('notebooks/robert', exist_ok=True)
os.makedirs('notebooks/chu', exist_ok=True)
os.makedirs('notebooks/kervy', exist_ok=True)
os.makedirs('notebooks/franco', exist_ok=True)
os.makedirs('src', exist_ok=True)

# Download data

# kaggle competitions download -c bbvadatachallenge-recomendador -p data/raw

# Notebook header

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# import os
# import numpy as np, pandas as pd
# import matplotlib.pyplot as plt, seaborn as sns
# from tqdm import tqdm, tqdm_notebook
# from pathlib import Path
# # pd.set_option('display.max_columns', 1000)
# # pd.set_option('display.max_rows', 400)
# sns.set()

# os.chdir('..')

# DATA = Path('data')
# RAW  = DATA/'raw'
# PROCESSED = DATA/'processed'
# SUBMISSIONS = DATA/'submissions'