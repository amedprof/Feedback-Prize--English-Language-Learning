
import os
import gc
import glob
import sys
import time
import json
import random

import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
import joblib

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# from plotnine import *

import warnings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
warnings.simplefilter("ignore")



import torch
import pickle


def make_sub(df):
    df = df.reset_index(drop=True)
    df[["discourse_id",'Ineffective','Adequate','Effective']].to_csv('submission.csv',index=False)

#==============================================================================================
def save_pickle(name,var):
    # print(f"Saving {name} ....")
    with open(name+'.pkl','wb') as fout:
        pickle.dump(var,fout)
    fout.close()
    
#==============================================================================================    
def load_pickle(name):
    # print(f"Loading {name} .....")
    with open(name,'rb') as fin:
        md = pickle.load(fin)
    fin.close()
    return md
#==============================================================================================
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False


