import argparse
import yaml 
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from tqdm.auto import tqdm

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from train_utils import kfold

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def parse_args():
    parser = argparse.ArgumentParser()


    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("--device", type=int, default='0', required=False)   
    parser_args, _ = parser.parse_known_args(sys.argv)
    return parser.parse_args()

# ------------------------------------------ ------------------------------------------- #

if __name__ == "__main__":
    cfg = parse_args()

    with open(cfg.config, 'r') as f:
        args = yaml.safe_load(f)

    args = SimpleNamespace(**args)
    args.device = cfg.device
    Path(args.checkpoints_path).mkdir(parents=True,exist_ok=True)

    train_df = pd.read_csv('data/feedback-prize-english-language-learning/train.csv')

    TARGET = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar','conventions']
    seeds = [42]
    for K in [5]:  
        for seed in seeds:
            mskf = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
            name = f"fold_k_{K}_seed_{seed}"
            train_df[name] = -1
            for fold, (trn_, val_) in enumerate(mskf.split(train_df, train_df[TARGET])):
                train_df.loc[val_, name] = fold

    kfold(args,train_df)

