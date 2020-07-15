import torch
from torch.utils.data import Dataset

from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv('./data/train.csv')
    train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    targets = train['selected_text']
    del train['selected_text']
    del train['textID']
    train.dropna(inplace=True)
    # train['sentiment'] = train['sentiment'].map(dict(neutral=0, positive=1, negative=-1))
    return train, targets

class TweetSentiment(Dataset):
    def __init__(self, mode='train'):
        self.X, self.Y = load_data()
        self.max_length = max(len(s) for s in self.X['text'])

    def __getitem__(self, index):
        return {'X': self.X.iloc[index], 'Y': self.Y.iloc[index], 'max_length': self.max_length}

    def __len__(self):
        return len(self.X)