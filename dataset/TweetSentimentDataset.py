import torch
from torch.utils.data import Dataset
from config import hyper_params

from transformers import *

import pandas as pd
import numpy as np

OFFSET_FOR_ENCODING = 4
MAX_LENGTH = hyper_params['max_length']
MAX_CHAR_TWEET = 141
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def subfinder(mylist, pattern):
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            return i, i+len(pattern)

def process_data(sentiment, text, selected_text):
    start_index, end_index = subfinder(text, selected_text)
    text_in_selected_text = [0] * len(text)
    for idx in range(start_index, end_index):
        text_in_selected_text[idx] = 1

    token_text = tokenizer.encode_plus(sentiment, text, return_attention_mask=True,
                                       return_token_type_ids=True, max_length=MAX_LENGTH,
                                       pad_to_max_length=True, return_offsets_mapping=True,
                                       truncation=True)

    start = text.find(selected_text)
    end = start + len(selected_text)
    # part_of_selected_text = []
    # # idx is the index of the word where the character was found to be part
    # # of selected_text. offset1 and offset2 are just of the original string encoded
    # # for each token. Eg) tokenizer.encode('I am the dog'), offset[2] = (2, 3). NOTE: offset[x] = (a,b) means token 2 maps to  Note we do 3:-1
    # # because first few are <s>Sentiment</s></s> therefore we start after those useless parts
    # SKIP_USELESS_TOKENS = 4
    # for idx, (offset1, offset2) in enumerate(token_text['offset_mapping'][SKIP_USELESS_TOKENS:-1]):
    #     # We do sum here because if any character is a 1, that means the whole word is a 1
    #     if (offset2 - offset1) > 0 and sum(text_in_selected_text[offset1: offset2]) > 0:
    #         part_of_selected_text.append(idx+SKIP_USELESS_TOKENS)
    return {
        'start': start,
        'end': end,
        'input_ids': token_text['input_ids'],
        'attention_mask': token_text['attention_mask'],
        'token_type_ids': token_text['token_type_ids'],
        'offset_mapping': token_text['offset_mapping']
    }

def load_data(mode):
    data = pd.read_csv(f'./data/{mode}.csv')
    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    targets = data['selected_text']
    del data['selected_text']
    del data['textID']
    data.dropna(inplace=True)
    targets.dropna(inplace=True)
    return data, targets

def construct_test_output(start, end):
    actual_start = [0] * 285
    actual_end = [0] * 285
    actual_start[start] = 1
    actual_end[end] = 1
    return actual_start, actual_end

class TweetSentiment(Dataset):
    def __init__(self, mode='train'):
        self.X, self.Y = load_data(mode)
        self.max_length = max(len(s) for s in self.X['text'])

    def __getitem__(self, index):
        X = self.X.iloc[index]
        Y = self.Y.iloc[index]
        obj = process_data(X['sentiment'].strip(), X['text'].strip(), Y.strip())
        # start, end = construct_test_output(obj['start'], obj['end'])
        start, end = obj['start'], obj['end']
        return {
            'original_tweet': X['text'].strip(),
            'sentiment':X['sentiment'].strip(),
            'selected_sentence': Y.strip(),
            'original_input_ids': obj['input_ids'],
            'offset_mapping': torch.LongTensor(obj['offset_mapping']),
            # size should be (batch_size, sequence_length) for roberta inputs
            'input_ids': torch.tensor(obj['input_ids'], dtype=torch.long, device=device),
            'token_type_ids': torch.tensor(obj['token_type_ids'], dtype=torch.long, device=device),
            'attention_mask': torch.tensor(obj['attention_mask'], dtype=torch.long, device=device),
            'start': torch.tensor(start, dtype=torch.long, device=device).unsqueeze(0),
            'end': torch.tensor(end, dtype=torch.long, device=device).unsqueeze(0),
        }

    def __len__(self):
        return len(self.X)