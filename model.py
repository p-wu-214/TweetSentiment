import torch
from transformers import *

from dataset.TweetSentimentDataset import TweetSentiment

def train():
    train_dataset = TweetSentiment()
    for epoch in range(1):
        for batch_num, batch in enumerate(train_dataset):
            if (batch_num == 10):
                break
            tokenizer = batch['tokenizer']
            start, end = batch['start'], batch['end']
            print('test: ', batch['selected_sentence'])
            print('tokenized: ', tokenizer.decode(batch['original_input_ids'][start:end]))



def test_start_end_index():
    train_dataset = TweetSentiment()
    for epoch in range(1):
        for batch_num, batch in enumerate(train_dataset):
            if (batch_num == 10):
                break
            tokenizer = batch['tokenizer']
            start, end = batch['start'], batch['end']
            print('test: ', batch['selected_sentence'])
            print('tokenized: ', tokenizer.decode(batch['original_input_ids'][start:end]))

if __name__ == '__main__':
    train()