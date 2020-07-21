from dataset.TweetSentimentDataset import TweetSentiment
from model.TweetRobertaModel import TweetRobertaModel
import torch
from transformers import *
import pandas as pd
from config import hyper_params
BATCH_SIZE = hyper_params['batch']
MAX_LENGTH = hyper_params['max_length']

def loss_fn(start_prob, end_prob, actual_start, actual_end):
    loss = torch.nn.CrossEntropyLoss()
    loss_start = loss(input=start_prob, target=actual_start)
    loss_end = loss(input=end_prob, target=actual_end)
    return (loss_start + loss_end)/2

def train():
    train_dataset = TweetSentiment()
    model = TweetRobertaModel()
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['lr'], betas=hyper_params['betas'], eps=1e-08)
    for epoch in range(1):
        for batch_num, batch in enumerate(dataloader):
            if batch_num > 10:
                return
            start, end = batch['start'], batch['end']
            pred_start, pred_end = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            loss = loss_fn(pred_start, pred_end, start, end)
            print('epoch:', epoch, 'loss is:', loss)
            loss.backward()
            optimizer.step()

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
    # loss = torch.nn.CrossEntropyLoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # print('input', input)
    # print('target', target)
    # output = loss(input, target)
    # output.backward()