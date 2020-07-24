from dataset.TweetSentimentDataset import TweetSentiment
from model.TweetRobertaModel import CharLevelModel
import torch
from transformers import *
import numpy as np
from config import hyper_params
BATCH_SIZE = hyper_params['batch']
MAX_LENGTH = hyper_params['max_length']

def loss_fn(start_prob, end_prob, actual_start, actual_end):
    loss = torch.nn.CrossEntropyLoss()
    loss_start = loss(input=start_prob, target=actual_start)
    loss_end = loss(input=end_prob, target=actual_end)
    return (loss_start + loss_end)/2

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = TweetSentiment(mode='train')
    # test_dataset = TweetSentiment(mode='test')
    model = CharLevelModel()
    model.to(device)
    training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
    #                         shuffle=True, num_workers=0)
    optimizer = AdamW(model.parameters(), lr=hyper_params['lr'])
    for epoch in range(1):
        model.train()
        avg_loss = []
        start_accuracy = 0
        end_accuracy = 0
        for batch_num, batch in enumerate(training_dataloader):
            if batch_num > 0:
                return
            start, end = batch['start'], batch['end']
            pred_start, pred_end = model(batch['original_tweet'], batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['offset_mapping'])
            # loss = loss_fn(pred_start, pred_end, start, end)
            # loss.backward()
            # optimizer.step()
            # avg_loss.append(loss.item())
            # print('epoch:', epoch, 'loss:', loss)
        print('epoch:', epoch, 'average loss:', np.mean(avg_loss))

    # with torch.no_grad():
    #     for epoch in range(1):
    #         model.eval()
    #         avg_loss = []
    #         start_accuracy = 0
    #         end_accuracy = 0
    #         for batch_num, batch in enumerate(test_dataloader):
    #             start, end = batch['start'], batch['end']
    #             pred_start, pred_end = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
    #             loss = loss_fn(pred_start, pred_end, start, end)
    #             avg_loss.append(loss.item())
    #             size = pred_start.size(0)
    #             start_accuracy += (pred_start == start).sum().item()
    #             end_accuracy += (pred_end == end).sum().item()
    #         print('epoch:', epoch, 'average loss:', np.mean(avg_loss))
    #         print('epoch:', epoch, 'start accuracy:', start_accuracy / size)
    #         print('epoch:', epoch, 'end accuracy:', end_accuracy / size)


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