from torch import nn
from transformers import RobertaModel
import torch

class TweetRobertaModel(nn.Module):
    def __init__(self):
        super(TweetRobertaModel, self).__init__(self)
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(768*2, 2)

    def forward(self, input_ids, mask, token_type_ids):
        _, _, X = self.roberta(input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        X = self.drop(X)
        X = self.linear(X)
        return X