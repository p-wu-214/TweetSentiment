from torch import nn
from transformers import RobertaModel
from config import hyper_params
MAX_LENGTH = hyper_params['max_length']

class TweetRobertaModel(nn.Module):
    def __init__(self):
        super(TweetRobertaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.linear = nn.Linear(768, MAX_LENGTH*2)
        # self.adaptiveStart = nn.AdaptiveLogSoftmaxWithLoss(in_features=MAX_LENGTH, n_classes=MAX_LENGTH,
        #                                                    cutoffs=[5, 10, 59])
        # self.adativeEnd = nn.AdaptiveLogSoftmaxWithLoss(in_features=MAX_LENGTH, n_classes=MAX_LENGTH,
        #                                                 cutoffs=[10, 25, 59])

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, X = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = self.linear(X)
        X = X.view(2, 1, MAX_LENGTH)
        print('model x.shape: ', X.shape)
        for x in range(5):
            X[0][-1][x] = 0
            X[1][-1][x] = 0
        # # what the hell is target!!!
        # X[0] = self.adaptiveStart(X[0])
        # X[1] = self.adaptiveEnd(X[1])
        return X