import torch
import transformers
from config import hyper_params
MAX_LENGTH = hyper_params['max_length']
BATCH_SIZE = hyper_params['batch']

class TweetRobertaModel(torch.nn.Module):
    def __init__(self):
        super(TweetRobertaModel, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base', config=model_config)
        self.linear = torch.nn.Linear(768 * 2, 2)
        self.dropout = torch.nn.Dropout(0.1)
        # self.adaptiveStart = nn.AdaptiveLogSoftmaxWithLoss(in_features=MAX_LENGTH, n_classes=MAX_LENGTH,
        #                                                    cutoffs=[5, 10, 59])
        # self.adativeEnd = nn.AdaptiveLogSoftmaxWithLoss(in_features=MAX_LENGTH, n_classes=MAX_LENGTH,
        #                                                 cutoffs=[10, 25, 59])

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, _, X = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = torch.cat((X[0], X[1]), dim=-1)
        X = self.dropout(X)
        X = self.linear(X)
        start, end = X.split(1, dim=-1)
        # start = self.softmax(start)
        # end = self.softmax(end)
        return start, end