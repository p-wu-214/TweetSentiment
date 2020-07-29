import torch
import transformers
from config import hyper_params
from torch import nn

MAX_LENGTH = hyper_params['max_length']
BATCH_SIZE = hyper_params['batch']
MAX_CHAR_TWEET = 141
ROBERTA_OUTPUT_SIZE = 768
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TokenLevelModel(torch.nn.Module):
    def __init__(self):
        super(TokenLevelModel, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base', config=model_config)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(ROBERTA_OUTPUT_SIZE * 2, 768 * 2)

    def forward(self, text, input_ids, attention_mask, token_type_ids, offset_mapping):
        _, _, X = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = torch.cat((X[0], X[1]), dim=-1)
        # Why don't we try dumping the output from roberta directly into the lstm layer!!!
        X = self.dropout(X)
        X = self.linear(X)
        start, end = X.split(ROBERTA_OUTPUT_SIZE, dim=-1)
        start_result = torch.zeros(input_ids.shape[0], MAX_CHAR_TWEET, ROBERTA_OUTPUT_SIZE).to(device)
        end_result = torch.zeros(input_ids.shape[0], MAX_CHAR_TWEET, ROBERTA_OUTPUT_SIZE).to(device)
        for idx1, row in enumerate(offset_mapping):
            for idx2, (offset1, offset2) in enumerate(row):
                if (offset2 - offset1) > 0:
                    # this part really needs fixing
                    for idx3 in range(offset1, offset2):
                        start_result[idx1][idx3] = start[idx1][idx2]
                        end_result[idx1][idx3] = end[idx1][idx2]

        return start_result, end_result

lstm_hidden_size = 768
lstm_num_layers = 4
class CharLevelModel(torch.nn.Module):
    def __init__(self):
        super(CharLevelModel, self).__init__()
        self.token_level_model = TokenLevelModel()
        # self.lstm = torch.nn.LSTM(ROBERTA_OUTPUT_SIZE * 2, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(ROBERTA_OUTPUT_SIZE * 2, 282)
        self.linear2 = nn.Linear(282, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, input_ids, attention_mask, token_type_ids, offset_mapping, hidden):
        start, end = self.token_level_model(text, input_ids, attention_mask, token_type_ids, offset_mapping)
        X = torch.cat((start, end), dim=-1)
        # X, hidden = self.lstm(X, hidden)

        X = self.dropout(X)

        X = self.linear1(X)
        X = self.linear2(X)

        X1, X2 = X.split(1, dim=-1)

        return X1, X2, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(lstm_num_layers*2, batch_size, lstm_hidden_size).zero_().to(device),
                  weight.new(lstm_num_layers*2, batch_size, lstm_hidden_size).zero_().to(device))
        return hidden


