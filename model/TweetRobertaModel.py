import torch
import transformers
from config import hyper_params

MAX_LENGTH = hyper_params['max_length']
BATCH_SIZE = hyper_params['batch']
MAX_CHAR_LENGTH = 280
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TokenLevelModel(torch.nn.Module):
    def __init__(self):
        super(TokenLevelModel, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base', config=model_config)
        self.linear = torch.nn.Linear(768 * 2, 2)
        self.dropout = torch.nn.Dropout(0.1)
        # self.adaptiveStart = nn.AdaptiveLogSoftmaxWithLoss(in_features=MAX_LENGTH, n_classes=MAX_LENGTH,
        #                                                    cutoffs=[5, 10, 59])
        # self.adativeEnd = nn.AdaptiveLogSoftmaxWithLoss(in_features=MAX_LENGTH, n_classes=MAX_LENGTH,
        #                                                 cutoffs=[10, 25, 59])

    def forward(self, text, input_ids, attention_mask, token_type_ids, offset_mapping):
        _, _, X = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = torch.cat((X[0], X[1]), dim=-1)
        X = self.dropout(X)
        X = self.linear(X)
        start, end = X.split(1, dim=-1)
        start_batch = []
        end_batch = []
        for idx1, row in enumerate(offset_mapping):
            char_start_prob = [0] * 280
            char_end_prob = [0] * 280
            flattened_start = start[idx1].squeeze(1)
            flattened_end = end[idx1].squeeze(1)
            for idx2, (offset1, offset2) in enumerate(row):
                if (offset2 - offset1) > 0:
                    for idx3 in range(offset1, offset2):
                        char_start_prob[idx3] = flattened_start[idx2]
                        char_end_prob[idx3] = flattened_end[idx2]

            start_batch.append(char_start_prob)
            end_batch.append(char_end_prob)
        return torch.FloatTensor(start_batch).to(device), torch.FloatTensor(end_batch).to(device)

lstm_hidden_size = 2
lstm_num_layers = 10
class CharLevelModel(torch.nn.Module):
    def __init__(self):
        super(CharLevelModel, self).__init__()
        self.token_level_model = TokenLevelModel()
        self.lstm = torch.nn.LSTM(1, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, text, input_ids, attention_mask, token_type_ids, offset_mapping):
        start_batch, end_batch = self.token_level_model(text, input_ids, attention_mask, token_type_ids, offset_mapping)
        h_0 = torch.zeros(lstm_num_layers * 2, BATCH_SIZE, lstm_hidden_size).to(device)
        c_0 = torch.zeros(lstm_num_layers * 2, BATCH_SIZE, lstm_hidden_size).to(device)
        lstm_input = start_batch + end_batch
        lstm_input = lstm_input.unsqueeze(2)
        lstm_output = self.lstm(lstm_input, (h_0, c_0))
        output = self.linear(lstm_output[0])
        return output


