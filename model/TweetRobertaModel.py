import torch
import transformers
from config import hyper_params
MAX_LENGTH = hyper_params['max_length']
BATCH_SIZE = hyper_params['batch']

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
        char_start_prob = [0] * len(text[0])
        char_end_prob = [0] * len(text[0])
        flattened_start = start[0].squeeze(1)
        flattened_end = end[0].squeeze(1)
        SKIP_USELESS_TOKENS = 4
        for idx, (offset1, offset2) in enumerate(offset_mapping[SKIP_USELESS_TOKENS:-1]):
            if (offset2 - offset1) > 0:
                for idx2 in range(offset1, offset2):
                    char_start_prob[idx2] = flattened_start[SKIP_USELESS_TOKENS+idx]
                    char_end_prob[idx2] = flattened_end[SKIP_USELESS_TOKENS+idx]
        char_start_prob = torch.FloatTensor(char_start_prob)
        char_end_prob = torch.FloatTensor(char_end_prob)

        return char_end_prob, char_end_prob

class CharLevelModel(torch.nn.Module):
    def __init__(self):
        super(CharLevelModel, self).__init__()
        lstm_hidden_size = 6
        self.token_level_model = TokenLevelModel()
        self.lstm = torch.nn.LSTM(280, lstm_hidden_size, num_layers=2, bidirectional=True)
        self.linear = torch.nn.Linear(lstm_hidden_size * 2, 2)

    def forward(self, text, input_ids, attention_mask, token_type_ids, offset_mapping):
        char_start_prob, char_end_prob = self.token_level_model(text, input_ids, attention_mask, token_type_ids, offset_mapping)
        lstm_input = char_start_prob + char_end_prob
        print(lstm_input.shape)
        lstm_output = self.lstm(lstm_input)
        print(lstm_output.shape)
        return lstm_output


