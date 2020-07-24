import torch
import transformers
from config import hyper_params
MAX_LENGTH = hyper_params['max_length']
BATCH_SIZE = hyper_params['batch']

class TweetRobertaModel(torch.nn.Module, ):
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

    def forward(self, text, input_ids, attention_mask, token_type_ids, offset_mapping):
        _, _, X = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = torch.cat((X[0], X[1]), dim=-1)
        X = self.dropout(X)
        X = self.linear(X)
        start, end = X.split(1, dim=-1)
        char_level_start_probability = [0] * len(text[0])
        char_level_end_probability = [0] * len(text[0])
        flattened_start = start[0].squeeze(1)
        flattened_end = end[0].squeeze(1)
        SKIP_USELESS_TOKENS = 4
        for idx, (offset1, offset2) in enumerate(offset_mapping[SKIP_USELESS_TOKENS:-1]):
            if offset2 - offset1 > 0:
                for idx2 in range(offset1, offset2):
                    char_level_start_probability[idx2] = flattened_start[SKIP_USELESS_TOKENS+idx].item()
                    char_level_end_probability[idx2] = flattened_end[SKIP_USELESS_TOKENS+idx].item()
        print('char_level_start:', char_level_start_probability)
        print('char_level_end:', char_level_end_probability)
        print(text)
        return start, end