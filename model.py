import torch
from transformers import *

from dataset.TweetSentimentDataset import TweetSentiment

OFFSET_FOR_ENCODING = 4
MAX_LENGTH = 280+5
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=MAX_LENGTH, padding_side='right')

def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            return i, i+len(pattern)

def generate_indices(sentiment, text, selected_text):
    start_index, end_index = subfinder(text, selected_text)
    text_in_selected_text = [0] * len(text)
    for idx in range(start_index, end_index):
        text_in_selected_text[idx] = 1

    token_text = tokenizer.encode_plus(sentiment, text, return_offsets_mapping=True)
    part_of_selected_text = []
    # idx is the index of the word where the character was found to be part
    # of selected_text. offset1 and offset2 are just of the original string encoded
    # for each token. Eg) tokenizer.encode('I am the dog'), offset[2] = (2, 3). Note we do 3:-1
    # because first few are <s>Sentiment</s></s> therefore we start after those useless parts
    SKIP_USELESS_TOKENS = 4
    for idx, (offset1, offset2) in enumerate(token_text['offset_mapping'][SKIP_USELESS_TOKENS:-1]):
        # We do sum here because if any character is a 1, that means the whole word is a 1
        if (offset2 - offset1) > 0 and sum(text_in_selected_text[offset1: offset2]) > 0:
            part_of_selected_text.append(idx+SKIP_USELESS_TOKENS)
    return {
        'start': part_of_selected_text[0],
        'end': part_of_selected_text[-1]+1,
        'encoding': token_text['input_ids']
    }

def train():
    train_dataset = TweetSentiment()
    cls = '<s>'
    sep = '</s>'
    for epoch in range(1):
        for batch_num, batch in enumerate(train_dataset):
            if (batch_num == 10):
                break
            X, Y = batch['X'], batch['Y']
            obj = generate_indices(X['sentiment'], X['text'].strip(), Y.strip())
            start = obj['start']
            end = obj['end']
            encoding = obj['encoding']
            print('X:'+X['text'])
            print('Y:'+Y)
            # No clue why but when decoding it will include the space before when it is in middle of sentence or not with a character before it
            # Eg) I'd have, retrieving d will be fine without space but retrieving have will include space before it
            print('decoded:', tokenizer.decode(encoding[start:end]).strip())
            print('encoding: ', encoding)




if __name__ == '__main__':
    train()