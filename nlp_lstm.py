import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets.lstm import LSTM


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def main():
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

    model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)

    for epoch in range(300):
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)


if __name__ == '__main__':
    main()
