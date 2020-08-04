from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import sys
import unicodedata
import string
import random

import torch
import torch.nn as nn

ALLLETTERS = string.ascii_letters + " .,;'"
NLETTERS = len(ALLLETTERS)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALLLETTERS
    )


def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def letter2index(letter):
    return ALLLETTERS.find(letter)


def letter2tensor(letter):
    tensor = torch.zeros(1, NLETTERS)
    tensor[0][letter2index(letter)] = 1
    return tensor


def line2tensor(line):
    tensor = torch.zeros(len(line), 1, NLETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(all_categories, category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor


def main():
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)

    n_hidden = 128
    rnn = RNN(NLETTERS, n_hidden, n_categories)
    tinput = letter2tensor('A')
    hidden =torch.zeros(1, n_hidden)
    output, next_hidden = rnn(tinput, hidden)


if __name__ == '__main__':
    main()
