from io import open
import glob
import os
import sys
import unicodedata
import string
import random
import time
import math

from nets import rnn
import torch
import torch.nn as nn

ALLLETTERS = string.ascii_letters + " .,;'"
NLETTERS = len(ALLLETTERS)



def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALLLETTERS
    )


def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_training_example(all_categories, category_lines):
    category = all_categories[random.randint(0, len(all_categories)-1)]
    word = category_lines[category]
    line = word[random.randint(0, len(word)-1)]
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = torch.zeros(len(line), 1, NLETTERS)
    for li, letter in enumerate(line):
        line_tensor[li][0][ALLLETTERS.find(letter)] = 1
    return category, line, category_tensor, line_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(net, all_categories, category_lines):
    criterion = nn.NLLLoss()
    learning_rate = 0.005
    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    current_loss = 0
    all_losses = []
    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = \
            random_training_example(all_categories, category_lines)
        hidden = net.initHidden()
        net.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = net(line_tensor[i], hidden)
        loss = criterion(output, category_tensor)
        loss.backward()
        # Add parameters' gradients to their values, multiplied by learning rate
        for p in net.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        current_loss += loss
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = 'T' if guess == category else 'F (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


def main():
    category_lines = {}
    all_categories = []
    for filename in glob.glob('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)
    n_hidden = 128
    net = rnn.RNN(NLETTERS, n_hidden, n_categories)
    train(net, all_categories, category_lines)


if __name__ == '__main__':
    main()
