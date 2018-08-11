import numpy as np
import json
import sys

class Dataloader():
    def __init__(
            self,
            data_path='../data/crf_1001_2000.sent_line.txt',
            n_fold=2,
            word2idx='../data/word2idx.json',
            label2idx='../data/label2idx.json',
            batch_size=10,
            padding='PADDING',
            max_sent_length=250,
            prop=0.8,
            shuffle_train=False,
            shuffle_all=False,
            sep='<|>'
    ):
        self.data_path = data_path
        self.n_fold = n_fold
        self.prop = prop
        self.batch_size = batch_size
        self.word2idx = json.load(open(word2idx))
        self.label2idx = json.load(open(label2idx))
        self.max_sent_length = max_sent_length
        self.padding = self.word2idx[padding]
        self.shuffle_train = shuffle_train
        self.shuffle_all = shuffle_all

        self.idx2word = dict([(v, k) for (k, v) in self.word2idx.items()])
        self.idx2label = dict([(v, k) for (k, v) in self.label2idx.items()])
        self.sep = sep

        self._initialize()
        self.o_tag = 100

        self.for_test_flat = False

    def _check_n_fold(self):
        if isinstance(self.n_fold, int): return True
        else: False

    def _initialize(self):
        f = open(self.data_path)
        x, y = [], []
        max_length = 0
        for line in f:
            xs = []
            ys = []
            word_label_l = line.strip().split()
            for idx, word_label in enumerate(word_label_l):
                word, label = word_label.split(self.sep)[0], word_label.split(self.sep)[1]
                if word not in self.word2idx:
                    print(word_label_l)
                xs.append(self.word2idx[word])
                try: 
                    if label == '':
                        print(line)
                        print('word:', word)
                        print('pre word:', word_label_l[idx - 1])
                        input()
                    ys.append(self.label2idx[label])
                except:
                    print('ys label wrong')
                    sys.exit(0)
            if len(xs) > max_length:
                max_length = len(x)
            x.append(xs)
            y.append(ys)
        if max_length < self.max_sent_length:
            max_length = self.max_sent_length
        else:
            self.max_sent_length = max_length
        for id, _ in enumerate(x):
            x[id] = x[id] + [self.padding] * (max_length - len(x[id]))
            y[id] = y[id] + [0] * (max_length - len(y[id]))
        self.x = np.array(x, dtype=np.int32)
        self.y = np.array(y, dtype=np.int32)
        if self.shuffle_all is True:
            indices = np.random.permutation(self.x.shape[0])
            self.x, self.y = self.x[indices], self.y[indices]
        # format to n fold
        if self._check_n_fold is False:
            raise TypeError
        if self.n_fold == 0:
            raise ValueError
        self.steps = [0] * self.n_fold
        for id in range(self.x.shape[0] % self.n_fold):
            self.steps[id] += 1
        for id, _ in enumerate(self.steps):
            self.steps[id] += int((self.x.shape[0] - self.x.shape[0] % self.n_fold) / self.n_fold)


    def gen_batch(self, train_flag=True, ith_fold=0):
        if self.for_test_flat:
            x, y = self.x, self.y
            row = x.shape[0]
            lengths = (x != self.padding).astype(np.int32).sum(axis=1)
            mask = (x != self.padding).astype(np.int32)
            i = 0
            while i + self.batch_size < row:
                yield (
                    x[i:i+self.batch_size],
                    y[i:i+self.batch_size],
                    lengths[i:i+self.batch_size],
                    mask[i:i+self.batch_size]
                )
                i = i + self.batch_size
            print('i: {}, row: {}'.format(i, row))
            if i < row:
                yield (
                    x[i:i+self.batch_size],
                    y[i:i+self.batch_size],
                    lengths[i:i+self.batch_size],
                    mask[i:i+self.batch_size]
                )
        if self.n_fold == 1 or self.n_fold == 0:
            row = len(self.x)
            row = int(row * self.prop)
            if train_flag is True:
                x, y = self.x[:row], self.y[:row]
            else:
                x, y = self.x[row:], self.y[row:]
        else:
            train_x, train_y, test_x, test_y = \
                    np.empty((0, self.max_sent_length), dtype=np.int32), \
                    np.empty((0, self.max_sent_length), dtype=np.int32), \
                    np.empty((0, self.max_sent_length), dtype=np.int32), \
                    np.empty((0, self.max_sent_length), dtype=np.int32)
            start = 0
            for ith, step in enumerate(self.steps):
                if ith != ith_fold:
                    train_x = np.concatenate((train_x, self.x[start: start+step]), axis=0)
                    train_y = np.concatenate((train_y, self.y[start: start+step]), axis=0)
                else:
                    test_x = np.concatenate((test_x, self.x[start: start+step]), axis=0)
                    test_y = np.concatenate((test_y, self.y[start: start+step]), axis=0)
                start += step
            if train_flag is True:
                x, y = train_x, train_y
            else:
                x, y = test_x, test_y

        row = x.shape[0]
        lengths = (x != self.padding).astype(np.int32).sum(axis=1)
        mask = (x != self.padding).astype(np.int32)
        if self.shuffle_train:
            indices = np.random.permutation(row)
            x, y, lengths, mask = x[indices], y[indices], lengths[indices], mask[indices]
        i = 0
        while i + self.batch_size < row:
            yield (
                x[i:i+self.batch_size],
                y[i:i+self.batch_size],
                lengths[i:i+self.batch_size],
                mask[i:i+self.batch_size]
            )
            i = i + self.batch_size
        print('i: {}, row: {}'.format(i, row))
        if i < row:
            yield (
                x[i:i+self.batch_size],
                y[i:i+self.batch_size],
                lengths[i:i+self.batch_size],
                mask[i:i+self.batch_size]
            )
    
    def set_to_test(self):
        self.for_test_flat = True


if __name__ == '__main__':
    dataloader = Dataloader()
    dataloader.gen_batch()
