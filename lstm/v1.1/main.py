# -*- coding utf-8 -*-

import numpy as np
import tensorflow as tf
from model import *
from dataloader import *
from train_test import *
import json
import os
import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('--vocab_size', type=int, default=500,
        help='the size of vocab')
parser.add_argument('--lstm_unit', type=int, default=40,
        help='the size of the hidden biLSTM')
parser.add_argument('--fc_unit', type=int, default=40,
        help='the size of the dense layer')
parser.add_argument('--max_sent_length', type=int, default=250,
        help='the max length of a sentence')
parser.add_argument('--lr', type=float, default=0.01,
        help='learning rate')
parser.add_argument('--label_num', type=int, default=5,
        help='the num of labels')
parser.add_argument('--show_n_right_res', type=int, default=0,
        help='show n right result sentences, 0 for none')
parser.add_argument('--show_n_wrong_res', type=int, default=0,
        help='show n wrong result sentences, 0 for none')
parser.add_argument('--data_path', type=str, default='../data/crf_1001_2000.sent_line.txt',
        help='data path')
parser.add_argument('--padding', type=str, default='PADDING',
        help='the symbol for padding')
parser.add_argument('--n_fold', type=int, default=1,
        help='n fold cross validation, if n_fold is 1, we save model.')
parser.add_argument('--batch_size', type=int, default=20,
        help='batch size')
parser.add_argument('--prop', type=float, default=0.8,
        help='proportion of the train set, only used when n_fold is 1 or 0')
parser.add_argument('--max_epoch', type=int, default=30,
        help='max epoch')
parser.add_argument('--out_path', type=str, default='./',
        help='the out directory of the results')
parser.add_argument('--shuffle_train', type=bool, default=False,
        help='shuffle the train set')
parser.add_argument('--shuffle_all', type=bool, default=False,
        help='shuffle all the data set')
parser.add_argument('--sep', type=str, default='<|>',
        help='separator between the words and labels.')
parser.add_argument('--tag_only', type=bool, default=False,
        help='use saved model to label the input file.')
parser.add_argument('--model_path', type=str, default='./model/model_trained',
        help='path of model.')

parser.add_argument('--word2idx',type=str,default='../data/word2idx.json',
                    help='word2idx file from pretreament')
parser.add_argument('--label2idx',type=str,default='../data/label2idx.json',
                    help='label2idx file from pretreament')

args = parser.parse_args()

word2idx = json.load(open(args.word2idx))
label2idx = json.load(open(args.label2idx))
idx2word = dict([(v, k) for (k, v) in word2idx.items()])
idx2label = dict([(v, k) for (k, v) in label2idx.items()])

dataloader = Dataloader(
        data_path=args.data_path,
        padding=args.padding,
        batch_size=args.batch_size,
        max_sent_length=args.max_sent_length,
        prop=args.prop,
        n_fold=args.n_fold,
        shuffle_all=args.shuffle_all,
        shuffle_train=args.shuffle_train,
        word2idx=args.word2idx,
        label2idx=args.label2idx,
        sep=args.sep
    )

# emb size
if len(word2idx) > args.vocab_size:
    vocab_size = len(word2idx)
else:
    vocab_size = args.vocab_size
if len(label2idx) > args.label_num:
    label_num = len(label2idx)
else:
    label_num = args.label_num

print('vocab_size:', vocab_size)
print('label_num:', label_num)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
fold_res = {}
f1_all, p_all, r_all = 0, 0, 0
if args.tag_only:
    tf.reset_default_graph()
    dataloader.set_to_test()
    model = Model(
            max_sent_length=dataloader.max_sent_length,
            vocab_size=vocab_size,
            lr=args.lr,
            label_num=label_num,
            lstm_unit=args.lstm_unit,
            fc_unit=args.fc_unit 
        )
    model.build_model()
    sess = tf.Session(config=config)
    model.saver.restore(sess, args.model_path)
    f1, p, r, loss = test(sess, model, dataloader, idx2word, idx2label, 1, outpath=args.out_path)
    print('test: f1: {:.4f}, p: {:.4f}, r: {:.4f}, loss: {:.4f}'.format(f1, p, r, loss))
else:
    # 只有不做交叉验证的时候，才存储模型
    if args.n_fold == 1:
        ith_fold = 1
        max_f1 = 0
        tf.reset_default_graph()
        model = Model(
            max_sent_length=dataloader.max_sent_length,
            vocab_size=vocab_size,
            lr=args.lr,
            label_num=label_num,
            lstm_unit=args.lstm_unit,
            fc_unit=args.fc_unit 
        )
        model.build_model()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(args.max_epoch):
                print('############# ith fold: {:4d} epoch :{:4d} ###############'.format(ith_fold, e))
                f1, p, r, loss = train(sess, model, dataloader, idx2word, idx2label, ith_fold)
                print('train: f1: {:.4f}, p: {:.4f}, r: {:.4f}, loss: {:.4f}'.format(f1, p, r, loss))
                f1, p, r, loss = test(sess, model, dataloader, idx2word, idx2label, ith_fold, outpath=args.out_path)
                print('test: f1: {:.4f}, p: {:.4f}, r: {:.4f}, loss: {:.4f}'.format(f1, p, r, loss))
                if f1 >= max_f1:
                    model.saver.save(sess, args.model_path)
                    print('model saved')
                    max_f1 = f1
            fold_res[ith_fold] = {}
            fold_res[ith_fold]['f1'], fold_res[ith_fold]['p'], fold_res[ith_fold]['r'] = f1, p, r
            f1_all += f1
            p_all += p
            r_all += r
        fold_res['avg'] = {}
        fold_res['avg']['f1'], fold_res['avg']['p'], fold_res['avg']['r'] = f1_all / args.n_fold, p_all / args.n_fold, r_all / args.n_fold
        print(fold_res)
        fold_res_path = os.path.join(args.out_path, 'fold_res.json')
        if os.path.exists(fold_res_path):
            os.remove(fold_res_path)
        s = json.dumps(fold_res, indent=4)
        f = open(fold_res_path, 'w')
        f.write(s)
        f.close()
    else:
        for ith_fold in range(args.n_fold):
            tf.reset_default_graph()
            model = Model(
                    max_sent_length=dataloader.max_sent_length,
                    vocab_size=vocab_size,
                    lr=args.lr,
                    label_num=label_num,
                    lstm_unit=args.lstm_unit,
                    fc_unit=args.fc_unit 
                )
            model.build_model()
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                for e in range(args.max_epoch):
                    print('############# ith fold: {:4d} epoch :{:4d} ###############'.format(ith_fold, e))
                    f1, p, r, loss = train(sess, model, dataloader, idx2word, idx2label, ith_fold)
                    print('train: f1: {:.4f}, p: {:.4f}, r: {:.4f}, loss: {:.4f}'.format(f1, p, r, loss))
                    f1, p, r, loss = test(sess, model, dataloader, idx2word, idx2label, ith_fold, outpath=args.out_path)
                    print('test: f1: {:.4f}, p: {:.4f}, r: {:.4f}, loss: {:.4f}'.format(f1, p, r, loss))
                fold_res[ith_fold] = {}
                fold_res[ith_fold]['f1'], fold_res[ith_fold]['p'], fold_res[ith_fold]['r'] = f1, p, r
                f1_all += f1
                p_all += p
                r_all += r
            fold_res['avg'] = {}
            fold_res['avg']['f1'], fold_res['avg']['p'], fold_res['avg']['r'] = f1_all / args.n_fold, p_all / args.n_fold, r_all / args.n_fold
            print(fold_res)
            fold_res_path = os.path.join(args.out_path, 'fold_res.json')
            if os.path.exists(fold_res_path):
                os.remove(fold_res_path)
            s = json.dumps(fold_res, indent=4)
            f = open(fold_res_path, 'w')
            f.write(s)
            f.close()

