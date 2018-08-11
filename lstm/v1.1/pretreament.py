# ! /Users/wangxuwen/anaconda3/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 18/4/11 下午4:22
# @Author: bcsong
# @Site  : www.geiri.sgcc.com.cn
# @File  : pretreament.py
# @Software : PyCharm
import os
import argparse
import json

sep = '<|>'

def GetFileNameAndExt(filename):
 (filepath,tempfilename) = os.path.split(filename)
 (shotname,extension) = os.path.splitext(tempfilename)
 return shotname,extension

# get sent_lint.txt for train
def sent_line(input_file):
    print(input_file)
    print('{0}/{1}.sent_line.txt'.format(os.path.split(input_file)[0],GetFileNameAndExt(input_file)[0]))
    fin = open(input_file)

    fout = open('{0}/{1}.sent_line.txt'.format(os.path.split(input_file)[0],GetFileNameAndExt(input_file)[0]), 'w')

    sent = []
    for line in fin:
        line_list = line.strip().split()
        if line_list == ['.', '.', '.', '.']:
            if sent != []:
                fout.write(' '.join(sent))
                fout.write('\n')
                sent = []
        else:
            sent.append(line_list[0] + sep + line_list[-1])
    fin.close()
    fout.close()

# word2idx
def word2idx(sent_line_input,word2idx_path):
    word2idx = {}
    cnt = 2
    f = open(sent_line_input)
    for line in f:
        l = line.strip().split()
        for word in l:
            word = word.split(sep)[0]
            if word not in word2idx:
                word2idx[word] = cnt
                cnt += 1
    if 'UNK' not in word2idx:
        word2idx['UNK'] = 1
        print('UNK')
    if 'PADDING' not in word2idx:
        word2idx['PADDING'] = 0
        print('PADDING')
    f.close()
    import json
    s = json.dumps(word2idx, indent=4)
    fout = open('{0}/{1}.json'.format(os.path.split(word2idx_path)[0],GetFileNameAndExt(word2idx_path)[0]), 'w')
    fout.write(s)
    fout.close()
    print(cnt)

#label2idx
def label2idx(sent_line_file,label2idx_path):
    label2idx = {}
    cnt = 0
    f = open(sent_line_file)
    for line in f:
        l = line.strip().split()
        for label in l:
            label = label.split(sep)[1]
            if label not in label2idx:
                label2idx[label] = cnt
                cnt += 1
    s = json.dumps(label2idx, indent=4)
    f.close()
    fout = open('{0}/{1}.json'.format(os.path.split(label2idx_path)[0],GetFileNameAndExt(label2idx_path)[0]), 'w')
    fout.write(s)
    fout.close()
    print(cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_file',help='origin crf format text')
    parser.add_argument('--sent_line_file', type=str, help='crf sent line text ')
    parser.add_argument('--label2idx',help='label2idx.json')
    parser.add_argument('--word2idx',help='word2idx.json')
    args = parser.parse_args()


    if args.input_file != None and args.sent_line_file == None and args.label2idx == None and args.word2idx == None:
        print("sent file process start\n")
        sent_line(args.input_file)
        print("sent file process end\n")
    elif args.input_file == None and args.sent_line_file != None and args.label2idx == None and args.word2idx == None:
        print("not only sent_line_file is setted\n")
        print(args.print_help())
    elif args.input_file == None and args.sent_line_file != None and args.label2idx != None and args.word2idx == None:
        print("label2idx process start\n")
        label2idx(args.sent_line_file,args.label2idx)
        print("label2idx process end\n")
    elif args.input_file == None and args.sent_line_file != None and args.label2idx == None and args.word2idx != None:
        print("word2idx process start\n")
        word2idx(args.sent_line_file,args.word2idx)
        print("word2idx process end\n")
    elif args.input_file == None and args.sent_line_file != None and args.label2idx != None and args.word2idx != None:
        print("word2idx process start \n")
        word2idx(args.sent_line_file,args.word2idx)
        print("word2idx process end\n label2idx process start\n")
        label2idx(args.sent_line_file,args.label2idx)
        print("label2idx process end \n")
    else:
        print(args.print_help())




