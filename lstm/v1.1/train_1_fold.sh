#! /bin/bash
python3 main.py --data_path ../data/crf_1_1000.sent_line.txt --word2idx ../data/word2idx.json --label2idx ../data/label2idx.json --n_fold 1 --max_epoch 50 --tag_only True
