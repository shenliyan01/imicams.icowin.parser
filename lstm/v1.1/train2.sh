#! /bin/bash
python3 main.py --data_path ../data/crf_1_1000.sent_line.txt --word2idx ../data/word2idx_1_1000.json --label2idx ../data/label2idx_1_1000.json --n_fold 5 --max_epoch 50
