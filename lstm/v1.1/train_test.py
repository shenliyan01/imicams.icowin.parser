import numpy as np
import sys, os
import tensorflow as tf

def _convert2sequence(data, mask, head, body, end, single, o_tag, padding_idx):
    res = []
    tmp = []
    for idx, x in enumerate(data):
        if mask[idx] == padding_idx:
            tmp.append('<|>')
            break
        if x in head:
            tmp.append('<|>')
            tmp.append(idx)
        elif x in single:
            tmp.append('<|>')
            tmp.append(idx)
            tmp.append('<|>')
        elif x in o_tag:
            tmp.append('<|>')
        elif x in body:
            pass
        elif x in end:
            tmp.append(idx)
            tmp.append('<|>')
        else:
            print('word tag wrong')
            print(x)
            sys.exit(0)
    # to tuples
    p, tmp2 = '<|>', []
    for x in tmp:
        if p == '<|>' and x == '<|>':
            pass
        elif p == '<|>' and x != '<|>':
            tmp2.append(x)
        elif p != '<|>' and x == '<|>':
            tmp2.append(x)
            res.append(tuple(tmp2))
        elif p != '<|>' and x != '<|>':
            pass
        else:
            pass
        p = x
    return res

def mk_res_string(x, pred_y, gold_y, mask, idx2word, idx2label):
    tmp = []
    for xrow, pred_yrow, gold_yrow, mrow in zip(x, pred_y, gold_y, mask):
        s = ''
        for xw, pred_yw, gold_yw, mw in zip(xrow, pred_yrow, gold_yrow, mrow):
            if mw == 0:
                tmp.append(s)
                s = ''
                break
            else:
                s += (idx2word[xw] + '<|>' + idx2label[pred_yw] + '<|>' + idx2label[gold_yw] + '\t')
    return tmp

def get_fpr(pred, gold, mask, head=[2], body=[4], end=[3], o_tag=[0], single=[1], padding_idx=0):
    right_total, pred_total, gold_total = 0, 0, 0
    for pred_line, gold_line, mask_line in zip(pred, gold, mask):
        pred_s = _convert2sequence(pred_line, mask_line, head, body, end, single, o_tag, padding_idx)
        gold_s = _convert2sequence(gold_line, mask_line, head, body, end, single, o_tag, padding_idx)
        pred_total += len(pred_s)
        gold_total += len(gold_s)
        right_total += len(set.intersection(set(pred_s), set(gold_s)))
    print('pred_total', pred_total)
    print('gold_total', gold_total)
    print('right_total', right_total)
    def _get_r(a, b):
        return 0 if b == 0 else a / b
    p = _get_r(right_total, pred_total)
    r = _get_r(right_total, gold_total)
    f1 = _get_r(2 * p * r, p + r)
    return f1, p, r, right_total, pred_total, gold_total


def train(sess, model, dataloader, idx2word, idx2label, ith_fold):
    batch_gen = dataloader.gen_batch(ith_fold=ith_fold)
    pred_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    gold_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    input_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    mask_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    for inputs_x, inputs_y, lengths, mask in batch_gen:
        feed_dict = {
            model.inputs_x: inputs_x,
            model.inputs_y: inputs_y,
            model.mask: mask,
            model.lengths: lengths
        }
        loss, matrix_prob, _ = sess.run(
            [model.loss, model.prob_format, model.train_op],
            feed_dict=feed_dict
        )

        matrix_id = np.argmax(matrix_prob, axis=-1)
        pred_matrix = np.concatenate((pred_matrix, matrix_id), axis=0)
        gold_matrix = np.concatenate((gold_matrix, inputs_y), axis=0)
        input_matrix = np.concatenate((input_matrix, inputs_x), axis=0)
        mask_matrix = np.concatenate((mask_matrix, mask), axis=0)
    f1, p, r, _1, _2, gold_total = get_fpr(pred_matrix, gold_matrix, mask_matrix)
    print('train gold_total', gold_total)
    return f1, p, r, loss


def test(sess, model, dataloader, idx2word, idx2label, ith_fold, outpath='./'):
    filename = 'test_{}.txt'.format(ith_fold)
    filepath = os.path.join(outpath, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    fout = open(filepath, 'a')
    batch_gen = dataloader.gen_batch(train_flag=False, ith_fold=ith_fold)
    pred_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    gold_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    input_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    mask_matrix = np.empty([0, model.max_sent_length], dtype=np.int32)
    for inputs_x, inputs_y, lengths, mask in batch_gen:
        feed_dict = {
            model.inputs_x: inputs_x,
            model.inputs_y: inputs_y,
            model.mask: mask,
            model.lengths: lengths
        }
        loss, matrix_prob = sess.run(
            [model.loss, model.prob_format],
            feed_dict=feed_dict
        )
        matrix_id = np.argmax(matrix_prob, axis=-1)
        pred_matrix = np.concatenate((pred_matrix, matrix_id), axis=0)
        gold_matrix = np.concatenate((gold_matrix, inputs_y), axis=0)
        input_matrix = np.concatenate((input_matrix, inputs_x), axis=0)
        mask_matrix = np.concatenate((mask_matrix, mask), axis=0)
    f1, p, r, _1, _2, gold_total = get_fpr(pred_matrix, gold_matrix, mask_matrix)
    s_list = mk_res_string(input_matrix, pred_matrix, gold_matrix, mask_matrix, idx2word, idx2label)
    for x in s_list:
        fout.write(x)
        fout.write('\n')
    fout.close()

    print('test gold_total', gold_total)
    return f1, p, r, loss


if __name__ == '__main__':
    get_fpr([1, 2], [3, 2], [4, 4])
