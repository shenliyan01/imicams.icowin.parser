import tensorflow as tf
import os

class Model():
    def __init__(
            self,
            vocab_size=40,
            word_dim=64,
            lstm_unit=40,
            fc_unit=40,
            max_sent_length=280,
            label_num=7,
            lr=0.001
        ):
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_unit = lstm_unit
        self.fc_unit = fc_unit
        self.max_sent_length = max_sent_length
        self.label_num = label_num
        self.lr = lr
        pass

    def _create_place_holder(self):
        self.inputs_x = tf.placeholder(tf.int32, [None, self.max_sent_length], name='inputs_x')
        self.inputs_y = tf.placeholder(tf.int32, [None, self.max_sent_length], name='inputs_y')
        self.mask = tf.placeholder(tf.int32, [None, self.max_sent_length], name='mask')
        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')
    
    def _create_emb_layer(self):
        with tf.variable_scope('emb_layer'):
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.word_dim], -1.0, 1.0),
                name='embeddings'
            )
            print(self.embeddings.shape)
            self.emb_layer = tf.nn.embedding_lookup(self.embeddings, self.inputs_x, name='emb_layer')

    def _create_lstm(self):
        cell_fn = tf.contrib.rnn.BasicLSTMCell
        with tf.variable_scope('lstm'):
            with tf.variable_scope('fw'):
                cell_fw = cell_fn(self.lstm_unit)
            with tf.variable_scope('bw'):
                cell_bw = cell_fn(self.lstm_unit)
            bidirectional_outputs_raw, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=self.lengths,
                inputs=self.emb_layer
            )
            self.bidirectional_outputs_format = tf.concat(
                [bidirectional_outputs_raw[0], bidirectional_outputs_raw[1]],
                axis=-1,
                name='bidirectional_outputs_format'
            )

    def _create_fc(self):
        with tf.variable_scope('fc'):
            dense_w = tf.get_variable(
                'w',
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[2*self.lstm_unit, self.fc_unit]
            )
            dense_b = tf.get_variable(
                'b',
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self.fc_unit]
            )
            bidirectional_flat = tf.reshape(
                self.bidirectional_outputs_format,
                [-1, 2*self.lstm_unit],
                name='bidirectional_flat'
            )
            self.fc_flat = tf.tanh(
                tf.matmul(bidirectional_flat, dense_w) + dense_b,
                name='fc_flat'
            )
            self.fc_format = tf.reshape(
                self.fc_flat,
                [-1, self.max_sent_length, self.fc_unit],
                name='fc_format'
            )

    def _create_logits(self):
        with tf.variable_scope('logits'):
            dense_w = tf.get_variable(
                'w',
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self.fc_unit, self.label_num]
            )
            dense_b = tf.get_variable(
                'b',
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self.label_num]
            )
            self.logits_flat = tf.matmul(self.fc_flat, dense_w) + dense_b
            self.logits_format = tf.reshape(
                self.logits_flat,
                [-1, self.max_sent_length, self.label_num]
            )
            self.prob_format = tf.nn.softmax(
                logits=self.logits_format,
                dim=-1,
                name='softmax'
            )

    def _create_loss(self):
        with tf.variable_scope('loss'):
            loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_format,
                labels=self.inputs_y,
                name='loss'
            )
            self.loss_format = loss_raw * tf.cast(self.mask, dtype=tf.float32, name='loss_format')
            self.loss = tf.reduce_sum(self.loss_format, name='loss_all') / tf.cast(tf.reduce_sum(self.mask), dtype=tf.float32)

    def _create_optimizer(self):
        with tf.variable_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def build_model(self):
        self._create_place_holder()
        self._create_emb_layer()
        self._create_lstm()
        self._create_fc()
        self._create_logits()
        self._create_loss()
        self._create_optimizer()
        self.saver = tf.train.Saver(max_to_keep=1)

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        self.saver.restore(sess, path)


if __name__ == '__main__':
    model = Model()
    model.build_model()
