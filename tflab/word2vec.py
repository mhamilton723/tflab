# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from .utils import save_embs, get_or_create_path, optimize_
from .ops import skipgram_word2vec
from .mixins import Model
from .embedding_utils import AnalogyEvaluator


class Word2Vec(Model):
    """Word2Vec model (Skipgram)."""

    def __init__(self,
                 session,
                 # Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.
                 train_data="../data/text8",
                 # "The embedding dimension size.")
                 embedding_size=256,
                 # The minimum number of word occurrences for it to be included in the vocabulary.
                 min_count=7,

                 name="word2vec"):

        self.embedding_size = embedding_size
        self.train_data = train_data
        self.min_count = min_count
        self.name = name

        self.add_params({"data": self.train_data.split('/')[-1].split('.')[0],
                         "dim": self.embedding_size,
                         "mc": self.min_count})

        self.word2id = {}
        self.id2word = None
        self._init_embs(session)

        self.evaluator = None
        print("initialized")

    def _init_embs(self, session):
        with tf.variable_scope(self.name):
            """initialize the variables needed for the embeddings and vocab variables."""
            # The training data. A text file.
            (raw_words_, counts_, words_per_epoch_, epoch_, total_words_, words_,
             contexts_) = skipgram_word2vec(filename=self.train_data,
                                            min_count=self.min_count,
                                            batch_size=1)
            (self.vocab_words, self.vocab_counts, self.words_per_epoch) = \
                session.run([raw_words_, counts_, words_per_epoch_])
            self.vocab_size = len(self.vocab_words)
            print("Data file: ", self.train_data)
            print("Vocab size: ", self.vocab_size - 1, " + UNK")
            print("Words per epoch: ", self.words_per_epoch)
            self.id2word = self.vocab_words
            for i, w in enumerate(self.id2word):
                self.word2id[w] = i

            size = self.vocab_size

            # Declare all variables we need.
            # Embedding: [vocab_size, emb_dim]
            init_width = 0.5 / self.embedding_size
            self.word_embs_ = tf.Variable(
                tf.random_uniform([size, self.embedding_size], -init_width, init_width), name="embs")

            # Softmax weight: [vocab_size, emb_dim]. Transposed.
            self.context_embs_ = tf.Variable(tf.zeros([size, self.embedding_size]), name="sm_w_t")

            # Softmax bias: [emb_dim].
            self.context_biases_ = tf.Variable(tf.zeros([size]), name="sm_b")

            tf.global_variables_initializer().run()

    '''
    def _clip_at_max(self, var, max, default=0.0):
        condition = tf.less(var, max)
        # var = tf.Print(var, [tf.shape(var)])
        e = tf.ones(tf.shape(var), var.dtype) * default
        out = tf.select(condition, var, e)
        # out = tf.Print(out, [tf.reduce_max(out)])
        return out

    def _skipgram_with_max(self, filename, batch_size, window_size, min_count, subsample, max_value):
        (raw_words_, counts_, words_per_epoch_, epoch_, total_words_, words_,
         contexts_) = w2v.skipgram(filename, batch_size, window_size, min_count, subsample)
        maxed_words_ = self._clip_at_max(words_, max_value, 0)
        maxed_contexts_ = self._clip_at_max(contexts_, max_value, 0)
        maxed_total_words_ = tf.minimum(tf.size(raw_words_), max_value)
        maxed_raw_words_ = \
            tf.slice(raw_words_,
                     tf.constant([0], dtype=tf.int64),
                     tf.cast(tf.reshape(maxed_total_words_, (1,)), dtype=tf.int64))
        return (maxed_raw_words_, counts_, words_per_epoch_, epoch_, maxed_total_words_, maxed_words_,
                maxed_contexts_)
    '''

    def save_embeddings(self, save_path=None, name=None):
        if save_path is None:
            save_path = self.save_path
        if name is None:
            name = self.name

        filename = get_or_create_path(save_path, "embeddings", name + "_embs.h5")
        save_embs(filename, self.id2word, self.word_embs_.eval(), self.context_embs_.eval(),
                  self.context_biases_.eval())

    def save_vocab(self, save_path=None, name=None):
        if save_path is None:
            save_path = self.save_path
        if name is None:
            name = self.name

        """Save the vocabulary to a file so the model can be reloaded."""
        with open(get_or_create_path(save_path, "checkpoints", name + "_vocab.txt"), "w") as f:
            for i in xrange(self.vocab_size):
                vocab_word = tf.compat.as_text(self.vocab_words[i]).encode("utf-8")
                f.write("%s %d\n" % (vocab_word,
                                     self.vocab_counts[i]))

    def skipgram_(self, batch_size, window_size, subsample):
        (raw_words_, counts_, words_per_epoch_, epoch_, num_words_seen_, words_, contexts_) = \
            skipgram_word2vec(filename=self.train_data,
                              batch_size=batch_size,
                              window_size=window_size,
                              min_count=self.min_count,
                              subsample=subsample)
        return epoch_, num_words_seen_, words_, contexts_

    def get_embeddings_(self, word_indicies_):
        return tf.nn.embedding_lookup(self.word_embs_, word_indicies_)

    def nce_loss_(self, true_word_embs_, true_context_labels_, num_neg_samples):
        expanded_labels_ = tf.expand_dims(true_context_labels_, 1)
        loss_ = tf.reduce_mean(
            tf.nn.nce_loss(self.context_embs_, self.context_biases_, expanded_labels_, true_word_embs_,
                           num_neg_samples, self.vocab_size))
        return loss_

    '''
    def sampled_softmax_loss_(self, true_word_embs_, true_context_labels_):
        """Build the graph for the NCE loss."""
        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(true_context_labels_,
                    dtype=tf.int64),
            [self.batch_size, 1])

        # Negative sampling.
        sampled_ids, true_expected_count, sampled_expected_count = \
            (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=self.num_neg_samples,
                unique=True,
                range_max=self.vocab_size,
                distortion=self.distortion,
                unigrams=self.vocab_counts.tolist()))

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(self.context_embs_, true_context_labels_)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(self.context_biases_, true_context_labels_)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(self.context_embs_, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(self.context_biases_, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(true_word_embs_, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self.num_neg_samples])
        sampled_logits = tf.matmul(true_word_embs_,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            true_logits, tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            sampled_logits, tf.zeros_like(sampled_logits))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size
        # nce_loss_tensor = tf.Print(nce_loss_tensor, [tf.shape(nce_loss_tensor)])

        return nce_loss_tensor
    '''

    def train(self,
              session,
              # Directory to write the model and training summaries.
              save_path="../out/word2vec",

              run_from_checkpoint=True,
              # "File consisting of analogies of four tokens."
              # "embedding 2 - embedding 1 + embedding 3 should be close to embedding 4."
              # "See README.md for how to get 'questions-words.txt'."
              eval_data="../data/questions-words.txt",
              # name of the loss fn
              loss_name="nce",
              # "Number of epochs to train. Each epoch processes the training data once completely.")
              epochs_to_train=30,
              # distortion. normally .75
              distortion=.75,
              # "Negative samples per training example.")
              num_neg_samples=100,
              # "Number of training examples processed per step (size of a minibatch).")
              batch_size=128,
              # The number of concurrent training steps.
              window_size=1,
              # Subsample threshold for word occurrence. Words that appear "
              # with higher frequency will be randomly down-sampled. Set to 0 to disable.")
              subsample=1e-3,

              # "name of the sgd optimizer"
              optimizer_name="sgd",
              # decay type of learning rate
              decay="linear",
              # Initial learning rate. normally .2
              learning_rate=0.2,

              summary_interval=5000,
              checkpoint_interval=50000):

        self.save_path = save_path

        self.add_param_group("train", {
            "ws": window_size,
            "on": optimizer_name,
            "lr": learning_rate,
            "bs": batch_size,
            "ss": subsample,
            "dist": distortion,
            "loss": loss_name,
        })

        print("Training: {}".format(self))

        if self.evaluator is None:
            # lazy load evaluator
            self.evaluator = AnalogyEvaluator(eval_data, self.id2word, self.word_embs_)

        global_step_ = tf.Variable(0, name="global_step")

        epoch_, num_words_seen_, words_, contexts_ = self.skipgram_(batch_size, window_size, subsample)
        example_embs_ = self.get_embeddings_(words_)

        if loss_name == "sampled_softmax":
            raise NotImplementedError
            # loss_ = self.sampled_softmax_loss_(example_embs_, contexts_)
        elif loss_name == "nce":
            loss_ = self.nce_loss_(example_embs_, contexts_, num_neg_samples)
        else:
            raise ValueError("{} is not a valid loss function".format(loss_name))

        train_, lr_ = optimize_(loss_, optimizer_name, learning_rate, decay, global_step_)

        self.log_scalar("NCE loss", loss_)

        self.load_or_initialize(session, try_load=run_from_checkpoint)

        """Train the model."""
        epoch = 0
        while epoch < epochs_to_train:
            epoch, global_step, _ = session.run([epoch_, global_step_, train_])

            self.log_emitter(session, global_step, summary_interval,
                             additional={'lr': lr_, 'epoch': epoch_, 'step': global_step_})

            if global_step % checkpoint_interval == 0:
                self.save_embeddings(save_path, str(self))
                self.save(session)

                analogy_accuracy, correct, total = self.evaluator.analogy_accuracy(session)
                self.emit_non_tensor("Analogy accuracy", correct * 100.0 / total, global_step)
                self.emit_non_tensor("Analogy correct", correct, global_step)
                self.emit_non_tensor("Analogy total", total, global_step)

    def get_nearby_word_(self, query_words_):
        # Nodes for computing neighbors for a given word according to their cosine distance.
        normalized_embs_ = tf.nn.l2_normalize(self.word_embs_, 1)
        nearby_embs_ = tf.gather(normalized_embs_, query_words_)
        nearby_dists_ = tf.matmul(nearby_embs_, normalized_embs_, transpose_b=True)
        nearby_val_, nearby_idx_ = tf.nn.top_k(nearby_dists_, min(1000, self.vocab_size))
        return nearby_val_, nearby_idx_

    def nearby(self, session, words, num=20):
        """Prints out nearby words given a list of words."""
        ids = np.array([self.word2id.get(x, 0) for x in words])
        query_words = tf.placeholder(dtype=tf.int32)
        nearby_val_node, nearby_idx_node = self.get_nearby_word_(query_words)
        vals, idx = session.run([nearby_val_node, nearby_idx_node], {query_words: ids})

        word_neighbor_list = []
        for i in xrange(len(words)):
            neighbor_list = []
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                neighbor_list.append((self.id2word[neighbor], distance))
            word_neighbor_list.append(neighbor_list)
        return word_neighbor_list
