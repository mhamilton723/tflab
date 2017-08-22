import numpy as np
import tensorflow as tf 

class AnalogyEvaluator(object):
    """Class to evaluate analogies"""

    def __init__(self, analogy_path, id2word, word_embs_):
        self.word_embs_ = word_embs_
        self.id2word = id2word
        self.word2id = {}
        for i,w in enumerate(self.id2word):
            self.word2id[w] = i
        self.analogy_questions = self.read_analogies(analogy_path)

    def read_analogies(self, eval_data):
        """
        Reads through the analogy question file.

        questions_skipped: questions skipped due to unknown words.
        Returns:
          questions: a [n, 4] numpy array containing the analogy question's word ids.
        """
        questions = []
        questions_skipped = 0
        with open(eval_data, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self.word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file: ", eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        return np.array(questions, dtype=np.int32)

    def analogy_accuracy(self, session):
        analogy_questions = self.analogy_questions

        """Evaluate analogy questions and reports accuracy."""
        # How many questions we get right at precision@1.
        correct = 0
        total = analogy_questions.shape[0]

        start = 0
        while start < total:
            limit = start + 2500
            sub = analogy_questions[start:limit, :]
            answers = self.answer_analogies(session, sub[:, :3])
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if answers[question, j] == sub[question, 3]:
                        # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                        correct += 1
                        break
                    elif answers[question, j] in sub[question, :3]:
                        # We need to skip words already in the question.
                        continue
                    else:
                        # The correct label is not the precision@1
                        break
        analogy_accuracy = correct * 100.0 / total
        return analogy_accuracy, correct, total

    def answer_analogies(self, session, analogies):
        """Predict the top 4 answers for analogy questions."""
        analogies_ = tf.placeholder(dtype=tf.int32, shape=(None, 3))
        answers_node = self.answer_analogies_(analogies_)
        answers, = session.run([answers_node], {analogies_: analogies})
        return answers

    def answer_analogies_(self, analogies_):
        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.
        analogy_a_ = analogies_[:, 0]
        analogy_b_ = analogies_[:, 1]
        analogy_c_ = analogies_[:, 2]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        normalized_embs_ = tf.nn.l2_normalize(self.word_embs_, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector. They all have the shape [N, emb_dim]
        a_emb_ = tf.gather(normalized_embs_, analogy_a_)
        b_emb_ = tf.gather(normalized_embs_, analogy_b_)
        c_emb_ = tf.gather(normalized_embs_, analogy_c_)

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target_ = c_emb_ + (b_emb_ - a_emb_)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist_ = tf.matmul(target_, normalized_embs_, transpose_b=True)

        _, pred_idx_ = tf.nn.top_k(dist_, 4)
        return pred_idx_

    def analogy(self, session, w0, w1, w2):
        """Predict word w3 as in w0:w1 vs w2:w3."""
        wid = np.array([[self.word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self.answer_analogies(session, wid)
        for c in [self.id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                return c
        return None
