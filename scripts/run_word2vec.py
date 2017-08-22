import tensorflow as tf
import sys

try:
    from tflab.utils import call_with_flags, instantiate_with_flags
    from tflab.word2vec import Word2Vec
    top_dir = "../"
except ImportError:
    from tflab.tflab.utils import call_with_flags, instantiate_with_flags
    from tflab.tflab.word2vec import Word2Vec
    top_dir = "../../"


flags = tf.app.flags

flags.DEFINE_string("save_path", top_dir+"out/word2vec",
                    "Directory to write the model and training summaries.")
flags.DEFINE_bool("run_from_checkpoint", True,
                  "whether to try to load the model")

flags.DEFINE_string("train_data", top_dir+"data/text8",
                    "Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("eval_data", top_dir+"data/questions-words.txt",
                    "File consisting of analogies of four tokens."
                    "embedding 2 - embedding 1 + embedding 3 should be close to embedding 4."
                    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_string("optimizer_name", "sgd", "name of the sgd optimizer")
flags.DEFINE_string("loss_name", "nce", "name of the loss fn")
flags.DEFINE_integer("embedding_size", 256, "The embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 30,
                     "Number of epochs to train. Each epoch processes the training data once completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate. normally .2")
flags.DEFINE_float("distortion", .75, "distortion. normally .75")
flags.DEFINE_integer("num_neg_samples", 100, "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of training examples processed per step (size of a minibatch).")
flags.DEFINE_integer("window_size", 1,
                     "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("min_count", 3,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set to 0 to disable.")
flags.DEFINE_integer("summary_interval", 50,
                     "Print statistics every n steps.")
flags.DEFINE_integer("checkpoint_interval", 50000,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "steps")
flags.DEFINE_string("name", "word2vec", "name of word2vec for scoping")
flags.DEFINE_string("decay", "exponential", "type of decay on the learning rate")
FLAGS = flags.FLAGS


def main(_):
    """Train a word2vec model."""
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = instantiate_with_flags(Word2Vec, FLAGS, session=session)  # type: Word2Vec
            call_with_flags(model.train, FLAGS, session=session)
            print(model.evaluator.analogy_accuracy(session))


if __name__ == "__main__":
    tf.app.run()
