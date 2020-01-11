from models.convnets import ConvolutionalNet
from keras.models import load_model
from keras.preprocessing import sequence
from preprocessors.preprocess_text import clean
import sys
import string
import re
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
SEQUENCE_LENGTH = 20
EMBEDDING_DIMENSION = 30

UNK = "<UNK>"
PAD = "<PAD>"

vocabulary = open("data/vocabulary.txt").read().split("\n")
inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))


def words_to_indices(inverse_vocabulary, words):
    return [inverse_vocabulary.get(word, inverse_vocabulary[UNK]) for word in words]


class Predictor(object):
    def __init__(self, model_path):
        model = ConvolutionalNet(vocabulary_size=len(vocabulary), embedding_dimension=EMBEDDING_DIMENSION,
                                 input_length=SEQUENCE_LENGTH)
        model.load_weights(model_path)
        self.model = model

    def predict(self, headline):
        headline = headline.encode("ascii", "ignore")
        inputs = sequence.pad_sequences([words_to_indices(inverse_vocabulary, clean(headline).lower().split())],
                                        maxlen=SEQUENCE_LENGTH)
        clickbaitiness = self.model.predict(inputs)[0, 0]
        return clickbaitiness


predictor = Predictor("models/detector.h5")
if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1], header=None, sep="\t")
    labels = data.iloc[:, 0].values.tolist()
    gen_features = data.iloc[:, 1].values.tolist()
    features = data.iloc[:, 2].values.tolist()
    pred_gen = []
    pre = []
    pre_logits = []
    for gen_f in gen_features:
        pred_gen.append(1 if predictor.predict(sys.argv[1]) > 0.5 else 0)
    for f in features:
        pre.append(1 if predictor.predict(sys.argv[1]) > 0.5 else 0)
    acc = accuracy_score(labels, pre)
    acc_gen = accuracy_score(labels, pred_gen)
    c_m = confusion_matrix(labels, pre)
    c_m_gen = confusion_matrix(labels, pred_gen)
    print("The accuracy for raw is {}, Confusion Matrix is {}".format(acc, c_m))
    print("The accuracy for gen is {}, Confusion Matrix is {}".format(acc_gen, c_m_gen))
    # print("headline is {0} % clickbaity".format(round(predictor.predict(sys.argv[1]) * 100, 2)))