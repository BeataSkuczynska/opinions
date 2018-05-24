import json
import pickle

from keras.models import load_model
from sklearn.metrics import confusion_matrix

from opinions.rnn.dataset import parse_data
from opinions.rnn.predict import predict
from opinions.rnn.train import load_embeddings, prepare_data, train


def create_confusion_matrix(gold, path):
    print("Confusion matrix for: ".format(path))
    with open(path, "r") as f:
        rnn_data = [1 if word == "_" else 2 for word in f.read().splitlines() if word]
    confusion_m = confusion_matrix(rnn_data, gold)
    print(confusion_m)
    return confusion_m


def count_f1(confusion_m):
    tp = confusion_m[1][1]
    fn = confusion_m[0][1]
    fp = confusion_m[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 / ((1 / precision) + (1 / recall))
    return precision, recall, f1


def evaluate(trainset, test, emb_file):

    emb = load_embeddings(emb_file)
    prepare_data([trainset, test], emb)

    max_len = int(open("generated/max_len").read())
    with open("generated/vocab.json", "r") as f:
        vocab = json.load(f)
    index2vec = pickle.load(open("generated/embeddings.pkl", "rb"))

    model, _ = train([trainset], emb, vocab=vocab, max_len=max_len, index2vec=index2vec)
    predict(test, emb, vocab, max_len, index2vec, load_model('generated/model.h5'))

    _, _, targets, _, _, _ = parse_data([test], vocab=vocab, max_len=max_len)
    gold = [i for x in targets for i in x]

    cm_rnn = create_confusion_matrix(gold, "output/rnn_output.conll")
    cm_crf_s = create_confusion_matrix(gold, "output/tagged_output_S_skladnica_fixed.conll")
    cm_crf = create_confusion_matrix(gold, "output/tagged_output_skladnica_fixed.conll")
    rnn = count_f1(cm_rnn)
    crf_s = count_f1(cm_crf_s)
    crf = count_f1(cm_crf)
    print("Evaluation Results")
    print("Model\tprecision\trecall\tF1 score")
    for result in [rnn, crf_s, crf]:
        print("{}\t{}\t{}".format(result[0], result[1], result[2]))


if __name__ == "__main__":
    emb_file = "w2v_allwiki_nkjpfull_300"
    print("Train on OPTA-treebank test on skladnica")
    evaluate("input/OPTA-treebank/OPTA-treebank-0.1.json", "input/skladnica/skladnica_output.json", emb_file)
