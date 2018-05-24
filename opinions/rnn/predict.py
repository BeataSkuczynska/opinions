import argparse
import json
import pickle

import numpy as np

from keras.models import load_model

from opinions.rnn.train import prepare_data, load_embeddings


def predict(path, emb, vocab, max_len, index2vec, model):
    _, sentences, _, sentiments, _, targets, padded_targets, _, max_len, _, relevant = prepare_data([path], None,
                                                                                                    test=1.0,
                                                                                                    vocab=vocab,
                                                                                                    max_len=max_len,
                                                                                                    index2vec=index2vec)
    predictions = model.predict([sentences, sentiments, relevant], verbose=0)
    print(model.evaluate([sentences, sentiments, relevant], padded_targets))
    unpad = ""
    for idx, val in enumerate(targets):
        for prediction in predictions[idx][max_len - len(val):].tolist():
            predict_val = np.argmax(prediction)
            if predict_val != 2:
                unpad += "_"
            else:
                unpad += "A"
            unpad += "\n"
        unpad += "\n"
    with open("output/rnn_output.conll", "w+") as f:
        f.write(unpad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for option NN training')
    parser.add_argument('path', type=str, help='Data to prediction path')
    args = parser.parse_args()

    model = load_model('generated/model.h5')
    max_len = int(open("generated/max_len").read())
    with open("generated/vocab.json", "r") as f:
        vocab = json.load(f)
    index2vec = pickle.load(open("generated/embeddings.pkl", "rb"))

    predict(args.path, None, vocab, max_len, index2vec, model)