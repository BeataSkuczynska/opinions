from keras import optimizers
from keras.layers import GRU, LSTM
from sklearn.model_selection import ParameterGrid

from opinions.train import load_embeddings, train, prepare_data, train_eval


def grid_search():
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    params = {
        'rnn': [GRU, LSTM],
        'output_dim_rnn': [200, 300],
        'activation_rnn': ['relu', 'tanh'],
        'dropout': [0.5, 0.6],
        'optimizer': ['adam'],
        'trainable_embeddings': [True, False],

    }
    grid = ParameterGrid(params)
    emb_files = ["w2v_allwiki_nkjp300_50"]
    for emb in emb_files:
        emb = load_embeddings(emb)
        values = prepare_data("json/OPTA-treebank-0.1.json", emb)
        with open('grid_results_3.csv', 'w+') as f:
            for param in list(grid):
                print(param)
                train_accuracy, test_accuracy, confusion_m = train_eval(values, emb, param)
                tp = confusion_m[2][2]
                fn = confusion_m[0][2] + confusion_m[1][2]
                fp = confusion_m[2][1] + confusion_m[2][0]
                precision = tp /  (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2/ (1/precision+1/recall)
                f.write('{}\t{}\t{}\t{}\n{}\t{}\t{}'.format(param, train_accuracy, test_accuracy, confusion_m, precision,recall,f1))


if __name__ == "__main__":
    grid_search()
