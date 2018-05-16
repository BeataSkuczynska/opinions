from keras import optimizers
from keras.layers import GRU, LSTM
from sklearn.model_selection import ParameterGrid

from opinions.train import load_embeddings, train


def grid_search():
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    params = {
        'emb': ["w2v_allwiki_nkjp300_50"],
        'rnn': [GRU, LSTM],
        'output_dim_rnn': [50, 100, 200, 300],
        'activation_rnn': ['relu', 'tanh'],
        'dropout': [0.1, 0.2,0.5,0.7],
        'optimizer': ['adam', sgd, 'adagrad'],
        'trainable_embeddings': [True, False],
        'cut_length': [10, 20, 50, 65],

    }

    grid = ParameterGrid(params)
    with open('grid_results.csv', 'w+') as f:
        for param in list(grid):
            print(param)
            train_accuracy, test_accuracy, confusion_m = train("json/OPTA-treebank-0.1.json",
                                                               load_embeddings(param['emb']), param['cut_length'],
                                                               param)
            f.write('{}\t{}\t{}\t{}\n'.format(param, train_accuracy, test_accuracy, confusion_m))


if __name__ == "__main__":
    grid_search()
