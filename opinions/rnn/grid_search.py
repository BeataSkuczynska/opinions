from keras import optimizers
from keras.layers import GRU, LSTM
from sklearn.model_selection import ParameterGrid

from opinions.rnn.train import load_embeddings, prepare_data, train_eval


def grid_search():
    params = {
        'rnn': [GRU, LSTM],
        'output_dim_rnn': [200, 300],
        'activation_rnn': ['relu', 'tanh'],
        'dropout': [0.5, 0.6],
        'optimizer': ['adam'],
        'trainable_embeddings': [True, False],

    }
    grid = ParameterGrid(params)
    emb_files = ["w2v_allwiki_nkjp300_full"]
    for emb in emb_files:
        emb = load_embeddings(emb)
        values = prepare_data("OPTA-treebank/OPTA-treebank-0.1.json", emb)
        with open('output/grid_results.csv', 'w+') as f:
            for param in list(grid):
                _, test_accuracy = train_eval(values, emb, param)
                f.write('{}\t{}'.format(param, test_accuracy))


if __name__ == "__main__":
    grid_search()
