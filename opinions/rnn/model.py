from keras import Input, Model, optimizers
from keras.layers import Flatten, Bidirectional, Concatenate, Reshape, GRU
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding


def create_model(embeddings, max_features, maxlen, params):
    sentiments_input = Input(shape=(maxlen,), name='sentiments')
    sentiments = Reshape((maxlen, 1,))(sentiments_input)
    relevant_input = Input(shape=(maxlen,), name='relevant_attr')
    relevant = Reshape((maxlen, 1,))(relevant_input)
    emb_input = Input(shape=(maxlen,), name='emb_input')
    embedding = Embedding(embeddings.shape[0], max_features, input_length=maxlen, weights=[embeddings],
                          trainable=params['trainable_embeddings'])(
        emb_input)
    concat = Concatenate(axis=-1)([embedding, sentiments,relevant])
    lstm = Bidirectional(
        params['rnn'](output_dim=params['output_dim_rnn'], activation=params['activation_rnn'], return_sequences=True))(
        concat)
    dropout = Dropout(params['dropout'])(lstm)
    out = Dense(3, activation='sigmoid')(dropout)

    model = Model(inputs=[emb_input, sentiments_input, relevant_input], outputs=[out])

    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    model.save('model.h5')
    return model
