from keras import Input, Model, optimizers
from keras.layers import Flatten, Bidirectional, Concatenate, Reshape, GRU
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding

def create_model(embeddings, max_features=50, maxlen=45, ):
    sentiments_input = Input(shape=(maxlen,), name='sentiments')
    sentiments = Reshape((maxlen, 1,))(sentiments_input)
    emb_input = Input(shape=(maxlen,), name='emb_input')
    embedding = Embedding(embeddings.shape[0], max_features, input_length=maxlen, weights=[embeddings], trainable=True)(
        emb_input)
    concat = Concatenate(axis=-1)([embedding, sentiments])
    lstm = Bidirectional(LSTM(output_dim=200, activation='relu', return_sequences=True))(concat)
    dropout = Dropout(0.5)(lstm)
    dense = Dense(1, activation='sigmoid')(dropout)
    out = Flatten()(dense)

    model = Model(inputs=[emb_input, sentiments_input], outputs=[out])
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
