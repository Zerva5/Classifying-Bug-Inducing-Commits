import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_BLOCKTIME'] = '1'
import sys
from create_dataset import create_dataset
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
from tensorflow.keras.layers import Input, Embedding, LSTM, Concatenate, Dense, Dot, Softmax, TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train, y_train, X, P, d, Y, _, _ = create_dataset(sys.argv[1], 12)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

path_vocab = tf.Variable(np.random.randn(len(P), d))
value_vocab = tf.Variable(np.random.randn(len(X), d))
W = tf.Variable(np.random.randn(d, 3*d))
attention_vector = tf.Variable(np.random.randn(d))

path_input = Input(shape=(None,), dtype="int32", name="path_input")
path_embedding = Embedding(len(P), d, name="path_embedding")(path_input)
path_encoder = LSTM(d, return_sequences=True, name="path_encoder")(path_embedding)

token_input_1 = Input(shape=(None,), dtype="int32", name="token_input_1")
token_embedding_1 = Embedding(len(X), d, name="token_embedding_1")(token_input_1)
token_input_2 = Input(shape=(None,), dtype="int32", name="token_input_2")
token_embedding_2 = Embedding(len(X), d, name="token_embedding_2")(token_input_2)

combined_input = Concatenate(axis=-2, name="combined_input")([path_encoder, token_embedding_1, token_embedding_2])
dense_layer = TimeDistributed(Dense(3*d, activation="tanh"), name="dense_layer")(combined_input)

encoder_model = Model(inputs=[path_input, token_input_1, token_input_2], outputs=dense_layer)

# Decoder
decoder_input = Input(shape=(None,), dtype="int32", name="decoder_input")
decoder_embedding = Embedding(len(Y), d, name="decoder_embedding")(decoder_input)
decoder_lstm = LSTM(d, return_sequences=True, name="decoder_lstm")
decoder_output = decoder_lstm(decoder_embedding)

# Attention mechanism
attention_vector = tf.Variable(np.random.randn(d))
attention_weights = Dot(axes=[2, 1], name="attention_weights")([decoder_output, dense_layer])
attention_weights = Softmax(name="attention_softmax")(attention_weights)
context_vector = Dot(axes=[1, 1], name="context_vector")([attention_weights, dense_layer])

# Combined output
combined_output = Concatenate(axis=-1, name="combined_output")([context_vector, decoder_output])
output = Dense(len(Y), activation="softmax", name="output")(combined_output)

decoder_model = Model(inputs=[decoder_input, path_input, token_input_1, token_input_2], outputs=output)

# Training the model
encoder_input_data = [X_train, X_train[:, :, 0], X_train[:, :, -1]]
decoder_input_data = y_train[:, :-1]
decoder_output_data = tf.keras.utils.to_categorical(y_train[:, 1:], num_classes=len(Y))

encoder_model.compile(optimizer="adam", loss="categorical_crossentropy")
decoder_model.compile(optimizer="adam", loss="categorical_crossentropy")

history = decoder_model.fit(
    [decoder_input_data, *encoder_input_data],
    decoder_output_data,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
)
