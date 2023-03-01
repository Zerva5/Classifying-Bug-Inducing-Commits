import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_BLOCKTIME'] = '1'

import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LSTM, Dense, Dropout, Lambda, Dot, Activation, Reshape
import tensorflow.python.keras.backend as K
import sys;
from process_commit import process_commit
from random_commit import get_random_commit_shas
from tensorflow.python.keras.callbacks import LambdaCallback

def progress_report(epoch, logs):
    print('Epoch {}: loss={:.4f}, accuracy={:.4f}'.format(epoch, logs['loss'], logs['accuracy']))

if __name__ == "keras":

    commits = get_random_commit_shas(sys.argv[1], 128)
    X_train = [process_commit(commit, sys.argv[1]) for commit in commits]
    X_train = [data for data in X_train if len(data) > 0]
    y_train = [(1 if len(x) > 50 else 0) for x in X_train]

    print(y_train)

    # Determine the input and output dimensions
    input_dim = len(X_train[0][0])
    output_dim = len(X_train[0][0][0])

    # Define the model
    def attention_model(input_dim, output_dim):
        inputs = Input(shape=(None, input_dim))
        x = Reshape((-1, 1))(inputs)
        x = LSTM(output_dim, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = Lambda(lambda x: K.sum(x, axis=1))(x)
        x = Dense(output_dim, activation='relu')(x)
        x = Dense(1)(x)
        x = Activation('softmax')(x)
        x = Dot(axes=1)([x, x])
        model = Model(inputs=inputs, outputs=x)
        return model

    # Create an instance of the model
    model = attention_model(input_dim, output_dim)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Define the LambdaCallback function
    progress_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: progress_report(epoch, logs))

    # Train the model with the progress_callback
    model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=1, callbacks=[progress_callback])

    # Use the trained model to map variable-length arrays to fixed-length vectors
    fixed_length_vectors = model.predict(X_train)

    print(fixed_length_vectors)
    print(fixed_length_vectors[0])