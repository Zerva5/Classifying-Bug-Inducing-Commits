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
from tensorflow.keras.layers import Input, Dense, LSTM, Attention, Concatenate, Layer, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train, y_train, X, P, d, Y, _, _ = create_dataset(sys.argv[1], 12)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test)




class AttentionCodeVectorizer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(AttentionCodeVectorizer, self).__init__()
        self.path_vocab = self.add_weight(name='path_vocab', shape=(len(P), d), initializer='random_normal')
        self.value_vocab = self.add_weight(name='value_vocab', shape=(len(X), d), initializer='random_normal')
        self.W = self.add_weight(name='W', shape=(d, 3*d), initializer='random_normal')
        self.attention_vector = self.add_weight(name='attention_vector', shape=(d,), initializer='random_normal')
        self.dense_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs):
        path_contexts = inputs
        context_vectors = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(tf.shape(path_contexts)[0]):
            path_context = path_contexts[i]
            pj = path_context
            xs = tf.cast(path_context[0], tf.int32)
            xt = tf.cast(path_context[-1], tf.int32)
            xs_embedding = tf.gather(self.value_vocab, xs)
            pj_embedding = self.path_vocab[P.index(pj)]
            xt_embedding = tf.gather(self.value_vocab, xt) 

            context_vector = tf.concat([xs_embedding, pj_embedding, xt_embedding], axis=0)
            context_vectors = context_vectors.write(i, context_vector)

        context_vectors = context_vectors.stack()

        # Combine context vectors using fully connected layer
        context_weights = tf.matmul(context_vectors, tf.transpose(self.W))
        combined_context_vectors = tf.nn.tanh(context_weights)

        # Compute attention weights
        attention_weights = tf.nn.softmax(tf.matmul(combined_context_vectors, tf.expand_dims(self.attention_vector, axis=1)), axis=0)
        
        # Aggregate into code vector using attention
        code_vector = tf.reduce_sum(tf.multiply(combined_context_vectors, attention_weights), axis=0)

        # Reshape the code vector to have at least two dimensions
        code_vector = tf.reshape(code_vector, (1, -1))

        # Pass the code vector through a Dense layer for binary classification
        output = self.dense_layer(code_vector)

        return output


input_layer = tf.keras.layers.Input(shape=(None,))
code_vector = AttentionCodeVectorizer(len(P), d)(input_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(code_vector)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train.tolist(), y_train, batch_size=32, epochs=10)


###############



# path_vocab = tf.Variable(np.random.randn(len(P), d))
# value_vocab = tf.Variable(np.random.randn(len(X), d))
# W = tf.Variable(np.random.randn(d, 3*d))
# attention_vector = tf.Variable(np.random.randn(d))


# @tf.function
# def compute_attention_code_vector(path_contexts):
#     # Map each path-context to its corresponding embedding
#     context_vectors = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     for i in tf.range(tf.shape(path_contexts)[0]):
#         path_context = path_contexts[i]
#         pj = path_context
#         xs = tf.cast(path_context[0], tf.int32)
#         xt = tf.cast(path_context[-1], tf.int32)

#         xs_embedding = tf.gather(value_vocab, xs)
#         pj_embedding = tf.gather(path_vocab, tf.where(tf.equal(tf.cast(P, tf.float32), pj))[0][0])
#         xt_embedding = tf.gather(value_vocab, xt) 

#         context_vector = tf.concat([xs_embedding, pj_embedding, xt_embedding], axis=0)
#         context_vectors = context_vectors.write(i, context_vector)

#     context_vectors = context_vectors.stack()

#     # Combine context vectors using fully connected layer
#     context_weights = tf.matmul(context_vectors, tf.transpose(tf.cast(W, tf.float32)))
#     combined_context_vectors = tf.nn.tanh(context_weights)

#     # Compute attention weights
#     attention_weights = tf.nn.softmax(tf.matmul(combined_context_vectors, tf.expand_dims(tf.cast(attention_vector, tf.float32), axis=1)), axis=0)
    
#     # Aggregate into code vector using attention
#     code_vector = tf.reduce_sum(tf.multiply(combined_context_vectors, attention_weights), axis=0)
    
#     return code_vector


# # define the input shape for your variable-length array
# input_shape = (None,)

# # create your model
# model = tf.keras.models.Sequential()

# # add an attention layer to compute the code vector
# model.add(tf.keras.layers.Lambda(compute_attention_code_vector, input_shape=input_shape))

# # add a dense layer for binary classification
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# # compile the model with binary cross-entropy loss
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
