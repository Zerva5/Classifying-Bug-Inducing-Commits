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
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

X_train, y_train, input_dim, output_dim, X, P, d, Y = create_dataset(sys.argv[1], 128)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

path_vocab = np.random.randn(len(P), d)
value_vocab = np.random.randn(len(X), d)
W = np.random.randn(d, 3*d)
attention_vector = np.random.randn(d)

path_vocab = tf.Variable(path_vocab)
value_vocab = tf.Variable(value_vocab)
W = tf.Variable(W)
attention_vector = tf.Variable(attention_vector)

# Define a dense layer to converge code vector into a single prediction
dense_layer = Dense(1, activation='softmax')

# Define the forward pass
def forward_pass(path_contexts):
    
    # Map each path-context to its corresponding embedding
    context_vectors = []
    for path_context in path_contexts:
        pj = path_context
        xs = path_context[0]
        xt = path_context[-1]

        xs_embedding = value_vocab[xs]
        pj_embedding = path_vocab[P.index(pj)]
        xt_embedding = value_vocab[xs] 

        context_vector = tf.concat([xs_embedding, pj_embedding, xt_embedding], axis=0)
        context_vectors.append(context_vector)

    # Combine context vectors using fully connected layer
    # context_weights = tf.matmul(context_vectors, W)
    context_weights = tf.matmul(context_vectors, tf.transpose(W))

    combined_context_vectors = tf.nn.tanh(context_weights)

    # Compute attention weights
    attention_weights = tf.nn.softmax(tf.matmul(combined_context_vectors, tf.expand_dims(attention_vector, axis=1)), axis=0)
    
    # Aggregate into code vector using attention
    code_vector = tf.reduce_sum(tf.multiply(combined_context_vectors, attention_weights), axis=0)
    
    return code_vector


def train(X_train, Y_train, forward_pass, epochs, learning_rate):
    
    # Define the optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Iterate over the epochs
    for epoch in range(epochs):
        
        # Shuffle the training data
        indices = list(range(len(X_train)))
        random.shuffle(indices)
        X_train = [X_train[i] for i in indices]
        Y_train = [Y_train[i] for i in indices]

        # Iterate over the training examples
        for i in range(len(X_train)):
            
            # Get the current example
            x = X_train[i]
            y = Y_train[i]

            with tf.GradientTape() as tape:

                # Perform the forward pass
                code_vector = forward_pass(x)

                # Apply the dense layer to converge code vector into a single prediction
                code_prediction = dense_layer(tf.expand_dims(code_vector, 0))

                # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=code_prediction))
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y, (1, 1)), logits=code_prediction))

                # Compute gradients
                grads = tape.gradient(loss, [path_vocab, value_vocab, W, attention_vector])

                # Apply gradients
                optimizer.apply_gradients(zip(grads, [path_vocab, value_vocab, W, attention_vector]))

    return path_vocab, value_vocab, W, attention_vector

            

path_vocab, value_vocab, W, attention_vector = train(X_train, y_train, forward_pass, epochs=10, learning_rate=0.001)

y_pred = []
for x in X_test:
    code_vector = forward_pass(x)
    code_prediction = dense_layer(tf.expand_dims(code_vector, 0))
    y_pred.append(code_prediction.numpy()[0][0])
y_pred = np.array(y_pred)
y_true = np.array(y_test)
accuracy = np.mean(y_pred == y_true)
print("Accuracy:", accuracy)
