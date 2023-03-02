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

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

# def cross_entropy_loss(y_true, y_pred):
#     return -np.sum(y_true * np.log(y_pred))
def cross_entropy_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    print(y_true)
    print(y_pred)
    return loss_fn(y_true, y_pred)


# Initialize the model's parameters
def initialize_embeddings_matrix(vocab, d):
    return np.random.randn(len(vocab), d)


X_train, y_train, input_dim, output_dim, X, P, d, Y = create_dataset(sys.argv[1], 12)

path_vocab = initialize_embeddings_matrix(P, d)
value_vocab = initialize_embeddings_matrix(X, d)
W = np.random.randn(d, 3*d)
attention_vector = np.random.randn(d)
tags_vocab = initialize_embeddings_matrix(Y, d)

path_vocab = tf.constant(path_vocab, dtype=tf.float32)
value_vocab = tf.constant(value_vocab, dtype=tf.float32)
W = tf.constant(W, dtype=tf.float32)
attention_vector = tf.constant(attention_vector, dtype=tf.float32)
tags_vocab = tf.constant(tags_vocab, dtype=tf.float32)

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

        context_vector = np.concatenate([xs_embedding, pj_embedding, xt_embedding])
        context_vectors.append(context_vector)
    
    # Combine the context vectors using the learned weights
    combined_context_vectors = []
    for context_vector in context_vectors:
        # combined_context_vector = np.tanh(W @ context_vector)
        combined_context_vector = np.tanh(W @ context_vector.reshape(-1, 1))
        combined_context_vectors.append(combined_context_vector)
    
    # Compute the attention weights
    attention_weights = []
    for combined_context_vector in combined_context_vectors:
        # combined_context_vector = np.ravel(combined_context_vector)
        attention_weight = np.exp(combined_context_vector.transpose() @ attention_vector[:, np.newaxis])
        # print("att1 ",attention_weight.shape, attention_weight)
        divisor = sum([np.exp(c.transpose() @ attention_vector[:, np.newaxis]) for c in combined_context_vectors])
        attention_weight /= divisor
        # print("att2 ",attention_weight.shape, attention_weight)
        attention_weight = attention_weight[0][0]
        # print("att3 ",attention_weight.shape, attention_weight)
        attention_weights.append(attention_weight)
    
    # Aggregate the context vectors with the attention weights
    code_vector = sum([attention_weight * combined_context_vector for attention_weight, combined_context_vector in zip(attention_weights, combined_context_vectors)])
    
    code_vector = np.ravel(combined_context_vector)

    print(code_vector.shape)
    return code_vector

    # # Predict the tags using the code vector
    # tag_probabilities = softmax([code_vector @ tag_embedding for tag_embedding in tags_vocab])
    
    # return tag_probabilities


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

            # Perform the forward pass
            code_vector = forward_pass(X_train[i])

            # Compute the predicted tag probabilities
            tag_probabilities = tf.nn.softmax([code_vector[:, np.newaxis] @ tf.reshape(tag_embedding, [-1, d]) for tag_embedding in tags_vocab])

            # Compute the loss
            loss = cross_entropy_loss(Y_train[i], tag_probabilities)

            # Compute the gradients using backpropagation
            with tf.GradientTape() as tape:
                tape.watch([path_vocab, value_vocab, W, attention_vector, tags_vocab])
                code_vector = forward_pass(X_train[i])
                print("cv",code_vector.shape)
                print("tv",tags_vocab.shape)
                tag_probabilities = tf.nn.softmax([code_vector[:, np.newaxis] @ np.transpose(tag_embedding[:, np.newaxis]) for tag_embedding in tags_vocab])
                loss = cross_entropy_loss(Y_train[i], tag_probabilities)

            gradients = tape.gradient(loss, [path_vocab, value_vocab, W, attention_vector, tags_vocab])
            # gradients = tape.gradient(loss, [tf.convert_to_tensor(path_vocab), tf.convert_to_tensor(value_vocab), tf.convert_to_tensor(W), tf.convert_to_tensor(attention_vector), tf.convert_to_tensor(tags_vocab)])

            # Update the weights using the optimizer
            optimizer.apply_gradients(zip(gradients, [path_vocab, value_vocab, W, attention_vector, tags_vocab]))

    # Return the trained parameters
    return path_vocab, value_vocab, W, attention_vector, tags_vocab

path_vocab, value_vocab, W, attention_vector, tags_vocab = train(X_train, y_train, forward_pass, epochs=10, learning_rate=0.001)

X_eval, y_eval, _, _, _, _, _, _ = create_dataset(sys.argv[1], 16)

y_pred = []
for i in range(len(X_eval)):
    code_vector = forward_pass(X_eval[i])
    tag_probabilities = softmax([code_vector @ tag_embedding for tag_embedding in tags_vocab])
    predicted_tag = np.argmax(tag_probabilities)
    y_pred.append(predicted_tag)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_eval, y_pred)
precision = precision_score(y_eval, y_pred, average='macro')
recall = recall_score(y_eval, y_pred, average='macro')
f1 = f1_score(y_eval, y_pred, average='macro')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

