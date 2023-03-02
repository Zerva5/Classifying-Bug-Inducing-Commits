import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_BLOCKTIME'] = '1'
import sys
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation, Input, Multiply, Lambda
from create_dataset import create_dataset

X_train, y_train, input_dim, output_dim = create_dataset(sys.argv[1])

# Convert variable-length arrays into fixed-length vectors
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(X_train)

# Define attention layer
def attention_layer(input_shape):
    input_layer = Input(shape=input_shape)
    attention_weights = Dense(input_shape[-1], activation='softmax')(input_layer)
    attention_output = Multiply()([input_layer, attention_weights])
    output_layer = Lambda(lambda x: K.sum(x, axis=1))(attention_output)
    return Model(inputs=input_layer, outputs=output_layer)

# Define MLP classifier with attention layer
model = Sequential()
model.add(Dense(32, input_shape=(X.shape[1],)))
model.add(Activation('relu'))
model.add(attention_layer((X.shape[1],)))
model.add(Dense(1, activation='sigmoid'))

# Compile and fit the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y_train, epochs=10, batch_size=32)
