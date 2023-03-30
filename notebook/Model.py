import numpy as np
import os
import gc

#Disable import warnings for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Attention, Concatenate, Layer,
    Embedding, TimeDistributed, Multiply,
    Lambda, Add, Masking, GlobalMaxPooling2D, Reshape, MaxPooling1D, MaxPooling2D,
    Dropout, Conv1D, Conv2D, Bidirectional, GRU, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D, Softmax, Dot
)
from tensorflow.keras.initializers import RandomUniform
from gradient_accumulator import GradientAccumulateModel
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from vocab import MAX_NODE_LOOKUP_NUM
from tqdm.auto import tqdm

def CommitDiffModelFactory(
    BAG_SIZE = 256,
    CONTEXT_SIZE = 16,
    OUTPUT_SIZE = 128
):

    class ClearMemory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            k.clear_session()

    
    class SOMEncoderLayer(Layer):
        def __init__(self, max_node_lookup_num, embedding_dim, context_size, bag_size, som_grid_size, fixed_vector_size):
            super(SOMEncoderLayer, self).__init__()
            self.max_node_lookup_num = max_node_lookup_num
            self.embedding_dim = embedding_dim
            self.context_size = context_size
            self.bag_size = bag_size
            self.som_grid_size = som_grid_size

            self.embedding = Embedding(input_dim=self.max_node_lookup_num + 1, output_dim=self.embedding_dim)
            self.conv2d = Conv2D(filters=self.convolutional_filters, kernel_size=(6, 6), activation='relu')
            self.max_pooling = MaxPooling2D(pool_size=(self.context_size, 1))
            self.reshape = Reshape((-1, 32))
            self.lstm = LSTM(units=64)
            
            # Initialize SOM weights
            self.som_weights = self.add_weight(name="som_weights",
                                            shape=(self.som_grid_size * self.som_grid_size, 64),
                                            initializer=RandomUniform(minval=-1, maxval=1),
                                            trainable=True)
                                            
            self.dense = Dense(fixed_vector_size, activation='sigmoid')

        def call(self, inputs):
            embedded_inputs = self.embedding(inputs)
            conv = self.conv2d(embedded_inputs)
            max_pooling = self.max_pooling(conv)
            flattened = self.reshape(max_pooling)
            lstm = self.lstm(flattened)

            # Calculate the Euclidean distance between input and SOM grid
            tiled_inputs = tf.tile(lstm[:, tf.newaxis, :], [1, self.som_grid_size * self.som_grid_size, 1])
            distance = tf.reduce_sum(tf.square(tiled_inputs - self.som_weights), axis=-1)

            # Find the index of the minimum distance
            winning_index = tf.argmin(distance, axis=-1)

            # Convert the index to one-hot encoded matrix
            one_hot_winners = tf.one_hot(winning_index, self.som_grid_size * self.som_grid_size)

            # Map one-hot encoded matrix to the desired output size
            output_vector = self.dense(one_hot_winners)

            return output_vector

    class CapsuleEncoder(Layer):
        def __init__(self, num_capsules=10, capsule_dim=16, **kwargs):
            super(CapsuleEncoder, self).__init__(**kwargs)
            self.num_capsules = num_capsules
            self.capsule_dim = capsule_dim

        def squash_activation(self, x, axis=-1):
            squared_norm = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
            scale = squared_norm / (1 + squared_norm) / (tf.sqrt(squared_norm + tf.keras.backend.epsilon()) + tf.keras.backend.epsilon())
            return scale * x

        def build(self, input_shape):
            self.primary_capsule_conv1d = Conv1D(filters=32, kernel_size=self.kernel_size, strides=2, activation='relu')
            self.secondary_capsule_dense = Dense(self.num_capsules * self.capsule_dim)

        def call(self, inputs):
            # Create primary capsules
            primary_capsules = self.primary_capsule_conv1d(inputs)
            primary_capsules = Reshape((-1, 8))(primary_capsules)
            primary_capsules = Lambda(self.squash_activation)(primary_capsules)

            # Create secondary capsules
            secondary_capsules = self.secondary_capsule_dense(primary_capsules)
            secondary_capsules = Reshape((-1, self.num_capsules, self.capsule_dim))(secondary_capsules)
            secondary_capsules = Lambda(self.squash_activation)(secondary_capsules)

            # Calculate the vector length of the secondary capsules
            capsule_lengths = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)))(secondary_capsules)

            # Clip the capsule lengths to the range of [0, 1]
            capsule_lengths = tf.clip_by_value(capsule_lengths, clip_value_min=0.0, clip_value_max=1.0)

            # Flatten the capsule lengths
            flattened_capsule_lengths = Flatten()(capsule_lengths)

            return flattened_capsule_lengths


    class TCNEncoder(Layer):
        def __init__(self, filters=50, **kwargs):
            super(TCNEncoder, self).__init__(**kwargs)
            self.filters = filters

        def residual_block(self, x, dilation_rate, conv_layer):
            prev_x = x
            for k in range(2):
                x = conv_layer(x)
            return Add()([prev_x, x])

        def build(self, input_shape):
            self.conv_layer = Conv2D(filters=self.filters,
                                    kernel_size=(1, 2),
                                    dilation_rate=(1, 1),
                                    padding="same",
                                    activation="relu")

        def call(self, inputs):
            x = inputs

            # Apply residual blocks with different dilation rates
            for dilation_rate in [1, 2, 4]:
                x = self.residual_block(x, dilation_rate, self.conv_layer)

            return x


    class CommitDiffModel:
        def __init__(self, unsupervised_epochs, supervised_epochs, siam_batch_size, steps_per_update = 8):
            self.input_shape = (BAG_SIZE, CONTEXT_SIZE)
            self.bag_size = BAG_SIZE
            self.context_size = CONTEXT_SIZE
            self.fixed_vector_size = OUTPUT_SIZE
            self.rate = 0.1
            self.activation_fn1 = "relu"
            self.activation_fn2 = "relu"
            self.activation_fn3 = "sigmoid"
            self.loss_fn = "binary_crossentropy"
            self.embedding_dim = 64

            ##### Learning Rate Hyperparams #####
            self.base_lr = 0.075
            self.weight_decay = 0.00005
            self.momentum = 0.925

            self.supervised_base_lr = 0.0001
            #####################################
            
            self.unsupervised_epochs = unsupervised_epochs
            self.siam_batch_size = siam_batch_size
            self.steps_per_update = steps_per_update
            self.supervised_epochs = supervised_epochs

            #### These hyperparams should be experimented with #############
            self.convolutional_filters = 32
            self.lstm_memory_units = 64
            self.kernel_size = 3
            #################################################################
            
            self.encoder = None
            self.siam_model = None
            self.binary_classification_model = None

        def initialize(self, encoder=0):
            self.encoder = self.build_encoder(encoder=encoder)
            self.siam_model = self.build_siam_model()
            self.binary_classification_model = self.build_binary_classification_model()
            
        ##################################### Potential Encoders #####################################
            
            
        def encoder_recurrent_convolutional(self, inputs):   
        
            # Add a 1D convolutional layer to extract features from each context
            conv = Conv1D(filters=self.convolutional_filters, kernel_size=self.kernel_size, activation='relu')(inputs)

            # Add a max pooling layer to summarize the extracted features
            max_pooling = MaxPooling1D(pool_size=self.context_size)(conv)

            # Add a recurrent layer to capture temporal dependencies within each context
            lstm = LSTM(units=self.lstm_memory_units)(max_pooling)

            return lstm
        
        def embedded_encoder_recurrent_convolutional(self, inputs):   
            
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)
        
            # Add a 2D convolutional layer to extract features from each context
            conv = Conv2D(filters=self.convolutional_filters, kernel_size=(self.kernel_size, self.kernel_size), activation='relu')(embedded_inputs)

            # Add a max pooling layer to summarize the extracted features
            max_pooling = MaxPooling2D(pool_size=(self.context_size, 1))(conv)

            # Flatten the max_pooling output to feed it to the LSTM layer
            flattened = Reshape((-1, 32))(max_pooling)

            # Add a recurrent layer to capture temporal dependencies within each context
            lstm = LSTM(units=self.lstm_memory_units)(flattened)
            
            return lstm

        def transformer_encoder(self, inputs):
            
            # Add optional embedding layer
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)

            # Apply multi-head self-attention
            attn_output = Attention()([embedded_inputs, embedded_inputs, embedded_inputs])

            # Add a feed-forward layer
            ff_output = Dense(units=128, activation=self.activation_fn1)(attn_output)

            # Add a global average pooling layer to summarize the extracted features
            avg_pooling = GlobalAveragePooling2D()(ff_output)

            return avg_pooling

        def hierarchical_attention_encoder(self, inputs):
            
            # Add optional embedding layer
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)

            # Apply a bidirectional GRU to each context
            gru_output = TimeDistributed(Bidirectional(GRU(units=8, return_sequences=True)))(embedded_inputs)

            # Apply token-level attention
            token_attn_output = Attention()([gru_output, gru_output, gru_output])

            # Apply token-level global average pooling
            token_avg_pooling = TimeDistributed(GlobalAveragePooling1D())(token_attn_output)

            # Apply context-level attention
            context_attn_output = Attention()([token_avg_pooling, token_avg_pooling, token_avg_pooling])

            # Apply context-level global average pooling
            context_avg_pooling = GlobalAveragePooling1D()(context_attn_output)

            return context_avg_pooling
        
        def multi_head_self_attention_encoder(self, inputs):
            # Add optional embedding layer
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)

            # Apply multi-head self-attention to the input
            attention_output = Attention()([embedded_inputs, embedded_inputs, embedded_inputs])

            # Add a global average pooling layer to summarize the extracted features
            avg_pooling = GlobalAveragePooling2D()(attention_output)

            return avg_pooling

        def positional_pooling_encoder(self, inputs):
            # Add optional embedding layer
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)

            # Apply global max pooling to the input
            max_pooling = GlobalMaxPooling2D()(embedded_inputs)

            # Add a dense layer to capture the most significant features
            dense_output = Dense(units=128, activation=self.activation_fn1)(max_pooling)

            return dense_output
        
        def som_encoder(self, inputs):
            return SOMEncoderLayer(MAX_NODE_LOOKUP_NUM, self.embedding_dim, self.context_size, self.bag_size, 4, self.fixed_vector_size)(inputs)

        def capsule_encoder(self, inputs):
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)
            capsule_output = CapsuleEncoder()(embedded_inputs)
            return capsule_output

        def tcn_encoder(self, inputs):
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)
            tcn_output = TCNEncoder(filters=self.embedding_dim)(embedded_inputs)
            flattened_tcn_output = Flatten()(tcn_output)
            return flattened_tcn_output

        def attention_augmented_conv_encoder(self, inputs):

            # Add optional embedding layer
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)

            # Apply attention augmented convolutions
            conv = Conv2D(filters=self.embedding_dim, kernel_size=(self.kernel_size,self.kernel_size), padding="same")(embedded_inputs)
            attention = Conv2D(filters=self.embedding_dim, kernel_size=(self.kernel_size,self.kernel_size), padding="same", activation="sigmoid")(embedded_inputs)
            attended_conv = Multiply()([conv, attention])
            x = Add()([embedded_inputs, attended_conv])
            
            flattened = Flatten()(x)
            return flattened
        
        def bidirectional_lstm_encoder(self, inputs):

            # Add optional embedding layer
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)

            # Apply bidirectional LSTM layer
            reshaped_inputs = tf.keras.layers.Reshape((self.bag_size, self.context_size * self.embedding_dim))(embedded_inputs)

            # Apply bidirectional LSTM layer
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.lstm_memory_units, return_sequences=True))(reshaped_inputs)

            # Flatten the output
            flattened = Flatten()(x)
            return flattened
        
        def lstm_with_dense_layers_encoder(self, inputs):

            # Add optional embedding layer
            embedded_inputs = Embedding(input_dim=MAX_NODE_LOOKUP_NUM + 1, output_dim=self.embedding_dim)(inputs)
        
            flattened = Reshape((-1, 32))(embedded_inputs)

            # Apply LSTM layer followed by dense layers
            x = LSTM(units=self.lstm_memory_units, return_sequences=True)(flattened)
            x = Flatten()(x)
            x = Dense(units=self.lstm_memory_units, activation="relu")(x)

            return x
            
        
        def encoder_conv_attention(self, inputs):   
				
            conv = Conv1D(filters=self.convolutional_filters, kernel_size=self.kernel_size, activation='relu')(inputs)
            max_pooling = MaxPooling1D(pool_size=self.context_size)(conv)

            attention_output = Attention()([max_pooling, max_pooling])
            pooled_attention = GlobalAveragePooling1D()(attention_output)

            return pooled_attention

        def encoder_lstm_attention(self, inputs):
            conv = Conv1D(filters=self.convolutional_filters, kernel_size=self.kernel_size, activation='relu')(inputs)
            max_pooling = MaxPooling1D(pool_size=self.context_size)(conv)
            lstm = LSTM(units=self.lstm_memory_units, return_sequences=True)(max_pooling)

            attention_output = Attention()([lstm, lstm])
            pooled_attention = GlobalAveragePooling1D()(attention_output)

            return pooled_attention

        def simple_transformer_encoder(self, inputs):
            
            # Apply multi-head self-attention
            attn_output = Attention()([inputs, inputs, inputs])

            # Add a feed-forward layer
            ff_output = Dense(units=256, activation=self.activation_fn1)(attn_output)

            # Add a global average pooling layer to summarize the extracted features
            avg_pooling = GlobalAveragePooling1D()(ff_output)

            return avg_pooling


        
        ##################################### ################## #####################################
        
            
        def build_encoder(self, encoder=0):
            
            inputs = Input(shape=self.input_shape)

            masked_inputs = Masking()(inputs)
            
            
            ##########################################################

            if encoder == 0:
                encoded = self.encoder_recurrent_convolutional(masked_inputs)
            elif encoder == 1:
                encoded = self.embedded_encoder_recurrent_convolutional(masked_inputs)
            elif encoder == 2:
                encoded = self.transformer_encoder(masked_inputs)
            elif encoder == 3:
                encoded = self.hierarchical_attention_encoder(masked_inputs)
            elif encoder == 4:
                encoded = self.multi_head_self_attention_encoder(masked_inputs)
            elif encoder == 5:
                encoded = self.positional_pooling_encoder(masked_inputs)
            elif encoder == 6:
                encoded = self.som_encoder(masked_inputs)
            elif encoder == 7:
                encoded = self.capsule_encoder(masked_inputs)
            elif encoder == 8:
                encoded = self.tcn_encoder(masked_inputs)
            elif encoder == 9:
                encoded = self.attention_augmented_conv_encoder(masked_inputs)
            elif encoder == 10:
                encoded = self.bidirectional_lstm_encoder(masked_inputs)
            elif encoder == 11:
                encoded = self.lstm_with_dense_layers_encoder(masked_inputs)
            elif encoder == 12:
                encoded = self.encoder_conv_attention(masked_inputs)
            elif encoder == 13:
                encoded = self.encoder_lstm_attention(masked_inputs)
            elif encoder == 14:
                encoded = self.simple_transformer_encoder(masked_inputs)

            else:
                encoded = masked_inputs
            
            ##########################################################

            # Dropout layer to prevent overfitting
            dropout = Dropout(rate=self.rate)(encoded)

            # Fixed length vector representation
            outputs = Dense(units=self.fixed_vector_size)(dropout)

            # Define encoder model
            encoder = tf.keras.Model(inputs=inputs, outputs=outputs)
            return encoder
        
        def build_siam_model(self):
            
            # Create SimSiam model
            x1 = Input(shape=self.input_shape)
            x2 = x1

            # Define a lambda layer to shuffle the input tensors
            shuffle_layer = Lambda(lambda x: tf.random.shuffle(x))

            # Apply the shuffle layer to the input tensors
            x1_shuffled = shuffle_layer(x1)
            x2_shuffled = shuffle_layer(x2)
            
            # Encode the input twice using the same encoder
            z1 = self.encoder(x1_shuffled)
            z2 = self.encoder(x2_shuffled)
            
            # Predict a transformation of the first encoding
            p1 = Dense(units=self.fixed_vector_size, activation=self.activation_fn2)(z1)
            p2 = Dense(units=self.fixed_vector_size, activation=self.activation_fn2)(z2)
            
            #Loss function
            def D(p, z):
                z = tf.stop_gradient(z)
                p = tf.math.l2_normalize(p, axis=1)
                z = tf.math.l2_normalize(z, axis=1)
                return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))
            
            def siamese_loss(y_true, y_pred):
                p1, p2, z1, z2 = y_pred[:, 0:self.fixed_vector_size], y_pred[:, self.fixed_vector_size:self.fixed_vector_size * 2], y_pred[:, self.fixed_vector_size * 2:self.fixed_vector_size * 3], y_pred[:, self.fixed_vector_size * 3:]
                return D(p1, z2) / 2 + D(p2, z1) / 2

            # Concatenate the outputs
            concatenated_outputs = Concatenate(axis=-1)([p1, p2, z1, z2])

            # Define the model
            model = tf.keras.Model(inputs=x1, outputs=concatenated_outputs)

            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(self.base_lr * (self.siam_batch_size / self.steps_per_update)/256, self.unsupervised_epochs)

            # Define the optimizer with SGD, weight decay, and momentum
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)

            model = GradientAccumulateModel(accum_steps=self.steps_per_update, inputs=model.input, outputs=model.output)

            # Compile the model
            model.compile(optimizer=optimizer, loss=siamese_loss, run_eagerly=True)

            return model

        def build_binary_classification_model(self):
            
            # Create binary classification model
            name_input = Input(shape=(1,))
            timestamp_input = Input(shape=(1,))
            message_input = Input(shape=(3,))
            bag1_input = Input(shape=self.input_shape)
            bag2_input = Input(shape=self.input_shape)
            
            # Encode bag of contexts using the same encoder model
            encoded1 = self.encoder(bag1_input)
            encoded2 = self.encoder(bag2_input)
            # for layer in self.encoder.layers:
            #     layer.trainable = False
                
            name_reshaped = name_input#Reshape((1,))(name_input)
            timestamp_reshaped = timestamp_input#Reshape((1,))(timestamp_input)
            message_reshaped = message_input#Reshape((1,))(message_input)


            #Use attention between the 2 bags
            # Compute similarity matrix between bags
            context = Attention()([encoded1, encoded2])
            
            # merged = Concatenate()([encoded1, encoded2, name_reshaped, timestamp_reshaped, message_reshaped])
            merged = Concatenate()([context, name_reshaped, timestamp_reshaped, message_reshaped])
            
            # Binary classification output
            binary_classification = Dense(units=self.bag_size * 2, activation=self.activation_fn3)(merged)
            binary_classification = Dense(units=self.bag_size, activation=self.activation_fn3)(binary_classification)
            binary_classification = Dense(units=self.bag_size / 2, activation=self.activation_fn3)(binary_classification)
            binary_classification = Dense(units=self.bag_size / 4, activation=self.activation_fn3)(binary_classification)
            binary_classification = Dense(units=1, activation=self.activation_fn3)(binary_classification)
            
            # Define model
            model = tf.keras.Model(inputs=[name_input, timestamp_input, message_input, bag1_input, bag2_input], outputs=binary_classification)

            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(self.supervised_base_lr, self.supervised_epochs)
           
            # Compile model
            model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=self.loss_fn, metrics=['accuracy'])
            
            return model

        def fit_siam(self, X_train, epochs, verbose=0): 

            return self.siam_model.fit(X_train, X_train, epochs=epochs, batch_size=self.siam_batch_size, verbose=verbose, use_multiprocessing=True, callbacks=ClearMemory())

        def fit_siam_generator(self, generator, epochs, num_runs=4, run_epochs=8, verbose=0): 

            # Define a list to store the best weights obtained during training
            best_weights = None
            best_metric_value = float('inf')

            # Loop over each set of initial random weights
            for i in tqdm(range(num_runs), desc="Searching for a good initial randomization"):
                # Randomly initialize the weights of the model
                initial_weights = [np.random.normal(0, 0.1, w.shape) for w in self.siam_model.get_weights()]
                self.siam_model.set_weights(initial_weights)

                # Train the model for a fixed number of epochs
                history = self.siam_model.fit(generator, epochs=run_epochs, verbose=verbose, use_multiprocessing=True, callbacks=[ClearMemory()])

                # Retrieve the loss value from the last epoch
                metric_value = history.history['loss'][-1]
                if metric_value < best_metric_value:
                    # Update the best weights and metric value
                    best_weights = self.siam_model.get_weights()
                    best_metric_value = metric_value

            # Reset the weights of the model to the best weights obtained during training
            self.siam_model.set_weights(best_weights)

            # Train the model for the remaining epochs
            return self.siam_model.fit(generator, epochs=(epochs - run_epochs), verbose=verbose, use_multiprocessing=True, callbacks=ClearMemory())

        def fit_binary_classification(self, X_train, y_train, epochs, batch_size, verbose=0, validation_data=None):

            X_train_name = np.array([tup[0] for tup in X_train])
            X_train_timestamp = np.array([tup[1] for tup in X_train])
            X_train_message = np.array([tup[2] for tup in X_train])
            X_train_bag1 = np.array([tup[3] for tup in X_train])
            X_train_bag2 = np.array([tup[4] for tup in X_train])

            if(validation_data is not None):
                X_test = validation_data[0]
                y_test = validation_data[1]
                X_test_name = np.array([tup[0] for tup in X_test])
                X_test_timestamp = np.array([tup[1] for tup in X_test])
                X_test_message = np.array([tup[2] for tup in X_test])
                X_test_bag1 = np.array([tup[3] for tup in X_test])
                X_test_bag2 = np.array([tup[4] for tup in X_test])
                val_data = ([X_test_name, X_test_timestamp, X_test_message, X_test_bag1, X_test_bag2],y_test)
            else:
                val_data = None

            return self.binary_classification_model.fit(
                [X_train_name, X_train_timestamp, X_train_message, X_train_bag1, X_train_bag2],
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                validation_data=val_data,
                use_multiprocessing=True
            )

        def evaluate_binary_classification(self, X_test, y_test, verbose=0):
            X_test_name = np.array([tup[0] for tup in X_test])
            X_test_timestamp = np.array([tup[1] for tup in X_test])
            X_test_message = np.array([tup[2] for tup in X_test])
            X_test_bag1 = np.array([tup[3] for tup in X_test])
            X_test_bag2 = np.array([tup[4] for tup in X_test])

            # Evaluate the model on the test set
            return self.binary_classification_model.evaluate([X_test_name, X_test_timestamp, X_test_message, X_test_bag1, X_test_bag2], y_test, verbose=verbose)
    

    return CommitDiffModel
