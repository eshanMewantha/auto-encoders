from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

# Load data
print('Loading data')
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("\nError: Data file missing. "
          "\nDownload the 'creditcard.csv' dataset from https://www.kaggle.com/dalpozz/creditcardfraud/data "
          "\nPlace it in the same directory as this python file before executing the program again."
          )
    exit()

print('Processing data')
# Anomalies
anomalies = df[df['Class'] == 1]

# Remove label column
df = df[df.columns.difference(['Class'])]
anomalies = anomalies[anomalies.columns.difference(['Class'])]

# Standardize data
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(np_scaled)
np_scaled = min_max_scaler.fit_transform(anomalies)
anomalies = pd.DataFrame(anomalies)

# Training Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 256

display_step = 100

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features
num_input = 30  # credit card data input (30 columns (excluding column 'Class'))

# Input
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_prediction = decoder_op

# Actual expected output = input
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_prediction, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training
    for i in range(1, num_steps + 1):
        batch_x = df.ix[np.random.choice(df.index, batch_size)]

        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})

        if i % display_step == 0 or i == 1:
            print('Step %i: Mini batch Loss: %f' % (i, l))

    # Testing
    anomaly_transaction_error = 0
    for i in range(len(anomalies)):
        g = sess.run(decoder_op, feed_dict={X: anomalies.iloc[i].values.reshape(1, 30)})
        anomaly_transaction_error += np.sum(anomalies.iloc[i].values - g)

    normal_transaction_error = 0
    for i in range(len(df.head(len(anomalies)))):
        g = sess.run(decoder_op, feed_dict={X: df.iloc[i].values.reshape(1, 30)})
        normal_transaction_error += np.sum(df.iloc[i].values - g)

    print('\n Anomaly transaction average error = ' + str(anomaly_transaction_error / len(anomalies)))
    print('\n Normal transaction average error = ' + str(normal_transaction_error / len(anomalies)))
