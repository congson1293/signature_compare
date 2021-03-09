"""
Title: Metric learning for image similarity search
Author: [Mat Kelcey](https://twitter.com/mat_kelcey)
Date created: 2020/06/05
Last modified: 2020/06/09
Description: Example of using similarity metric learning on CIFAR-10 images.
"""
"""
## Overview

Metric learning aims to train models that can embed inputs into a high-dimensional space
such that "similar" inputs, as defined by the training scheme, are located close to each
other. These models once trained can produce embeddings for downstream systems where such
similarity is useful; examples include as a ranking signal for search or as a form of
pretrained embedding model for another supervised problem.

For a more detailed overview of metric learning see:

* [What is metric learning?](http://contrib.scikit-learn.org/metric-learn/introduction.html)
* ["Using crossentropy for metric learning" tutorial](https://www.youtube.com/watch?v=Jb4Ewl5RzkI)
"""

"""
## Setup
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib

X = joblib.load('data/X.pkl')
y = joblib.load('data/y.pkl')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

"""
To get a sense of the dataset we can visualise a grid of 25 random examples.

"""

height = 67
width = 200


"""
## Embedding model

We define a custom model with a `train_step` that first embeds both anchors and positives
and then uses their pairwise dot products as logits for a softmax.
"""


class EmbeddingModel(keras.Model):
    def train_step(self, data):
        X, y = data[0][0], data[0][1]
        anchors, positives = tf.expand_dims(X[0][0], axis=0), tf.expand_dims(X[0][1], axis=0)

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True) # 1x8
            positive_embeddings = self(positives, training=True) # 9x8

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            ) # 10x10
            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            loss = self.compiled_loss(tf.expand_dims(y, axis=0), similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(y, similarities)
        return {m.name: m.result() for m in self.metrics}


"""
Next we describe the architecture that maps from an image to an embedding. This model
simply consists of a sequence of 2d convolutions followed by global pooling with a final
linear projection to an embedding space. As is common in metric learning we normalise the
embeddings so that we can use simple dot products to measure similarity. For simplicity
this model is intentionally small.
"""

inputs = layers.Input(shape=(height, width, 3))
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x)
x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
embeddings = layers.Dense(units=16, activation=None)(x)
embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model = EmbeddingModel(inputs, embeddings)

"""
Finally we run the training. On a Google Colab GPU instance this takes about a minute.
"""

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.mean_squared_error,
)

history = model.fit((X_train, y_train), epochs=100, batch_size=1)

# plt.plot(history.history["loss"])
# plt.show()

"""
## Testing

We can review the quality of this model by applying it to the test set and considering
near neighbours in the embedding space.

First we embed the test set and calculate all near neighbours. Recall that since the
embeddings are unit length we can calculate cosine similarity via dot products.
"""
y_pred = []
for i, x in enumerate(X_test):
    embeddings = model.predict(x)
    anchor, positives = embeddings[0], embeddings[1:]
    sim = np.inner(anchor, positives)[0]
    if sim >= 0.5:
        y_pred.append(1)
    else: y_pred.append(0)

print(f'acc = {accuracy_score(y_test, y_pred)}')