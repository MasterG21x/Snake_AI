import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense

# ────────────────────────────────────────────────────────────────────────
#  Q-NETWORK
# ────────────────────────────────────────────────────────────────────────
class QNetwork(tf.keras.Model):
    """
    Simple fully connected neural network:
    - Input: 11-dimensional state vector
    - Hidden layers: 256 → 256
    - Output: Q-values for each of the 3 actions
    """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.d2 = tf.keras.layers.Dense(256, activation="relu")
        self.out = tf.keras.layers.Dense(action_size)

    def call(self, x, training=False):
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)


class QTrainer:
    """
    Handles training of the Q-Network using Mean Squared Error loss.
    """
    def __init__(self, model: QNetwork, lr=1e-3, gamma=0.9):
        self.model = model
        self.gamma = gamma
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step(self, s, a, r, s2, done):
        """
        Performs a single training step using a batch of experiences.
        Computes target Q-values, loss, and applies gradients.
        """
        q_next = tf.reduce_max(self.model(s2), axis=1)
        target = r + self.gamma * q_next * (1.0-done)

        with tf.GradientTape() as tape:
            q_vals = self.model(s)
            q_sel = tf.reduce_sum(q_vals * tf.one_hot(a, q_vals.shape[1]), axis=1)
            loss = self.loss_fn(target, q_sel)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

