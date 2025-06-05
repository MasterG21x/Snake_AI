from typing import Deque, Tuple
import numpy as np
import tensorflow as tf
from model import QNetwork, QTrainer
from collections import deque
import random
class DQNAgent:
    """
    DQN Agent for training the Snake game:
    - Replay memory
    - Selects actions using epsilon-greedy policy
    - Trains via experience replay
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        max_memory=100_000,
        batch=1024,
        gamma=0.9,
        lr=1e-3,
        eps_max=1.0,
        eps_min=0.01,
        eps_decay=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=max_memory
        )
        self.batch = batch
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.model = QNetwork(state_size, action_size)
        self.trainer = QTrainer(self.model, lr, gamma)

    def act(self, state: np.ndarray) -> int:
        """
        Chooses an action using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        return int(tf.argmax(q[0]))

    def remember(self, *exp) -> None:
        """
        Stores experience tuple in memory.
        """
        self.memory.append(exp)

    def replay(self) -> float:
        """
        Trains the model on a random minibatch of experiences.
        Returns the training loss.
        """
        if len(self.memory) < self.batch:
            return 0.0
        mb = random.sample(self.memory, self.batch)
        s, a, r, s2, done = map(np.array, zip(*mb))
        loss = self.trainer.train_step(
            tf.convert_to_tensor(s, dtype=tf.float32),
            tf.convert_to_tensor(a, dtype=tf.int32),
            tf.convert_to_tensor(r, dtype=tf.float32),
            tf.convert_to_tensor(s2, dtype=tf.float32),
            tf.convert_to_tensor(done, dtype=tf.float32),
        )
        return float(loss)

    def decay_eps(self):
        """
        Decays epsilon after each episode.
        """
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

