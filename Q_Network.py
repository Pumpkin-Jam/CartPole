import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.losses import huber_loss

class Q_Network:
    def __init__(self, state_size, action_size, learning_rate):
        self.model = Sequential([
            Dense(16, activation="relu", input_dim=state_size),
            Dense(16, activation="relu"),
            Dense(16, activation="relu"),
            Dense(action_size, activation="linear")
        ])

        self.model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))