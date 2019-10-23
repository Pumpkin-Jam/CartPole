from collections import deque
import numpy as np

class Memory:
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)