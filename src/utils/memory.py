from collections import deque
import random

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def push(self, transition):
        processed_transition = self.process(transition)
        self.memory.append(processed_transition)
        if len(self.memory) > self.max_size:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def process(self, transition):
        # This method should be overridden by subclasses to provide specific processing
        raise NotImplementedError

class ShortTermMemory(Memory):
    def __init__(self, max_size):
        super(ShortTermMemory, self).__init__(max_size)
        self.memory = deque(maxlen=max_size)

    def process(self, transition):
        # Here you could add code to process the transition in a way that's specific to short-term memory
        pass

class LongTermMemory(Memory):
    def __init__(self, max_size):
        super(LongTermMemory, self).__init__(max_size)
        self.memory = deque(maxlen=max_size)

    def process(self, transition):
        # Here you could add code to process the transition in a way that's specific to long-term memory
        pass


