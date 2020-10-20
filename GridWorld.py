

class GridWorld:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    #

    # Make step through environment
    def step(self):
        next_state, reward, done, _ = None, None, None, None
