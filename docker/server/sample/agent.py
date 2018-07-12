class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, observation, reward, done):
        return self.action_space.sample()