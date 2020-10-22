import numpy as np
import sys
from gym.envs.toy_text import discrete
import torch.nn.functional as F
import torch

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridworldEnv(discrete.DiscreteEnv): #
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 5x5 grid looks as follows:

    T   o   o  o   o
    o  -1   x  o   o
    o   o   o  o   o
    o   o   o  -1  o
    o   o   o  T   o

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[5,5], regular_reward=0.0, penalty=-1.0, final_reward=2.0):
        """
        This initializes the vars that we'll be using.
        :param shape: Shape of our grid world.
        :param regular_reward: The reward our agent receives if it arrives at a regular, non-terminal state.
        :param penalty: The penalty our agent recieves when it arrives in a bad state.
        :param final_reward: The terminal reward our agent recieves.
        """
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        # Grid World Structure.
        self.shape = shape
        self.nS, self.nA = np.prod(shape), 4
        self.starting_state = 12
        self.next_state = None
        self.terminal_states = [0, self.nS-1]
        self.penalty_states = [6, 19]

        # Keep track of where we are.
        self.current_state = self.starting_state

        # Rewards distribution
        self.penalty = penalty
        self.final_reward = final_reward
        self.regular_reward = regular_reward

        self.MAX_Y, self.MAX_X = shape[0], shape[1]

        self.P = {}
        self.grid = np.arange(self.nS).reshape(self.shape)
        self.it = np.nditer(self.grid, flags=['multi_index'])

        # Initial state distribution is uniform
        isd = np.ones(self.nS) / self.nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        # self.P = P

        super(GridworldEnv, self).__init__(self.nS, self.nA, self.P, isd)

    # New function! Encode state as one-hot torch tensor.
    def _get_obs(self, state):
        state, nS = torch.LongTensor([state]), np.prod(self.shape)
        state = F.one_hot(input=state, num_classes=nS).float().squeeze(dim=0).numpy().tolist()
        return state

    def reset(self):
        return self._reset()

    def _reset(self):
        # Reset to starting start.
        self.current_state = self.starting_state
        return self._get_obs(self.current_state)

    def step(self, action):
        return self._step(action)

    # def is_done(self):
    #     decision = self.current_state in self.terminal_states
    #     return decision

    def _step(self, action):
        assert self.action_space.contains(action)

        self.current_state = self.starting_state if self.next_state is None else self.next_state
        # Get coordinates of where we are in the grid given our current state.
        y, x = np.argwhere(self.grid==self.current_state)[0]

        self.P[self.current_state] = {a : [] for a in range(self.nA)}


        is_done = lambda s: s in self.terminal_states

        # Changes made here to rewards distribution.
        # There are three possible rewards: terminal state reward, penalty state reward, and regular state reward
        done = is_done(self.current_state)
        if done:
            reward = self.final_reward
        elif self.current_state in self.penalty_states:
            reward = self.penalty
        else:
            reward = self.regular_reward

        # We've reached the terminal state, so let's reset our self.next_state to zero.
        if done:
            self.next_state = None
        # Not a terminal state
        else:
            # Figure out agent's future states.
            ns_up = self.current_state if y == 0 else self.current_state - self.MAX_X
            ns_right = self.current_state if x == (self.MAX_X - 1) else self.current_state + 1
            ns_down = self.current_state if y == (self.MAX_Y - 1) else self.current_state + self.MAX_X
            ns_left = self.current_state if x == 0 else self.current_state - 1
            potential_future_states = [ns_up, ns_right, ns_down, ns_left]

            self.next_state = potential_future_states[action]

        # Return current state info.
        return self._get_obs(self.current_state), reward, done, {}


    # Code that I don't think we need anymore.
    # def _render(self, mode='human', close=False):
    #     if close:
    #         return
    #
    #     outfile = StringIO() if mode == 'ansi' else sys.stdout
    #
    #     grid = np.arange(self.nS).reshape(self.shape)
    #     it = np.nditer(grid, flags=['multi_index'])
    #     while not it.finished:
    #         s = it.iterindex
    #         y, x = it.multi_index
    #
    #         if self.s == s:
    #             output = " x "
    #         elif s == 0 or s == self.nS - 1:
    #             output = " T "
    #         else:
    #             output = " o "
    #
    #         if x == 0:
    #             output = output.lstrip()
    #         if x == self.shape[1] - 1:
    #             output = output.rstrip()
    #
    #         outfile.write(output)
    #
    #         if x == self.shape[1] - 1:
    #             outfile.write("\n")
    #
    #         it.iternext()



        # while not it.finished:
        #     s = it.iterindex
        #     y, x = it.multi_index
        #
        #     P[s] = {a : [] for a in range(nA)}
        #
        #     is_done = lambda s: s == 0 or s == (nS - 1)
        #     # Changes made here to rewards distribution
        #     reward = self.final_reward if is_done(s) else self.penalty
        #
        #     # We're stuck in a terminal state
        #     if is_done(s):
        #         P[s][UP] = [(1.0, s, reward, True)]
        #         P[s][RIGHT] = [(1.0, s, reward, True)]
        #         P[s][DOWN] = [(1.0, s, reward, True)]
        #         P[s][LEFT] = [(1.0, s, reward, True)]
        #     # Not a terminal state
        #     else:
        #         ns_up = s if y == 0 else s - MAX_X
        #         ns_right = s if x == (MAX_X - 1) else s + 1
        #         ns_down = s if y == (MAX_Y - 1) else s + MAX_X
        #         ns_left = s if x == 0 else s - 1
        #         P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
        #         P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
        #         P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
        #         P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            # it.iternext()

if __name__ == "__main__":
    env = GridworldEnv()
    state = env.reset()
    action_sequence = [3,3,0,0,0]
    print("State", state)
    for a in action_sequence:
        state, reward, is_done, _ = env.step(a)
        print("Action", a, "State", state, "reward", reward, "is done", is_done)
