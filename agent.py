from config import *
import time
import numpy as np
"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""


class Agent:
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        self.q_table = np.zeros((self.config[0],self.config[1]))
        self.beta = self.config[2]
        self.alpha = self.config[3]
        self.eps = 1
        self.eps_decay = 0.99
        pass

    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        self.action = np.argmax(self.q_table[obs])
        self.obs_prev = obs
        return self.action
        #return 1
        #raise NotImplementedError
        #return action

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        self.q_table[self.obs_prev][self.action] += self.beta*(reward + (1-done)*self.alpha*np.max(self.q_table[obs]) - \
                                                                            self.q_table[self.obs_prev][self.action])
        
        if np.random.uniform() > self.eps:
            self.action = np.argmax(self.q_table[obs])
        else:
            self.action = np.random.choice(self.config[1])

        self.eps = max(0.1, self.eps*self.eps_decay)
        self.obs_prev = obs

        return self.action
        #raise NotImplementedError
        #return action

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        action = np.argmax(self.q_table[obs])
        #raise NotImplementedError
        return action

    def compute_action_test(self, obs, reward, done, info):
        """
        Use this function in the test phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        action = np.argmax(self.q_table[obs])
        #raise NotImplementedError
        return action
