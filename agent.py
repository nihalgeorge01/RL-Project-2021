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

        if self.env_name == 'taxi':
            self.q_table = np.zeros((self.config[0],self.config[1]))
        
        if self.env_name == 'acrobot':
            np.random.seed(10)
            self.weights = np.clip(np.random.normal(0,np.sqrt(2/self.config[0]),size=(self.config[0],self.config[1])), -1, 1) #He initialisation

        self.beta = self.config[2]
        self.alpha = self.config[3]
        self.beta_inv = int(1/self.beta)
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
        if self.env_name == 'acrobot':
            obs = np.array(obs).reshape(1,-1)
            q_vals = np.matmul(obs, self.weights)
            self.action = np.argmax(q_vals)
            pass
        if self.env_name == 'taxi':
            self.action = np.argmax(self.q_table[obs])

        self.obs_prev = obs
        return self.action
        #return 1
        #raise NotImplementedError
        #return action

    def acro_reward(self, obs):
        m = 1.0
        g = 9.8
        L = 1.0
        Icm = m*L*L/12
        Iend = Icm*4
        Lb2 = L/2
        pi = 3.14159265

        # Hyperparams: relative importance of each variable
        pe_scale = 5
        ke_scale = 5
        ke1_scale = 1
        ke2_scale = 0.2
        te_scale = 5

        # c = cos(), s = sin()
        c1 = obs[0,0]
        s1 = obs[0,1]
        c2 = obs[0,2]
        s2 = obs[0,3]
        c12 = c1*c2 - s1*s2     #cos(theta1+theta2)
        s12 = s1*c2 + c1*s2     
        v1 = obs[0,4]
        v2 = obs[0,5]
        v1_max = 4*pi
        v2_max = 9*pi

        # Potential energy
        pe1 = -m*g*(Lb2)*c1
        pe2 = -m*g*(L*c1 + Lb2*c12)
        pe0 = 2*m*g*L
        pe = (pe1+pe2 - pe0)/pe0
        
        # Kinetic Energy
        ke1 = 0.5*Iend*v1*v1
        ke1_max = 0.5*Iend*v1_max*v1_max
        ke2 = 0.5*(m*L*L + Iend + 2*m*L*Lb2*c1)*v1*v1 + 0.5*Iend*v2*v2 + (Iend + m*L*Lb2*c2)*v1*v2
        ke2_max = 0.5*(m*L*L + Iend + 2*m*L*Lb2*1.0)*v1_max*v1_max + 0.5*Iend*v2_max*v2_max + (Iend + m*L*Lb2*1.0)*v1_max*v2_max
        ke_max = ke1_scale*ke1_max + ke2_scale*ke2_max
        ke = ((ke1_scale*ke1 + ke2_scale*ke2) - ke_max)/ke_max

        te_max = pe0 + ke_max
        te = te_scale *(ke_scale*ke + pe_scale*pe - te_max)/te_max

        return te

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
        if self.env_name == 'acrobot':
            obs = np.array(obs).reshape(1,-1)
            q_val_prev = np.matmul(self.obs_prev, self.weights[:,self.action].reshape(-1,1))[0]
            a_reward = self.acro_reward(obs)
            error = self.obs_prev*(a_reward + (1-done)*self.alpha*np.max(np.matmul(obs, self.weights)) - q_val_prev)
            self.weights[:, self.action] += self.beta*error[0]

            q_vals = np.matmul(obs, self.weights)
            #print(self.weights)

        if self.env_name == 'taxi':
            self.q_table[self.obs_prev][self.action] += self.beta*(reward + (1-done)*self.alpha*np.max(self.q_table[obs]) - \
                                                                                self.q_table[self.obs_prev][self.action])
            q_vals = self.q_table[obs]
        
        if np.random.uniform() > self.eps:
            self.action = np.argmax(q_vals)
        else:
            self.action = np.random.choice(self.config[1])

        self.eps = max(0.1, self.eps*self.eps_decay)
        self.obs_prev = obs
        
        # # beta decay
        # self.beta_inv += 0.0001
        # self.beta = max(0.0001, 1.0/self.beta_inv)

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
        if self.env_name == 'acrobot':
            obs = np.array(obs).reshape(1,-1)
            q_vals = np.matmul(obs, self.weights)
            action = np.argmax(q_vals)

        if self.env_name == 'taxi':
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
        if self.env_name == 'acrobot':
            obs = np.array(obs).reshape(1,-1)
            q_vals = np.matmul(obs, self.weights)
            action = np.argmax(q_vals)

        if self.env_name == 'taxi':
            action = np.argmax(self.q_table[obs])
        #raise NotImplementedError
        return action
