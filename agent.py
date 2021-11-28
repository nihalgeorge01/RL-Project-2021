from numpy.core.fromnumeric import argmax
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

        # KBC targets
        self.kbc_r0 = -1
        self.kbc_b = -1
        self.kbc_pe = -1
        self.kbc_ph = -1
        self.kbc_ge = -1
        self.kbc_gh = -1
        self.kbc_kc = 0
        self.kbc_step = 0
        self.kbc_n = self.config[0]
        self.kbc_pe_ct = [0 for i in range(self.kbc_n)]
        self.kbc_ph_ct = [0 for i in range(self.kbc_n)]
        self.kbc_train_ep = 0
        self.kbc_pe_ep = 0
        self.kbc_ph_ep = 0
        self.kbc_test_ep = 0
        self.kbc_easy = True

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
        
        elif self.env_name[:3] == 'kbc':
            self.kbc_train_ep += 1
            self.action = 1
            self.kbc_step = 0
            if self.env_name[3] == 'c' and not self.kbc_easy:
                self.action = 2

        if self.env_name == 'taxi':
            self.action = np.argmax(self.q_table[obs])

        self.obs_prev = obs
        return self.action
        #return 1
        #raise NotImplementedError
        #return action

    def acro_reward(self, obs, done):
        m = 1.0
        g = 9.8
        L = 1.0
        Icm = m*L*L/12
        Iend = Icm*4
        Lb2 = L/2
        pi = 3.14159265

        # Hyperparams: relative importance of each variable
        pe_scale = 1
        ke_scale = 1
        ke1_scale = 1
        ke2_scale = 1
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

        if done:
            return 1
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
            a_reward = self.acro_reward(obs, done)
            error = self.obs_prev*(a_reward + (1-done)*self.alpha*np.max(np.matmul(obs, self.weights)) - q_val_prev)
            self.weights[:, self.action] += self.beta*error[0]

            q_vals = np.matmul(obs, self.weights)
            #print(self.weights)
            if np.random.uniform() > self.eps:
                self.action = np.argmax(q_vals)
            else:
                self.action = np.random.choice(self.config[1])
        
        elif self.env_name[:3] == 'kbc':
            # hidden: reward - r0, b; probab - pe, ge
            
            self.kbc_step += 1
            # Finding r0 - Get 1 question correct, then leave to get r0
            if self.kbc_r0 == -1:  # Not yet found r0 
                if not done:        # Episode not ended => We got Q0 correct => We can pull out to get r0
                    self.action = 0
                elif self.kbc_step == 1:    # Done and only 1 step => We got Q0 wrong, have to start again
                    self.action = 1
                else:   # Done in 2 steps => We got Q0 correct, pulled out, got r0. Now store this value
                    self.action = 1
                    self.kbc_r0 = reward
            
            # Finding b - Get 2 questions correct, then leave to get r0*b, divide by r0 to get b
            elif self.kbc_b == -1:
                if not done:
                    if self.kbc_step == 1:  # Got Q0 correct, now have to attempt Q1
                        self.action = 1
                    else:       # Not done and more than 1 step (=> 2 steps) => Got Q0 and Q1 correct, now pull out to get r0*b
                        self.action = 0
                else:
                    if self.kbc_step in [1,2]:  # Done in 1 or 2 steps => Got Q0 wrong or got Q1 wrong, start again
                        self.action = 1
                    else:   # Done in 3 steps => Got Q0 and Q1 correct, then pulled out => got reward, now compute b and store it
                        self.kbc_b = float(reward)/self.kbc_r0

            # Finding px, gx, kc - Simulate as many paths as possible, find empirical probabs
            else:       # Just store counts of how many times a state was reached
                if self.env_name[3] == 'a': # KBC A - Finding pe, ge - Simulate as many paths as possible, find empirical probabs
                    if not done:    # Keep attempting questions
                        self.kbc_pe_ct[self.kbc_step-1] += 1 # Increment counter of correct attempts
                        self.action = 1
                    else:   # We lost at Q_step-1
                        self.action = 1
                        self.kbc_pe_ep += 1 # Increment episode count
                
                elif self.env_name[3] == 'b':   # KBC B - Finding pe, ge, kc - Simulate as many paths as possible, never pulling out, find empirical probabs.
                    if not done:    # Keep attempting questions
                        self.kbc_pe_ct[self.kbc_step-1] += 1 # Increment counter of correct attempts
                        self.action = 1
                    else:   # We lost at Q_step-1
                        self.action = 1
                        self.kbc_pe_ep += 1
                        if reward != 0:     # We crossed a checkpoint somewhere, note reward = r0*b^(kc-1), Use maths to get kc
                            self.kbc_kc = int(np.log(float(reward)/self.kbc_r0)/np.log(self.kbc_b)) + 1
                
                else:   # KBC C -Finding pe, ph, ge, gh - Simulate as many paths as possible, find empirical probabs. Alternate between episodes of all easy and all hard questions
                    if not done:   # Don't change self.action, just do the same
                        if self.kbc_easy:
                            self.kbc_pe_ct[self.kbc_step-1] += 1
                        else:
                            self.kbc_ph_ct[self.kbc_step-1] += 1
                    else:   # We lost at Q_step-1, switch to other question mode
                        if self.kbc_easy:
                            self.kbc_pe_ep += 1
                            self.kbc_easy = False
                            self.action = 2
                        else:
                            self.kbc_ph_ep += 1
                            self.kbc_easy = True
                            self.action = 1
                    
        
        elif self.env_name == 'taxi':
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
        
        elif self.env_name[:3] == 'kbc':
            # If first test episode, set variables and DP tables
            if self.kbc_test_ep == 0:
                # If any params are unset (at -1), put some default value
                if self.kbc_r0 == -1:
                    self.kbc_r0 = 1000
                if self.kbc_b == -1:
                    self.kbc_b = 2
                # kbc_kc is already set correctly to 0 if not KBC B
                
                # Build rewards and probab arrays
                self.kbc_rewards = [0] + [self.kbc_r0 * (self.kbc_b ** i) for i in range(self.kbc_n)]
                self.kbc_rewards[-1] = self.kbc_r0 * (self.kbc_b ** (self.kbc_n-2)) # Env has a mistake, setter's side issue
                # TODO : Least squares based estimate of px, gx
                # self.kbc_pe_log_emp = []
                # for val in self.kbc_pe_ct:
                #     if val>0:
                #         self.kbc_pe_log_emp.append(np.log(val/self.kbc_pe_ep))
                #     else:
                #         break
                
                # # Least square fit to get ln(ge) and ln(pe) + 0.5ln(ge)

                # # Finding hard q params for KBC C
                # if self.env_name[3] == 'c':
                #     self.kbc_ph_log_emp = []
                #     for val in self.kbc_ph_ct:
                #         if val>0:
                #             self.kbc_ph_log_emp.append(np.log(val/self.kbc_ph_ep))
                #         else:
                #             break
                #     # Least square fit to get ln(gh) and ln(ph) + 0.5ln(gh)
                # END TODO

                self.kbc_pe = self.kbc_pe_ct[0]/self.kbc_pe_ep
                self.kbc_ge = (self.kbc_pe_ct[1]/self.kbc_pe_ep)/(self.kbc_pe ** 2)
                self.kbc_pe_ns_probs = [self.kbc_pe * (self.kbc_ge ** i) for i in range(self.kbc_n)]

                if self.env_name[3] == 'c':
                    self.kbc_ph = self.kbc_ph_ct[0]/self.kbc_ph_ep
                    self.kbc_gh = (self.kbc_ph_ct[1]/self.kbc_ph_ep)/(self.kbc_ph ** 2)
                    self.kbc_ph_ns_probs = [self.kbc_ph * (self.kbc_gh ** i) for i in range(self.kbc_n)]
                
                # Build DP table, get optimal policy
                self.kbc_dp = [-1 for i in range(self.kbc_n)]
                self.kbc_pol = [0 for i in range(self.kbc_n)]

                self.kbc_dp[-1] = self.kbc_rewards[-1]
                self.kbc_pol[-1] = 0

                for i in range(self.kbc_n-2, -1, -1):
                    if self.env_name[3] == 'c':
                        choices = [self.kbc_rewards[i], self.kbc_pe_ns_probs[i]*self.kbc_dp[i+1], self.kbc_ph_ns_probs[i]*self.kbc_dp[i+1] + (1-self.kbc_ph_ns_probs[i])*(self.kbc_rewards[i]/2.0)]
                    elif i > self.kbc_kc:
                        choices = [self.kbc_rewards[i], self.kbc_pe_ns_probs[i]*self.kbc_dp[i+1] + (1-self.kbc_pe_ns_probs[i])*self.kbc_rewards[self.kbc_kc]]
                    else:
                        choices = [self.kbc_rewards[i], self.kbc_pe_ns_probs[i]*self.kbc_dp[i+1]]
                    
                    self.kbc_dp[i] = max(choices)
                    self.kbc_pol[i] = argmax(choices)

            # Take optimal action given obs and DP table
            self.kbc_step = 0
            action = self.kbc_pol[0]
        
        elif self.env_name == 'taxi':
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
        
        elif self.env_name[:3] == 'kbc':
            # Use previously computed optimal policy
            self.kbc_step += 1
            if done:
                self.kbc_test_ep += 1
                action = 0
            else:
                action = self.kbc_pol[self.kbc_step]
        
        elif self.env_name == 'taxi':
            action = np.argmax(self.q_table[obs])
        #raise NotImplementedError
        return action
