from hyperopt import fmin, tpe, hp
import aicrowd_gym
from agent import Agent
import numpy as np
from tqdm import tqdm

def objective(args):
    agent = Agent('taxi', args[0], args[1])
    
    for i in range(1500):
        obs = env.reset()
        action = agent.register_reset_train(obs)
        done = False
        while not done:
            obs, reward, done, info = env.step(action)
            action = agent.compute_action_train(obs, reward, done, info)
    
    rewards_history = []
    for i in range(100):
        rewards = 0
        obs = env.reset()
        action = agent.register_reset_test(obs)
        done = False
        while not done:
            obs, reward, done, info = env.step(action)
            action = agent.compute_action_test(obs, reward, done, info)
            rewards += reward
        rewards_history+=[rewards]
    
    return -np.mean(rewards_history)

env = aicrowd_gym.make("Taxi-v3")
space = [hp.uniform('alpha', 0, 0.99), 
         hp.uniform('beta', 0, 1),]

best = fmin(objective, space, algo = tpe.suggest, max_evals=100)
print(best)





