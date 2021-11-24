  Assume that you are a contestant at a quiz program, where you have an opportunity to win substantial amount of money by answering a series of questions. 
  These are the rules of the constant

  1. At each stage of the contest, you are allowed to chose between 2 actions:
      - Pull out and receive reward
      - Answer a question
  2. A reward r_i is received at the end of the contest. "i" is the max question reached before termination. The reward at all preceeding stages is 0.
  3. The outcome of a question "i" is determined by a probability "p_i". Note that "p_i > p_{i+1}". The probabilty p_i is not visible to the agent.
  4. There are a maximum of N questions(N=16). 
  6. There are a maximum of 16 questions. You receive full reward if the 16th question is reached. 
  7. If you are terminated after answering a question wrong, your reward is 0. 

Actions:
```
    0 - Pull out and receive reward
    1 - Answer the next question
```

Observations:

- A list of outcomes of all previous questions. 1 indicates answered correctly, 0 indicates answered wrong, and "" indicates unanswered. Example:

[1,1,1,1,1,0,"","","","","","","","","",""]

Rewards:

- Default reward of 0 at each step
- Reward at last step is 0 if terminated on an easy question
- Reward at last step is reward_i if pulled out after ith question
- Reward at last step is reward_N if Nth question is reached

Termination criteria:
  
- Nth question is reached (N=16)
- Last action is 0
- Last question is wrongly answered

The following are the environments you will be working on:

## KBC A:
Assume an environment with the above rules, model an agent that maximises the outcome. 

## KBC B:
Assume there is a checkpoint question K, after which you are guarenteed a reward of r_k, irrespective of what happens later. Model an agent under this condition. 

## KBC C:
Additional rules:

- At each stage of the question you now have an opportunity to choose between an easy and a hard question. 
- Answering an easy question wrong terminates with a 0 reward, like KBC A. 
- Answer a hard question wrong terminates with a reward of r_i/2. 
- Note that p_i_{easy} > p_i_{hard}
- Model an RL agent under these conditions. 


Actions:
 ```
  0 - Pull out and receive reward
  1 - Answer an easy question
  2 - Answer a hard question
```
