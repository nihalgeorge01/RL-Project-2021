<h1 align="center"><a href="https://www.aicrowd.com/challenges/rl-project-2021">IITM RL Final Project 2021</a> - Starter Kit</h1>


This is the starter kit for the [IITM RL Final Project](https://www.aicrowd.com/challenges/rl-project-2021) hosted on [AIcrowd](https://www.aicrowd.com). Clone the repository to compete now!
The following are the environments you will be working on. Click on the respective links for the description. 

- [Acrobot-v1](docs/acrobot.md)
- [Taxi-v3](docs/taxi.md)
- [KBC](docs/kbc.md)

This repository contains:

- **Documentation** on how to submit your models to the leaderboard.
- Information on **evaluating your agents locally**, **baselines** and some best practises to have hassle free submissions.
- **Starter code** for you to get started!

[IMPORTANT - Accept the rules before you submit](https://www.aicrowd.com/challenges/rl-project-2021/challenge_rules)



# Table of contents

- [📚 Competition procedure](#-competition-procedure)
- [💪 Getting started](#-getting-started)
- [🛠 Preparing your submission](#-preparing-your-submission)
  * [Write your agents](#write-your-agents)
  * [Evaluate your agents locally](#evaluate-your-agents-locally)
- [📨 Submission](#-submission)
  * [Repository Structure](#repository-structure)
  * [Runtime configuration](#runtime-configuration)
  * [🚀 Submitting to AIcrowd](#-submitting-to-aicrowd)
    + [`aicrowd.json`](#aicrowdjson)
    + [Configuring the submission repository](#configuring-the-submission-repository)
    + [Pushing the code to AIcrowd](#pushing-the-code-to-aicrowd)
- [📝 Submission checklist](#-submission-checklist)
- [📎 Important links](#-important-links)
- [✨ Contributors](#-contributors)

# 📚 Competition procedure

**The following is a high level description of how this process works.**

![](https://i.imgur.com/xzQkwKV.jpg)

1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/rl-project-2021).
2. **Clone** this repo and start developing your solution.
3. **Design and build** your agents that can compete in the environments and implement an agent class as described in [writing your agents](#write-your-agents) section.
4. [**Submit**](#-submission) your agents to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation. [[Refer this for detailed instructions]](#-submission).

# 💪 Getting started

> We recommend using `python 3.8`. If you are using Miniconda/Anaconda, you can install it using `conda install python=3.8`. Recommentded pip version is > 21.1.1

Clone the starter kit repository and install the dependencies.

```bash
git clone https://gitlab.aicrowd.com/siddhartha/iitm-rl-project-2021-starter-kit
cd iitm-rl-project-2021-starter-kit
pip install -U -r requirements.txt

```
   
# 🛠 Preparing your submission

## Write your agents

You need to implement the [`Agent`](agent.py#L10) class from [`agent.py`](agent.py). Check out the file for descriptions of the functions that need to be implemented. 

You could specify env specific config options in [`config.py`](config.py)

## Evaluate your agents locally

We have provided [`run.py`](run.py) to test your agents locally.

To run the evaluation locally, run the following commands.

```bash
ENV_NAME="acrobot" python run.py
ENV_NAME="taxi" python run.py
ENV_NAME="kbca" python run.py
ENV_NAME="kbcb" python run.py
ENV_NAME="kbcc" python run.py

```

**Note:** Please note that the changes you make to any file inside [`run.py`](run.py) will be dropped during evaluation. 

# 📨 Submission

## Repository structure

**File/Directory** | **Description**
--- | ---
[`agent.py`](agent.py) | File for implementing the Agent class. Your code goes in this file.
[`config.py`](config.py) | File containing the configuration options for  Agent class.
[`run.py`](run.py) | File used to evaluate the agent class. Use this file to test your agents locally
[`requirements.txt`](requirements.txt) | File containing the list of python packages you want to install for the submission to run. Refer [runtime configuration](#runtime-configuration) for more information.
[`apt.txt`](apt.txt) | File containing the list of packages you want to install for submission to run. Refer [runtime configuration](#runtime-configuration) for more information.
[`gym-bellman`](gym-bellman/) | Folder containing the gym environment for the Bellman's DP problem
[`docs`](docs/) | Folder containing the descriptions for the environments in the challenge
[`aicrowd.json`](docs/) | Submission configuration


## Runtime configuration

You can specify the list of python packages needed for your code to run in your [`requirements.txt`](requirements.txt) file. We will install the packages using `pip install` command.

You can also specify the OS packages needed using [`apt.txt`](apt.txt) file. We install these packages using `apt-get install` command.


## 🚀 Submitting to AIcrowd

### **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).


### `aicrowd.json`

Your repository should have an `aicrowd.json` file with following fields:

```json
{
    "challenge_id" : "rl-project-2021",
    "authors" : ["Your Name"],
    "description" : "Brief description for your submission"
}
```

This file is used to identify your submission as a part of the challenge. You must use the `challenge_id` as specified above.

### Configuring the submission repository

```bash
git remote add aicrowd git@gitlab.aicrowd.com:<username>/iitm-rl-project-2021-starter-kit.git
```

**Note:** This needs to be done only once. This configuration will be saved in your repository for future use.

### Pushing the code to AIcrowd

Create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**  
Then you can add the correct git remote, and finally submit by doing :

```
# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at :
[gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/iitm-rl-project-2021-starter-kit/issues](https://gitlab.aicrowd.com//<YOUR_AICROWD_USER_NAME>/iitm-rl-project-2021-starter-kit/issues)

**NOTE**: Remember to update your username instead of `<YOUR_AICROWD_USER_NAME>` above :wink:




# 📝 Submission checklist

- [x] **Accept the challenge rules**. You can do this by going to the [challenge overview page](https://www.aicrowd.com/challenges/rl-project-2021) and clicking the "Participate" button. You only need to do this once.
- [x] **Add your agent code** that implements the `Agent` class from `agent.py`.
- [x] **Evaluate your agents locally** to know that they work as expected. Ensure your agent is verified by running `python run.py`.
- [x] **Update runtime configuration** using `requirements.txt`, `apt.txt` as necessary. Please make sure that you specified the same package versions that you use locally on your machine.

# 📎 Important links

- 💪 Challenge information
   * [Challenge page](https://www.aicrowd.com/challenges/rl-project-2021)
   * [Leaderboard](https://www.aicrowd.com/challenges/rl-project-2021/leaderboards)
 - 🗣 Community
    * [Discussion forum](https://discourse.aicrowd.com/c/rl-project-2021/)
    

# ✨ Contributors

- [Siddhartha Laghuvarapu](https://www.aicrowd.com/participants/siddhartha)
- RL TAs

**Best of Luck** 🎉 
