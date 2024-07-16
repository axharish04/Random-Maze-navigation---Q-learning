# Random-Maze-navigation---Q-learning
Q-Learning for Maze Solving with OpenAI Gym
This repository contains Python code to solve a maze environment using Q-learning, implemented with different exploration and learning rate strategies. The maze environment is provided by OpenAI Gym.

Components Used:

Python Libraries:
gym: OpenAI Gym library for defining and interacting with reinforcement learning environments.
gym_maze: Provides maze environments compatible with OpenAI Gym.
numpy: For efficient numerical operations, used to manage Q-tables and state representations.

Maze Environment:
Taken from https://github.com/MattChanTK/gym-maze

Parameters:

Gamma (Discount Factor): Set to 0.99, influencing the importance of future rewards in Q-learning updates.
Exploration Strategies:

Two strategies are implemented:
Epsilon-Greedy: Balances exploration and exploitation by choosing random actions with a decreasing probability (epsilon) and exploiting learned knowledge otherwise.
Softmax: Uses the softmax function to probabilistically select actions based on Q-values, adjusting exploration through a temperature parameter.
Learning Rate Strategies:

Two strategies are implemented:
Constant: Uses a fixed learning rate throughout training.
Exponential Decay: Decreases the learning rate exponentially over time, potentially allowing for finer adjustments in learning as the agent gains more experience.




How to Use:
Installation:

Clone the repository and install dependencies using pip install -r requirements.txt.



