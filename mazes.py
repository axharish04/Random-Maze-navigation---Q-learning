import gym
import gym_maze
import numpy as np
import keyboard  # Library for detecting keypress
import matplotlib.pyplot as plt
env=gym.make("maze-random-5x5-v0") # maze dimensions should be the same for length and breadth
def greedy_ep(Q_table, curr_q_coor, expl_prob):
    if np.random.uniform(0, 1) < expl_prob:
        return env.action_space.sample()  # Random exploration
    else:
        return int(np.argmax(Q_table[curr_q_coor]))  # Exploit

def softmax(Q_table, curr_q_coor, temperature):
    ac_values = Q_table[curr_q_coor] / temperature #takes action values(finds max then scales)
    max = np.max(ac_values)
    scaled_ac_values = ac_values - max  # Stability adjustment
    ac_prob = np.exp(scaled_ac_values)
    ac_prob /= np.sum(ac_prob)
    return np.random.choice(env.action_space.n, p=ac_prob)
#USeful when q table is large 
def q_learn(strategy, expl_parameters):
    #env = gym.make("maze-sample-10x10-v0")
    states = {}
    count = 0
    for i in range(10):  # Change according to the maze dimensions
        for j in range(10):
            states[i, j] = count
            count += 1
    n_actions = env.action_space.n

    Q_table = np.zeros((len(states), n_actions))

    n_episodes = 1000
    iter = 100
    gamma = 0.99
    initial_lr = 0.1
    lr_decay = 0.0001

    reward_episodic = []
    best_reward = float('-inf')

    for e in range(n_episodes):
        curr_state = env.reset()
        done = False
        tot_reward = 0
        lr = initial_lr / (1 + lr_decay * e)

        for i in range(iter):
            curr_x = int(curr_state[0])
            curr_y = int(curr_state[1])
            curr_q_coor = states[curr_x, curr_y]

            if strategy == "epsilon_greedy":
                action = greedy_ep(
                    Q_table, curr_q_coor, expl_prob=expl_parameters["expl_prob"]
                )
            elif strategy == "softmax":
                action = softmax(
                    Q_table, curr_q_coor, temperature=expl_parameters["temperature"]
                )
            else:
                raise ValueError("Invalid exploration strategy.")

            next_state, reward, done, _ = env.step(action)

            next_x = int(next_state[0])
            next_y = int(next_state[1])
            next_Q_table_coordinates = states[next_x, next_y]

            Q_table[curr_q_coor, action] = (1 - lr) * Q_table[curr_q_coor, action] + lr * (
                reward + gamma * max(Q_table[next_Q_table_coordinates, :])
            )

            tot_reward += reward
            env.render()

            if done:
                break

            if keyboard.is_pressed('z'):
                env.close()
                exit()

            curr_state = next_state

        if strategy == "epsilon_greedy":
            expl_parameters["expl_prob"] = max(
                expl_parameters["min_expl_prob"], expl_parameters["expl_prob"] * expl_parameters["exploration_decay"]
            )
        elif strategy == "softmax":
            expl_parameters["temperature"] *= expl_parameters["temperature_decay"]

        reward_episodic.append(tot_reward)

        if tot_reward > best_reward:
            best_reward = tot_reward

        print(f"Episode: {e+1}, Reward: {tot_reward}")

    env.close()
    print(f"Best reward obtained: {best_reward}")

    return reward_episodic, best_reward

expl_strat = [
    {
        "name": "epsilon_greedy",
        "expl_prob": 1.0,
        "exploration_decay": 0.001,
        "min_expl_prob": 0.1,
    },
    {"name": "softmax", "temperature": 1.0, "temperature_decay": 0.99},
]

results = {}  # Dictionary to store results of each strategy

for strategy in expl_strat:
    reward_episodic, best_reward = q_learn(strategy["name"], strategy)
    results[strategy["name"]] = {"rewards": reward_episodic, "best_reward": best_reward}

plt.figure(figsize=(10, 6))
for strategy, result in results.items():
    plt.plot(result['rewards'], label=strategy)
plt.title('Episodic Rewards over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()

# Print comparison between exploration strategies
print("\nComparison between exploration strategies:")
for strategy, result in results.items():
    print(f"{strategy}: Average Reward: {np.mean(result['rewards'])}, Best Reward: {result['best_reward']}")
