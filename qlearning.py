import numpy as np
from tour_env import *
from dqn import *
import matplotlib.pyplot as plt

def Epsilon_Policy(Q, state, action_num, epsilon):
    rand_a = np.random.random()
    if rand_a < epsilon:
        a = np.random.randint(0, action_num)
    else:
        a = np.argmax(Q[state, :])
    return a


# build environment
env = tour_env()
print("number of actions: ", env.nA)
print("number of states:  ", env.nS)

Network = DeepQNetwork(n_actions=env.nA,
                       learning_rate=0.1,
                       reward_decay=0.95,
                       e_greedy=0.95,
                       e_greedy_increment=0.05,
                       replace_target_iter=300,
                       memory_size=1000,
                       batch_size=256)

count = 0
for episode in range(9000):
    state = env.reset()

    while True:
        action = Network.choose_action(state)

        s_prime, R, done, info = env.step(action)
        if not info:
            Network.store_transition(state, action, R, s_prime)
            count += 1

        if count > 300 and count % 10 == 0:
            Network.learn()

        state = s_prime

        if done:
            break

Network.plot_cost()

# # print(env.history)
# plt.plot(Reward_200episode, lw=0.5)
# plt.ylabel('total Reward')
# plt.xlabel('every 200 Episode')
# plt.savefig('QLearning.png')
# # plt.show()
# plt.clf()

# state = env.reset()
# Reward = 0
# count += 1
# # print('init state: ', state)
# while True:
#     a = Epsilon_Policy(Q, state, env.nA, 0)
#     # print('state:  ',state)
#     s_prime, R, done, info = env.step(a)
#     # print('choose next state:  ', s_prime)
#     if info:
#         # print('info == True, not update')
#         pass
#     else:
#         # print('Update Q matrix, reward = ', R)
#         Reward += R
#         print('Reward: ',R)
#         Q[state, a] += 0.1 * (R + 0.95 * np.max(Q[s_prime, :]) - Q[state, a])
#         state = s_prime
#     if done:
#         print('Episode Reward',Reward)
#         print(env.history)
#         Reward_episode.append(Reward)
#         break
