import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.losses import huber_loss
import matplotlib.pyplot as plt
import Q_Network as qn
import Memory as mem
import Select_Action as sa
import Environment as env

NUM_EPISODES = 10000
MAX_STEPS = 200
GAMMA = 0.99
WARMUP = 10

LEARNING_RATE = 0.0001

E_START = 0.9
E_STOP = 0.01
E_DECAY_RATE = 0.001

MEMORY_SIZE = 10000
BATCH_SIZE = 16

Game_Name = "CartPole-v0"


# 環境の作成
env = gym.make(Game_Name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

main_qn = qn.Q_Network(state_size, action_size, LEARNING_RATE)
target_qn = qn.Q_Network(state_size, action_size, LEARNING_RATE)
memory = mem.Memory(MEMORY_SIZE)

# 学習の開始
state = env.reset()
state = np.reshape(state, [1, state_size])

total_step = 0
success_count = 0
fit_count = 0

loss = []
steps = []

for episode in range(1, NUM_EPISODES+1):
    step = 0

    target_qn.model.set_weights(main_qn.model.get_weights())

    # 1エピソードのループ
    for _ in range(1, MAX_STEPS+1):
        env.render()
        step += 1
        total_step += 1

        epsilon = E_STOP + (E_START - E_STOP)*np.exp(-E_DECAY_RATE*total_step)
        
        action = sa.select_action(epsilon, env, main_qn, state)
        
        next_state, _, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        if done:
            if step == 200:
                success_count += 1
                reward = 1
            else:
                success_count = 0
                reward = -1
            
            next_state = np.zeros(state.shape)

            if step > WARMUP:
                memory.add((state, action, reward, next_state))
        
        else:
            reward = 0

            if step > WARMUP:
                memory.add((state, action, reward, next_state))
            
            state = next_state
        
        if len(memory) >= BATCH_SIZE:
            inputs = np.zeros((BATCH_SIZE, 4))
            targets = np.zeros((BATCH_SIZE, 2))

            minibatch = memory.sample(BATCH_SIZE)

            for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                inputs[i] = state_b

                if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                    target = reward_b + GAMMA * np.amax(target_qn.model.predict(next_state_b)[0])
                else:
                    target = reward_b
                
                targets[i] = main_qn.model.predict(state_b)
                targets[i][action_b] = target

            fit = main_qn.model.fit(inputs, targets, epochs=1, verbose=0, callbacks=[])
            fit_count += 1
            if fit_count % 10 == 0:
                loss.append(fit.history["loss"])

        if done:
            steps.append(step)
            main_qn.model.save("best_model.h5")
            break

    print("エピソード: {}, ステップ数: {}, epsilon: {:.4f}".format(episode, step, epsilon))

    if success_count >= 10:
        break
    
    # 環境のリセット
    state = env.reset()
    state = np.reshape(state, [1, state_size])

plt.subplot(1,2,1)
plt.plot(loss, "r")
plt.title("loss")

plt.subplot(1,2,2)
plt.plot(steps, "b")
plt.title("steps")

plt.savefig("figure_loss_step.png")

plt.show()