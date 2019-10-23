import numpy as np

def select_action(epsilon, env, main_qn, state):
    if epsilon > np.random.rand():
        action = env.action_space.sample()
    else:
        action = np.argmax(main_qn.model.predict(state)[0])

    return action