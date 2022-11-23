from unityagents import UnityEnvironment
import numpy as np
import torch
from src.utils import print_device_info, print_env_info, get_env


def random_exploration(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    unity_env = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = unity_env.vector_observations[0]
    state_size = len(state)


    num_agents = len(unity_env.agents)

    unity_env = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = unity_env.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)

    while True:
        print(f"State: {states}")
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        print(f"Original actions: {actions}")
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        print(f"Clipped actions: {actions}")
        unity_env = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = unity_env.vector_observations         # get next state (for each agent)
        rewards = unity_env.rewards                         # get reward (for each agent)
        dones = unity_env.local_done                        # see if episode finished
        scores += unity_env.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_device_info()
    env = get_env(multi_agent=True, visual_mode=True)
    print_env_info(env)
    
    # ReacherBrain
    action_size = 4 # Vector Action space size (per agent): 4
    state_size = 33 # Vector Observation space size (per agent): 33
    random_exploration(env)


    env.close() # otherwise the terminal hangs


