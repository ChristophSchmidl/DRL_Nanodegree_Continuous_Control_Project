from unityagents import UnityEnvironment
import torch
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import time
from src.utils import print_device_info, print_env_info, get_env
import src.agents as Agents
from src.noise import NoNoise


def parse_bool(s: str) -> bool:
    '''
    Small helper function for argparse to convert string to boolean.
    Note: Argparse has this weird behavior when using bool as type,
    it does not behave as expected. You can also avoid using this helper 
    function by just using action='store_true' with choices=["true", "false"]? because then you do not have
    to specify any argument for that specific flag, e.g.:
    - parser.add_argument('-enable_stuff', action='store_true')
    - parser.add_argument('-disable_stuff', action='store_false', dest='enable_stuff')
    - python -m src.main -enable_stuff
    OR:
    - parser.add_argument('-enable_stuff', type=parse_bool, default=False)
    - python -m src.main -enable_stuff True
    '''
    try:
        return {'true': True, 'false': False}[s.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Expected true/false, git: {s}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ActorCritic methods - Continuous Control Project'
    )

    # the hyphen makes the argument optional
    parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1. Default is 0.')
    parser.add_argument('-episodes', type=int, default=1000, help='Number of games/episodes to play. Default is 1000.')
    parser.add_argument('-alpha', type=float, default=0.0001, help='Learning rate alpha for the actor network. Default is 0.0001.')
    parser.add_argument('-beta', type=float, default=0.0001, help='Learning rate beta for the critic network. Default is 0.0001.')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation. Default is 0.99.')
    parser.add_argument('-tau', type=float, default=0.001, help='Update network parameters. Default is 0.001.')
    parser.add_argument('-algo', type=str, default='DDPGAgent',
                    help='You can use the following algorithms: DDPGAgent. Default is DDPGAgent.')
    parser.add_argument('-buffer_size', type=int, default=1000000, help='Maximum size of memory/replay buffer. Default is 1000000.')
    parser.add_argument('-batch_size', type=int, default=128, help='Batch size for training. Default is 128.')
    parser.add_argument('-load_checkpoint', action='store_true',
                        help='Load model checkpoint/weights. Default is False.')
    parser.add_argument('-model_path', type=str, default='data/',
                        help='Path for model saving/loading. Default is data/')
    parser.add_argument('-plot_path', type=str, default='plots/',
                        help='Path for saving plots. Default is plots/')
    parser.add_argument('-use_eval_mode', action='store_true',
                        help='Evaluate the agent. Deterministic behavior. Default is False.')
    parser.add_argument('-use_multiagent_env', action='store_true',
                        help='Using the multi agent environment version. Default is False.')
    parser.add_argument('-use_visual_env', action='store_true',
                        help='Using the visual environment. Default is False.')
    parser.add_argument('-save_plot', action='store_true',
                        help='Save plot of eval or/and training phase. Default is False.')

    # TODO: add more arguments
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    ##########################################
    #             Set correct GPU            #
    ##########################################

    # set GPU (if you have multiple GPUs)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print_device_info()

    ##########################################
    #       Load correct environment         #
    ##########################################

    env = get_env(multi_agent=args.use_multiagent_env, visual_mode=args.use_visual_env)
    print_env_info(env)
    #env.close()

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    unity_env = env.reset(train_mode=True)[brain_name] # True means that the environment speed is faster (use False for visualiation)
    action_size = brain.vector_action_space_size
    state = unity_env.vector_observations[0]
    state_size = len(state)
    num_agents = len(unity_env.agents)

    ##########################################
    #       Training/Evaluation loop
    ##########################################
    best_score = -np.inf
    n_steps = 0
    episode_rewards, steps_array = [], []

    # Neat trick without using if/else/switch: get me the correct agent/algorithm
    conrete_agent = getattr(Agents, args.algo)


    '''
    Note: We diverge from the hyperparameters of the DDPG paper "Continuous control with deep reinforcement learning" that
    are mentioned in section 7 "Experiment details" of the supplementary information.
    We use 128 and 128 neurons/units instead of 400 and 300 units for the first and second layer 
    given that we have a different evnironment.
    '''
    agent = conrete_agent(
                  alpha=args.alpha,
                  beta=args.beta,  
                  gamma=args.gamma,
                  tau=args.tau,
                  fc1_dims=128, 
                  fc2_dims=128,
                  #input_dims=env.observation_space.shape,
                  input_dims=state_size,
                  #n_actions=env.action_space.n,
                  n_actions=action_size,
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  checkpoint_dir=args.model_path,
                  algo=args.algo,
                  #env_name=args.env
                  env_name='Reacher'
    )

    # If in evaluation mode
    if args.use_eval_mode:
        print("Evaluating agent...")
        # This leads to a deterministic behavior without exploration
        #agent.epsilon = 0.0
        #agent.epsilon_min = 0.0
        #agent.epsilon_dec = 0.0
        agent.noise = NoNoise()
        args.load_checkpoint = True


    if args.load_checkpoint:
        agent.load_models()

    plot_filename = agent.algo + '_' + agent.env_name + '_lr' + str(agent.alpha) + '_' + str(agent.beta) + '_' + \
                str(args.episodes) + 'episodes'
    figure_file = args.plot_path + plot_filename + '.png'

    start_time = time.time()
    solution_txt =""

    for i in range(args.episodes):
        done = False
        score = 0

        obs = unity_env.vector_observations[0]
        agent.reset_noise()
        while not done:
            action = agent.choose_action(obs)
            unity_env = env.step(action)[brain_name] 
            next_obs = unity_env.vector_observations[0]   # get the next state
            reward = unity_env.rewards[0]                   # get the reward
            done = unity_env.local_done[0]  

            score += reward
        
            if not args.load_checkpoint:
                agent.store_transition(obs, action, reward, next_obs, int(done))
                agent.learn()
            
            obs = next_obs
            n_steps += 1

        episode_rewards.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(episode_rewards[-100:])
        print(f"Episode: {i}, Score: {score:.2f}, Average score: {avg_score:.2f}, Steps: {n_steps}")

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score
        
        if avg_score >= 30 and not args.use_eval_mode:
            solution_txt = f"Solved in {i} episodes with an average reward score of {avg_score:.2f} of the last 100 episodes"
            print(solution_txt)
            break

    end_time = (time.time() - start_time)/60
    print(f"\nTotal training time = {end_time:.1f} minutes")


    np.save("data/episode_rewards.npy", episode_rewards) # TODO: Only save when necessary (in training?)

    # plot the scores
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111)
    #ax = fig.add_subplot(111)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)

    if args.use_eval_mode:
        plt.title(f"Continuous control project: {args.algo} - Evaluation") 
    else:
        plt.title(f"Continuous control project: {args.algo} - Training") 

    plt.ylabel('Rewards')
    new_line = '\n'
    plt.xlabel(f"Episodes{new_line}{solution_txt}")
    plt.grid(True)

    if args.save_plot:
        if not os.path.exists(args.plot_path):
            os.makedirs(args.plot_path)
        if args.use_eval_mode:
            plt.savefig(f"{args.plot_path}/Continuous_control_project_{args.algo}_eval.png")
        else:
            plt.savefig(f"{args.plot_path}/Continuous_control_project_{args.algo}_train.png")
    plt.show()
    
    env.close()