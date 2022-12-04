import os
import torch as T
import torch.nn.functional as F
import numpy as np
from src.networks import ActorNetwork, CriticNetwork
from src.replay_buffer import ReplayBuffer
from src.noise import OUActionNoise

class DDPGAgent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                buffer_size=1000000, fc1_dims=400, fc2_dims=300, batch_size=64, algo=None, env_name=None,
                checkpoint_dir='tmp/ddpg'):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.env_name = env_name
        self.algo = algo
        self.checkpoint_dir = checkpoint_dir

        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f"{self.env_name}_{self.algo}_actor",
                                    checkpoint_dir=self.checkpoint_dir)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f"{self.env_name}_{self.algo}_critic",
                                    checkpoint_dir=self.checkpoint_dir)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                            n_actions=n_actions, name=f"{self.env_name}_{self.algo}_target_actor",
                            checkpoint_dir=self.checkpoint_dir)
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name=f"{self.env_name}_{self.algo}_target_critic",
                                checkpoint_dir=self.checkpoint_dir)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # evaluation mode: deactivate layernorm, batchnorm, dropout...
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        
        self.actor.train()
        # detach it from the graph
        return mu_prime.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()

        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        target_actions = self.target_actor.forward(next_states)
        next_critic_value = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        next_critic_value[dones] = 0.0
        next_critic_value = next_critic_value.view(-1) # flatten

        target = rewards + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        #self.update_network_parameters()
        self.soft_update(self.actor, self.target_actor, self.tau)
        self.soft_update(self.critic, self.target_critic, self.tau)

        
    def soft_update(self, online_model, target_model, tau):
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_state_dict[name].clone()
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False) # if you want to use batchnorm
        #self.target_actor.load_state_dict(actor_state_dict, strict=False) # if you want to use batchnorm