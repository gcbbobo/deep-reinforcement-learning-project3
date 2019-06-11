import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 2**10        # minibatch size
GAMMA = 0.99            # discount factor

# the large soft updating rate of target parameters matters much
TAU = 1e-3              # for soft update of target parameters,from 1e-3 to 3e-1
OPPO_TAU = 1e-1            # To turn down the function 'learning from the opponent', place 0 here         
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.000  # L2 weight decay

# 4 is bad in this case
net_update_every = 5   # the network weights in both actor and critic cases are updated every # time steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params for single agent
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actorA_local = Actor(state_size, action_size, random_seed).to(device)
        self.actorA_target = Actor(state_size, action_size, random_seed).to(device)
        self.actorB_local = Actor(state_size, action_size, random_seed).to(device)
        self.actorB_target = Actor(state_size, action_size, random_seed).to(device)
        
        self.actorA_optimizer = optim.Adam(self.actorA_local.parameters(), lr=LR_ACTOR)
        self.actorB_optimizer = optim.Adam(self.actorB_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        # input: states and actions of all the agents-->reward only from individual itsellf
        self.criticA_local = Critic(2*state_size, 2*action_size, random_seed).to(device)
        self.criticA_target = Critic(2*state_size, 2*action_size, random_seed).to(device)
        self.criticA_optimizer = optim.Adam(self.criticA_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.criticB_local = Critic(2*state_size, 2*action_size, random_seed).to(device)
        self.criticB_target = Critic(2*state_size, 2*action_size, random_seed).to(device)
        self.criticB_optimizer = optim.Adam(self.criticB_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)        

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        "memoryA for agentA, memoryB for agentB"
        self.memoryA = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memoryB = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    
    def step(self, all_states, all_actions, reward, all_next_states,all_done,t,scores,agent_index):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        
        '''
        params in memory:
        ------------------------------------------------------
        state: observations from all agents
        action: actions taken by all agents
        reward: reward for individual agent
        next_state: next_observations from all agents
        done: dones from all agents
        '''
        
        if agent_index == 0:
            self.memoryA.add(all_states, all_actions, reward, all_next_states, all_done)
            
            # Learn, if enough samples are available in memory
            # CAUTION: python use 'and' instead of '&&'
            if len(self.memoryA) > BATCH_SIZE and t%net_update_every==0:
                experiences = self.memoryA.sample()
                self.learn(experiences, GAMMA,agent_index)
        else:
            self.memoryB.add(all_states, all_actions, reward, all_next_states, all_done)  
            
            if len(self.memoryB) > BATCH_SIZE and t%net_update_every==0:
                experiences = self.memoryB.sample()
                self.learn(experiences, GAMMA,agent_index)
        
        
        if scores[0]<=scores[1]:
            self.opponent_learn(self.actorB_local,self.actorA_local,OPPO_TAU)
        else:
            self.opponent_learn(self.actorA_local,self.actorB_local,OPPO_TAU)
        
        

    def act(self,state, agent_index, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        # CAUTION: Here the input state should be the PRIVATE observation of individual agents
        state = torch.from_numpy(state).float().to(device)
        
        if agent_index == 0: # agent A
            self.actorA_local.eval()
            with torch.no_grad():
                action = self.actorA_local(state).cpu().data.numpy()
            self.actorA_local.train()
            
        else: # agent B
            self.actorB_local.eval()
            with torch.no_grad():
                action = self.actorB_local(state).cpu().data.numpy()
            self.actorB_local.train()
            
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    # As described by others, ddpg algorithm is very sensitive to update frequency.
    def learn(self,experiences, gamma,agent_index):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        all_states, all_actions, rewards, all_next_states, all_dones = experiences
        
        "---------------agent A---------------------"
        if agent_index == 0:
            states = all_states[0::2]
            actions = all_actions[0::2]
            next_states = all_next_states[0::2]
            dones = all_dones[0::2]
            all_actions_next = self.actorA_target(all_next_states)
            
            # ---------------------------- update critic ---------------------------- #
            "use global info for critic"
            # Get predicted next-state actions and Q values from target models
            # Same as act, learn serves for individual agent        
            
            # each set of all_next_states, all_actions_next, the batch_size = n
            # the Q_targets_next should also be size = n
            Q_targets_next = self.criticA_target(all_next_states, all_actions_next)
            
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - all_dones))
            
            # Compute critic loss
            Q_expected = self.criticA_local(all_states, all_actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            
            # Minimize the loss
            self.criticA_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.criticA_local.parameters(),1)
            self.criticA_optimizer.step()
            
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            "local info only for actor"
            actions_pred = self.actorA_local(states)
            actor_loss = -self.criticA_local(states, actions_pred).mean()

            # Minimize the loss
            self.actorA_optimizer.zero_grad()
            actor_loss.backward()
            self.actorA_optimizer.step()
            
            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.criticA_local, self.criticA_target, TAU)
            self.soft_update(self.actorB_local, self.actorB_target, TAU) 
            
            "------------------Agent B-----------------"    
        else:
            states = all_states[1::2]
            actions = all_actions[1::2]
            next_states = all_next_states[1::2]
            dones = all_dones[1::2]   
            all_actions_next = self.actorB_target(all_next_states) 
            
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            # Same as act, learn serves for individual agent        
        
            Q_targets_next = self.criticB_target(all_next_states, all_actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - all_dones))
            # Compute critic loss
            Q_expected = self.criticB_local(all_states, all_actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            
            # Minimize the loss
            self.criticB_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.criticB_local.parameters(),1)
            self.criticB_optimizer.step()
            
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actorB_local(states)
            actor_loss = -self.criticB_local(states, actions_pred).mean()

            # Minimize the loss
            self.actorB_optimizer.zero_grad()
            actor_loss.backward()
            self.actorB_optimizer.step()


        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.criticB_local, self.criticB_target, TAU)
        self.soft_update(self.actorB_local, self.actorB_target, TAU) 
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def opponent_learn(self, strong_model, weak_model, oppo_tau):
        # update the target one / weak one
        self.soft_update(strong_model,weak_model,oppo_tau)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        
        # random.random is uniform distribution
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) +self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)