import abc
import utils 
import environment

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, id, env_params, spt=None):
        self.env_params = env_params
        self.spt = spt
        self.location = (0,0)
        self.prev_location = (0,0)
        self.food = 0
        self.active = 0
        self.id = id
        self.bfs_active = 0
    
    def get_state(self):
        return self.location, self.food

    def set_spt(self, spt):
        self.spt = spt
    
    @abc.abstractmethod
    def get_action(self, observation, valid_movements):
        # Returns movement, pheromone
        pass 

    def __str__(self):
        return ''+ str(self.location) + ' ' + str(self.active) + ' ' + str(self.food)

class RandomAgent(Agent):
    def get_action(self, observation, valid_movements):
        return random.choice(valid_movements), random.random()
    
    
class SwarmAgent(Agent):
    def __init__(self, id, env_params, spt=None):
        super().__init__(id, env_params, spt)
        self.orientation = None
        self.obs_rad = self.env_params['observation_radius']
        self.obs_window = self.env_params['observation_radius']*2+1
        
    def flatten(self, obs_r, obs_c):
        return self.obs_window * r + c
    
    def unflatten(self, obs_pos):
        return (obs_pos // self.obs_window, obs_pos % self.obs_window)
        
    def get_action(self, observation, valid_movements, fwd_only = 1):
        agent_grid, static_grid, dynamic_grid = observation
        d = dynamic_grid.copy()
        last_action = tuple(np.array(self.location) - np.array(self.prev_location))
        if self.food:
            # choose direction with BFS
            lay_pheromone = self.env_params['pheromone']['step_if_food']
            good_actions = self.spt[self.location[0]][self.location[1]]
            next_movement = good_actions[np.random.randint(len(good_actions))]
        
        else:
            lay_pheromone = self.env_params['pheromone']['step']
            # move forward, probs weighted by pheromone strength
            if last_action == (-1,0):
                d[self.obs_rad + 1 - fwd_only:, :] = 0

            elif last_action == (0,1):
                d[:, :self.obs_rad + fwd_only] = 0

            elif last_action == (1,0):
                d[:self.obs_rad + fwd_only, :] = 0

            elif last_action == (0,-1):
                d[:, self.obs_rad + 1 - fwd_only:] = 0
        
            dflat = d.flatten()
            if np.sum(dflat) == 0:
                probs = [1/len(dflat)] * len(dflat)
            else:
                probs = dflat/np.sum(dflat)

            # choose random target location in fwd direction, weighted by pheromone
            best_loc = np.random.choice(np.arange(len(dflat)), p = probs)
            best_loc = self.unflatten(best_loc)
                
            good_actions = []     
            # if best action row > agent row in observation window
            if best_loc[0] > self.obs_rad:
                good_actions.append((1,0))
            if best_loc[0] < self.obs_rad:
                good_actions.append((-1,0))
            if best_loc[1] > self.obs_rad:
                good_actions.append((0,1))
            if best_loc[1] < self.obs_rad:
                good_actions.append((0,-1))

            assert(len(valid_movements) > 0)
            # if list empty take random action from all actions not blocked by obstacles
            good_actions = list(set(good_actions).intersection(set(valid_movements)))
            if len(good_actions) > 0:
                next_movement = good_actions[np.random.randint(len(good_actions))]
            else:
                next_movement = valid_movements[np.random.randint(len(valid_movements))]

        return next_movement, lay_pheromone

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

class DQNAgent(Agent):
    def __init__(self, id, env_params):
        self.env_params = self.env_params
        obs_window = self.env_params['observation_radius']*2+1
        self.input_shape = (obs_window, obs_window)

        self.resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
        
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.screen_height, self.screen_width = 100, 200

        self.n_actions = 5
        self.policy_net = DQN(screen_height, screen_width, n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(policy_net.parameters())
        self.memory = ReplayMemory(10000)    
        self.steps_done = 0


        super(self)
    
    def prepare_observation(self, observation, training=True):
        # Add color channels
        agent_static_grid, dynamic_grid = utils.process_grids(observation, visual=False)
        appended_grid = np.append(agent_static_grid, dynamic_grid, axis=0)
        # Change H,W,C --> C,H,W
        appended_grid = appended_grid.transpose((2, 0, 1))
        
        if training:
            # Add a batch dimension --> B, C, H, W
            appended_grid = torch.from_numpy(appended_grid)
            return self.resize(appended_grid).unsqueeze(0)
        else:
            return self.resize(appended_grid)
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=DEVICE, dtype=torch.long)

    def optimize_model():
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def train(self, env_params, grid, agents, num_episodes=50):
        env = environment.Environment(env_params, grid)

        for i_episode in range(num_episodes):
            # Initialize the environment and spawn queue
            env.reset(grid)
            env.initialize_spawn_queue(agents)
            old_observations = [self.prepare_observation(env.get_empty_observation()) for agent in agents]
            observations = None
            
            for step in range(env_params['steps']):
                # Spawn agents if necessary
                env.spawn_agents(agents)

                # Get observation for each agent and calculate an action for each agent
                observations = [self.prepare_observation(obs) for obs in env.update_observation(agents)]
                actions = []
                for agent_idx in range(len(agents)):
                    agent = agents[agent_idx]
                    state = observations[agent_idx]-old_observations[agent_idx]
                    action = self.select_action(state)
                    actions.append(action)
                actions_tensor = torch.tensor(actions, device=DEVICE)
                
                # Step actions in the environment
                environment_actions = [(env.environment_actions[action], env_params['pheromone']['step']) if agents[agent_idx].food else (env.environment_actions[action], env_params['pheromone']['step']) for agent_idx in range(len(agents))]
                rewards = env.step(agents, actions)
                rewards_tensor = torch.tensor(rewards, device=DEVICE)

                # Get new observation for each agent
                next_observations = [self.prepare_observation(obs) for obs in env.update_observation(agents)]

                # Store the transitions in memory
                for agent_idx in range(len(agents)):
                    state = observations[agent_idx]-old_observations[agent_idx]
                    next_state = next_observations[agent_idx]-observations[agent_idx]
                    self.memory.push(state, actions_tensor[agent_idx], next_state, rewards_tensor[agent_idx])

                # Save observations
                old_observations = observations

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


    def get_action(self, observation, valid_movements):
        


        


