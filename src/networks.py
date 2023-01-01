import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np    


##########################################
#                  DDPG                  #
##########################################

'''
Taken from the DDPG paper: 

"The final layer weights and biases of both the actor and critic
were initialized from a uniform distribution [-3x10^{-3}, 3x10^{-3}] and
[-3x10^{-4}, 3x10^{-4}] for the low dimensional and pixel cases respectively. 
This was to ensure the initial outputs for the policy and value estimates were near zero. 
The other layers were initialized from uniform distributions [-\frac{1}{\sqrt{f}}, \frac{1}{\sqrt{f}}]
where f is the fan-in of the layer."

Note: The term "fan-in" and "fan-out" is usually used in the context of digital electronics and logic gates.

- Fan-in: The number of inputs of the gate / The maximum number of inputs that a logic gate can accept 
(in our case: number of layer inputs)

- Fan-out: The number of outputs of the gate / The maximum number of inputs (load) that can be 
connected to the output of a gate without degrading the normal operation.
'''

def init_layer_uniformly_(layer, min_value=None, max_value=None):
    '''
    Initialize weights and biases of the layer. This is necessary to 
    mitigate the problem of disappearing gradients caused by the form of many activation functions.

    https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
    https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
    
    Note: layer.weight.data is a PyTorch Tensor
    '''
    if min_value is not None and max_value is not None:
        layer.weight.data.uniform_(min_value, max_value)
        layer.bias.data.uniform_(min_value, max_value)
    else:
        # This is basically Xavier init after Xavier Glorot
        # See: "Understanding the difficulty of training deep feedforward neural networks"
        # And: https://stats.stackexchange.com/questions/326710/why-is-weight-initialized-as-1-sqrt-of-hidden-nodes-in-neural-networks
        input_dim = layer.weight.data.size()[0]
        val = 1./np.sqrt(input_dim)
        layer.weight.data.uniform_(-val, val) # in-place
        layer.bias.data.uniform_(-val, val) # in-place

class ActorNetwork(nn.Module):
    '''
    Actor (Policy) model (called \mu(s|\theta^{\mu}) in the paper) that maps states to actions
    '''
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions,
                name, checkpoint_dir="tmp/ddpg"):
        super().__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # This is not batch norm but layer norm. It normalizes inputs but
        # does not depend on the batch size
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # self.bn1 = nn.BatchNorm1d(self.fc1_dims) # This would be BatchNorm
        # self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions) # Final layer "mu". Later wrapped with tanh [-1,1]

        self.init_weights_and_biases_uniformly()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') 

        self.to(self.device)

    def init_weights_and_biases_uniformly(self):
        init_layer_uniformly_(self.fc1)
        init_layer_uniformly_(self.fc2)
        init_layer_uniformly_(self.mu, min_value=-0.003, max_value=0.003)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x) # after batchnorm seems better

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # tanh is bounded by -1 and +1
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        if os.path.isfile(self.checkpoint_file):
            print("Loading model from file: ", self.checkpoint_file)
            # map_location is required to ensure that a model that is trained on GPU can be run on CPU
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
        else:
            print(f"File not found: {self.checkpoint_file}. Continue training from scratch.")

class CriticNetwork(nn.Module):
    '''
    Critic (Evaluation) model (called Q(s,a|\theta^Q) in the paper) that maps states, actions to Q values
    '''
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                name, checkpoint_dir='tmp/ddpg'):
        super().__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims) # LayerNorm is not the same as batchNorm!
        self.bn2 = nn.LayerNorm(self.fc2_dims) # LayerNorm is not the same as batchNorm!

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        self.init_weights_and_biases_uniformly()

        # Note: In the paper they state:
        # "For Q we included L2 weight decay of 10−2"
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta, weight_decay=0.00)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights_and_biases_uniformly(self):
        init_layer_uniformly_(self.fc1)
        init_layer_uniformly_(self.fc2)
        init_layer_uniformly_(self.q, min_value=-0.003, max_value=0.003)
        init_layer_uniformly_(self.action_value)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        # There seems to be a debate: relu before or after batch normalization?!
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        # Note: In the paper they state:
        # "Actions were not included until the 2nd hidden layer of Q."
        action_value = self.action_value(action) # <- Second hidden layer

        # torch.add(input, other, *, alpha=1, out=None) -> Tensor
        # Adds other, scaled by alpha, to input.
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        if os.path.isfile(self.checkpoint_file):
            print("Loading model from file: ", self.checkpoint_file)
            # map_location is required to ensure that a model that is trained on GPU can be run on CPU
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
        else:
            print(f"File not found: {self.checkpoint_file}. Continue training from scratch.")