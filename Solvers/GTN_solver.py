from __future__ import print_function
from __future__ import division
import math
from typing import Optional, Tuple, Union
from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch import Tensor
import torch_scatter
import torch.distributions as dist
from torch.distributions import MultivariateNormal
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.data import Data, DataLoader
from torch.utils.data.dataset import Dataset
import torch.multiprocessing
from tqdm import tqdm
import numpy as np
import argparse
from torch_geometric.utils import softmax
import os, sys
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('--sim', default='spring', type=str,help='Choose a simulation to load, train and test')
parser.add_argument('--dim', default=None, type=int,help='Input simulation dimension')
parser.add_argument('--num_epochs', default=400, type=int,help='No. of training/test epochs')
parser.add_argument('--num_train_test_sim', default=200,type=int, help='No. of training/test simulations')
parser.add_argument('--batch', default=1, type=int, help='No. of batches')
parser.add_argument('--device', default='cpu', type=str, help='CUDA Device ID')
parser.add_argument('--look_back', default=1, type=int, help='Look Back Length')

args = parser.parse_args()

# Potential (see below for options)
sim = args.sim
# Number of nodes
n = 4
# Dimension
dim = args.dim
# Number of time steps
nt = 2000
# Number of simulations
ns = 230
# Number of simulations
train_test_sim = args.num_train_test_sim
n_set = [4, 8]
sim_sets = [
    {'sim': 'r1', 'dt': [5e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'r2', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'spring', 'dt': [1e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'charge', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'damped', 'dt': [2e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
]
#Select the hand-tuned dt value for a smooth simulation
# (since scales are different in each potential):
fps = [ss['dt'][0] for ss in sim_sets if ss['sim'] == sim][0]
title = '{}_n={}_dim={}_nt={}_dt={}_ns={}'.format(sim, n, dim, nt, fps, ns)
print(title)
s_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
script_folder_dir = os.path.abspath(os.path.join(s_dir, os.pardir))
root_dir = f"{script_folder_dir}"
# Import simulation file
simulation = torch.from_numpy(np.load(root_dir + '/Particle_Physics_Datasets/' + title + '_x.npy')) # Import simulation file
simulation_y=torch.from_numpy(np.load(root_dir + '/Particle_Physics_Datasets/' + title + '_y.npy'))

print ("Simulation Shape:{}".format(simulation.shape))
nparticles = simulation.shape[2]
n_time_steps = 1000
figure_n_time_steps = 300
simulation = simulation[train_test_sim:, :n_time_steps, :, :]
simulation_y = simulation_y[train_test_sim:, :n_time_steps, :, :]


simulation_length = simulation.shape[0]
look_back_length = args.look_back
print ("Look Back Length:{}".format(look_back_length))
total_epochs = 200
dataset_length = n_time_steps-look_back_length+1

np.random.seed(42)
torch.manual_seed(42)
look_back_length_addition = look_back_length*dim + dim*2

## Select CUDA Device ID
device = torch.device(args.device)
print("Solving the model on:{}".format(device))
np.random.seed(42)
torch.manual_seed(42)

def get_edge_index(n):
    adj = (np.ones((n, n)) - np.eye(n)).astype(int)
    edge_index = torch.from_numpy(np.array(np.where(adj)))
    return edge_index

def rk4_step(f, x, t, h):
  # one step of runge-kutta integration
  x = x.reshape(n, dim*2)
  k1 = h * f(x, t)
  k2 = h * f(x + k1/2, t + h/2)
  k3 = h * f(x + k2/2, t + h/2)
  k4 = h * f(x + k3, t + h)
  return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def f(x, t, conditioning=None, tenso=None):
    x = x.reshape(-1, dim*2)
    x = np.concatenate([conditioning[:,:-2], x, conditioning[:,-2:]], axis=1)
    test_regress_acceleration = ogn.just_derivative(tenso.to(device))
    test_regress_acceleration = test_regress_acceleration.cpu().detach().numpy()
    x_dot = np.concatenate([x[:, -dim-2:-2].reshape(-1, dim),test_regress_acceleration.reshape(-1, dim)], axis=1)

    return x_dot

def nn_solve_ode(x0, t, fps):
    tenso = x0
    x0 = x0.x.cpu().detach().numpy()
    hh = np.concatenate([x0[:, -dim*2-2:-2], x0[:, -2:]], axis=1)
    LL = partial(f, conditioning=hh, tenso=tenso)
    return rk4_step(LL, x0[:, -dim*2-2:-2].reshape(n*dim*2), t, fps).reshape(-1, dim*2)

## Process Data
def window_maker(input, nparticles, look_back_length, dim):
    time_period = int(input.shape[0]/nparticles)
    current_appender = []
    new_appender = []
    for seq in range(time_period):
        current_appender.append(input[seq*nparticles:(seq+1)*nparticles])
        if len(current_appender) >= look_back_length:
            current_appenderr = [current_appender[i] for i in range(len(current_appender))]  # Note difference between appender and appenderr
            look_back = torch.cat(current_appenderr[-look_back_length:], dim=1)
            new_appender.append(look_back)
        else:
            pass
    updated_current_state = torch.cat(new_appender, dim=0)
    assert updated_current_state.shape[1] == look_back_length*dim*2
    return updated_current_state


def another_window_maker(input, nparticles, look_back_length, dim, assert_size):
    time_period = int(input.shape[0]/nparticles)
    current_appender = []
    new_appender = []
    for seq in range(time_period):
        current_appender.append(input[seq*nparticles:(seq+1)*nparticles])
        if len(current_appender) >= look_back_length:
            current_appenderr = [current_appender[i] for i in range(len(current_appender))]  # Note difference between appender and appenderr
            look_back = torch.cat(current_appenderr[-look_back_length:], dim=-1)
            new_appender.append(look_back)
        else:
            pass
    updated_current_state = torch.cat(new_appender, dim=0)
    assert updated_current_state.shape[1] == assert_size
    return updated_current_state

def processed(simulation, simulation_y, look_back_length, nparticles, dim):
    current_states = []
    updated_states = []
    for sim in range(simulation.shape[0]):
        current_state = torch.cat([simulation[sim, i, :, :-2] for i in range(0, simulation.shape[1], 1)])
        current_state = window_maker(current_state, nparticles, look_back_length, dim)
        
        miscellaneous = torch.cat([simulation[sim, i, :, -2:]for i in range(look_back_length-1, simulation.shape[1], 1)])
        current_state = torch.cat([current_state, miscellaneous], dim=1)
        current_states.append(current_state)

        updated_state = torch.cat([torch.cat([simulation_y[sim, i, :, :]],1) for i in range(0, simulation.shape[1], 1)])
        updated_state = another_window_maker(updated_state, nparticles, look_back_length, dim, look_back_length*dim)
        updated_states.append(updated_state)

    current_states = torch.stack(current_states)
    current_states = current_states.view(-1,nparticles, current_states.shape[-1])
    
    updated_states = torch.stack(updated_states)
    updated_states = updated_states.view(-1,nparticles, updated_states.shape[-1])

    return current_states, updated_states


current_states, updated_states = processed(simulation, simulation_y, look_back_length, nparticles, dim)
print (current_states.shape, updated_states.shape)

class Particle_Dataset(Dataset):
    def __init__(self, simulation, current_states, updated_states, look_back_length):
        self.simulation = simulation.type(torch.FloatTensor)
        self.current_states = current_states.type(torch.FloatTensor)
        self.updated_states = updated_states.type(torch.FloatTensor)
        self.look_back_length = look_back_length

    def __len__(self):
        return dataset_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.current_states[idx], self.updated_states[idx])


def enc_edge_block(n_f, hidden_in, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden_in),
        nn.ReLU(),
        nn.Linear(hidden_in, hidden_in),
        nn.ReLU(),
        nn.Linear(hidden_in, hidden),
        nn.LayerNorm(hidden)
    )


def enc_node_block(n_f, hidden_in, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden_in),
        nn.ReLU(),
        nn.Linear(hidden_in, hidden_in),
        nn.ReLU(),
        nn.Linear(hidden_in, hidden),
        nn.LayerNorm(hidden)
    )


def edge_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden)
    )


def node_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden)
    )
    
    
def final_node_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, n_f),
        nn.ReLU(),
        nn.Linear(n_f, n_f),
        nn.ReLU(),
        nn.Linear(n_f, hidden))
    
class Encoder(MessagePassing):
    def __init__(self, dim, hidden=256, aggr='add'):
        super(Encoder, self).__init__(aggr=aggr)
        self.enc_edge_block = edge_block(dim+1, hidden)
        self.enc_node_block = node_block(dim+2, hidden)
        self.dim = dim

    def forward(self, node_in, edge_index, node_edge):
        relative_displacements_t = (node_edge[edge_index[0]] - node_edge[edge_index[1]])
        normed_relative_displacements_t = torch.norm(relative_displacements_t, dim=1, keepdim=True)
        e_features_t = torch.cat([relative_displacements_t, normed_relative_displacements_t], dim=1)
        edge_out = self.enc_edge_block(e_features_t)
        node_out = self.enc_node_block(node_in)
        return node_out, edge_out

class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.lin_edge = Linear(edge_dim, heads * out_channels)
        self.lin_dim = Linear(in_channels[0]*2, in_channels[0])

        
        self.enc_node_block = node_block(hidden, hidden)
        self.enc_edge_block = edge_block(hidden*5, hidden)
        self.m = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_dim.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, l_cache: OptTensor = None,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
            
        query = self.lin_query(x[1]).view(-1, H, C)
        key_in = self.lin_dim(torch.cat([l_cache, x[0]],1))
        key = self.lin_key(key_in).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        edge_attr_in = self.lin_edge(edge_attr).view(-1, H, C)
            
        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        aggr_out, e_features = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr_in, size=None, previous=l_cache)
        
        alpha = self._alpha
        self._alpha = None
        
        x_out = x[1] + self.enc_node_block(aggr_out)
        
        return x_out, e_features + edge_attr, key_in



    def message(self, edge_index, query_i, key_j, value_j, edge_attr, size_i, previous):
        key_j = key_j + edge_attr
        alpha = (query_i * key_j).sum(-1) / math.sqrt(self.out_channels)
        
        # alpha = self.m(alpha)
        # self._alpha = alpha
        alpha = softmax(alpha, edge_index[0])
        self._alpha = alpha
        edge_out = key_j * alpha.view(-1, self.heads, 1)

        return self.enc_edge_block(edge_out.view(-1, self.heads*hidden))

    def aggregate(self, edge_out, index, dim_size=None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """
        # The axis along which to index number of nodes.
        node_dim = self.node_dim
        out = torch_scatter.scatter(edge_out, index, dim=node_dim, reduce='sum')

        return out, edge_out
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
        
class OGN(MessagePassing):
    def __init__(self, dim, device, look_back_length, message_passing_steps, aggr='add', hidden=128):

        super(OGN, self).__init__(aggr=aggr)
        self.device = device
        self.dim = dim
        self.look_back_length = look_back_length
        self._encoder = Encoder(dim, hidden=hidden, aggr='add')
        self.gnn_stacks = nn.ModuleList([TransformerConv(in_channels=hidden, out_channels=hidden, heads=5, edge_dim=hidden, bias=True, aggr='add') for _ in range(message_passing_steps)])
        self.spatial_decoder = final_node_block(hidden, dim)
        
    def just_derivative(self, g):
        x = g.x
        edge_index = g.edge_index
        
        accel_out = []
        ko = torch.zeros((x.shape[0], hidden)).to(device)

        ## Temporal Processor a.k.a. temporal propagator at time t
        for i in range(self.look_back_length):
            
            ### Encoder
            node_in = x[:, (i+1)*dim + dim*i:(i+1)*dim + dim*i + dim] ## Enforce Translation Invariance
            node_edge = x[:, i*dim + dim*i:i*dim + dim*i + dim]  # Enforce Translation Invariance
            node_in, e_features = self._encoder(torch.cat([node_in, x[:, -2:]],1), edge_index, node_edge)
            

            ### Processor
            for gnn in self.gnn_stacks:
                node_in, e_features, ko = gnn(node_in, edge_index, l_cache=ko, edge_attr=e_features, return_attention_weights=False)


            ### Decoder
            node_in = self.spatial_decoder(node_in)
            accel_out.append(node_in)
            
        return accel_out[-1]

for test_sims in range(30):
    print("Running rollout #{}".format(test_sims))

    test_dataset = Particle_Dataset(simulation[test_sims*dataset_length:(test_sims+1)*dataset_length],current_states[test_sims*dataset_length:(test_sims+1)*dataset_length], updated_states[test_sims*dataset_length:(test_sims+1)*dataset_length], look_back_length)
    testloader = DataLoader([Data(x=test_dataset[i][0], edge_index=get_edge_index(nparticles), y=test_dataset[i][1]) for i in range(len(test_dataset))], batch_size=1, shuffle=False)
    
    aggr = 'add'
    hidden = 280
    message_passing_steps = 3
    ogn = OGN(dim, device, look_back_length, message_passing_steps, hidden=hidden, aggr=aggr)
    ogn.to(device)

    training_steps = int(2e7)
    lr_init = 7e-5
    lr_min = 1e-6
    lr_decay = 0.1
    lr_decay_steps = int(5e6)
    lr_new = lr_init
    opt = torch.optim.Adam(ogn.parameters(), lr=lr_init, weight_decay=1e-8)

    checkpoint = torch.load('{}/LTS_Orig_forward_{}_{}_{}_{}_transformer_trans_inv_t_steps.pt'.format(root_dir, sim, dim, look_back_length, train_test_sim))
    ogn.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

    ogn.eval()
    with torch.no_grad():
        nn_test_solved = np.empty((len(test_dataset), n, dim*2))
        nn_test_solution = np.empty((len(test_dataset), n, dim*2))
        error = np.empty((len(test_dataset), n, dim*2))
        
        for i, test_ginput in enumerate(tqdm(testloader)):
            test_ginput = test_ginput.to(device)
            nn_test_solution[i] = test_ginput.x[:,-dim*2-2:-2].cpu().detach().numpy()

        nn_test_solved_long = np.empty((len(test_dataset), n, dim*2))
            
        for i, test_ginput in enumerate(tqdm(testloader)):
            if i == 0:
                initial_condition_x = test_ginput.x
                previous_velocities = initial_condition_x[:, dim*2:-2]
            if i > 0:
                initial_condition_x = torch.tensor(initial_condition_x)
                if look_back_length == 1:
                    initial_condition_x = torch.cat([initial_condition_x, test_ginput.x[:,-2:]], axis=1)
                else:
                    initial_condition_x = torch.cat([previous_velocities, initial_condition_x, test_ginput.x[:,-2:]], axis=1)
                    previous_velocities = torch.cat([previous_velocities[:, dim*2:], initial_condition_x[:,-dim*2-2:-2]], axis=1)
                test_ginput.x = initial_condition_x
            initial_condition_x = nn_solve_ode(test_ginput, fps, fps)
            nn_test_solved_long[i] = initial_condition_x

        mse = (np.square(nn_test_solution[1:, :, :dim] - nn_test_solved_long[:-1, :, :dim])).mean(axis=1)
        if dim==2:
            mse = np.sqrt((np.square(mse[1:, 0]) + np.square(mse[1:, 1])))
        else:
            mse = np.sqrt((np.square(mse[1:, 0]) + np.square(mse[1:, 1]) + np.square(mse[1:, 2])))
        

        base_dir = f"{root_dir}/MSE_NPJ/Gformer_{look_back_length}_more/{sim}"

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        np.save("{}/MSE_NPJ/Gformer_{}_more/{}/Autoregress_Gformer_{}_{}_non_pos_no_ln_multi.npy".format(root_dir, look_back_length, sim,test_sims, dim), mse)
        np.save("{}/MSE_NPJ/Gformer_{}_more/{}/Autoregress_Gformer_test_solution_{}_{}_non_pos_no_ln_multi.npy".format(root_dir, look_back_length, sim, test_sims, dim), nn_test_solution)
        np.save("{}/MSE_NPJ/Gformer_{}_more/{}/Autoregress_Gformer_test_solved_{}_{}_non_pos_no_ln_multi.npy".format(root_dir, look_back_length, sim, test_sims, dim), nn_test_solved_long)
        plt.plot(range(figure_n_time_steps), mse[:figure_n_time_steps], c="blue")
        print("Done")
        plt.savefig("{}/MSE_NPJ/Gformer_{}_more/{}/Autoregress_Gformer_{}_{}_non_pos_no_ln_multi.png".format(root_dir, look_back_length, sim, test_sims, dim))
        plt.close()
