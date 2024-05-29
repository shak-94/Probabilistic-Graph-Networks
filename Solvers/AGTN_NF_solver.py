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
from torch_geometric.utils import softmax
import torch.multiprocessing
from tqdm import tqdm
import numpy as np
import argparse
import sys
import os
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('--sim', default='spring', type=str,help='Choose a simulation to load, train and test')
parser.add_argument('--dim', default=None, type=int,help='Input simulation dimension')
parser.add_argument('--num_epochs', default=200, type=int,help='No. of training/test epochs')
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


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE_Encoder(nn.Module):
    def __init__(self, dim_in, hidden, hidden_out):
        super(MADE_Encoder, self).__init__()
        self.MADEncoder = MADEE_2(
            dim_in, [hidden, hidden], hidden_out, natural_ordering=True)  # test nout > nin

    def forward(self, x):
        out = self.MADEncoder(x)
        return out


class MADEE_2(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        Adapted from https://github.com/karpathy/pytorch-made
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks

        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        # self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        ordering = np.arange(start=1, stop=self.nin+1,
                             step=1) if self.natural_ordering else rng.permutation(self.nin)
        self.m[-1] = np.sort(np.concatenate([ordering for i in range(1)], axis=0))

        for l in range(L):
            self.m[l] = rng.randint(
                self.m[l-1].min(), self.nin+1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l-1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L-1][:, None] <= self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)


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


def node_up_sample_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden),
        nn.LayerNorm(hidden)
    )


def edge_up_sample_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden),
        nn.LayerNorm(hidden)
    )


def final_node_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, n_f),
        nn.ReLU(),
        nn.Linear(n_f, n_f),
        nn.ReLU(),
        nn.Linear(n_f, hidden))


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

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, ko: OptTensor = None,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key_in = self.lin_dim(torch.cat([ko, x[0]], 1))
        key = self.lin_key(key_in).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        edge_attr_in = self.lin_edge(edge_attr).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        aggr_out, e_features = self.propagate(edge_index, query=query, key=key, value=value,
                                              edge_attr=edge_attr_in, size=None)

        alpha = self._alpha
        self._alpha = None
        # print (aggr_out.shape)
        x_out = x[1] + self.enc_node_block(aggr_out)

        return x_out, e_features + edge_attr, key_in

    def message(self, edge_index, query_i, key_j, value_j, edge_attr, size_i):
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
        out = torch_scatter.scatter(
            edge_out, index, dim=node_dim, reduce='sum')

        return out, edge_out


class OGN(MessagePassing):
    def __init__(self, dim, device, look_back_length, k_fact, message_passing_steps, aggr='add', hidden=128):

        super(OGN, self).__init__(aggr=aggr)
        self.device = device
        self.dim = dim
        self.temporal_node_encoder = MADE_Encoder(
            dim*look_back_length, hidden=hidden, hidden_out=k_fact*dim*look_back_length)
        self.temporal_edge_encoder = MADE_Encoder(
            dim*look_back_length, hidden=hidden, hidden_out=k_fact*dim*look_back_length)
        self.gnn_stacks = nn.ModuleList([TransformerConv(in_channels=hidden, out_channels=hidden, heads=5, edge_dim=hidden, bias=True, aggr='add') for _ in range(message_passing_steps)])
        self.flow = NormalizingFlowModel(MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device)), [MAF(dim=dim, parity=i % 2) for i in range(message_passing_steps)])
        self.node_up_sample = node_up_sample_block(k_fact*dim+2, hidden)
        self.edge_up_sample = edge_up_sample_block(k_fact*dim, hidden)
        self.k_fact = k_fact

    def just_derivative(self, g):
        x = g.x
        edge_index = g.edge_index
        node_x = x[:, :-2]
        node_in = torch.cat([node_x[:, dim*(stride+1)+dim*stride:dim*(stride+1)+dim*stride+dim]
                            for stride in range(look_back_length*dim*2)], 1)  # Enforce Translation Invariance
        node_edge = torch.cat([node_x[:, dim*(stride)+dim*stride:dim*(stride)+dim*stride+dim]
                              for stride in range(look_back_length*dim*2)], 1)  # Enforce Translation Invariance

        ### Temporal Node Encoder
        temporal_node_out = self.temporal_node_encoder(node_in)

        ### Temporal Edge Encoder
        relative_displacements = (node_edge[edge_index[0]] - node_edge[edge_index[1]])
        e_features = self.temporal_edge_encoder(relative_displacements)

        accel_out = []
        ko = torch.zeros((x.shape[0], hidden)).to(device)
        nll_all_loss = 0.0

        ## Temporal Processor a.k.a. temporal propagator at time t
        for i in range(look_back_length):
            temporal_node_in = torch.cat([temporal_node_out[:, self.k_fact*dim*(i):self.k_fact*dim*(i+1)], x[:, -2:]], axis=1)
            temporal_node_in = self.node_up_sample(temporal_node_in)

            e_features_in = e_features[:,self.k_fact*dim*(i):self.k_fact*dim*(i+1)]
            e_features_in = self.edge_up_sample(e_features_in)

            ### Processor
            for gnn in self.gnn_stacks:
                temporal_node_in, e_features_in, ko = gnn(x=temporal_node_in, edge_index=edge_index, ko=ko, edge_attr=e_features_in, return_attention_weights=False)

            ### Decoder
            if self.training:
                zs, prior_logprob, log_det = self.flow(g.y[:, i*dim:(i+1)*dim], temporal_node_in)
                logprob = prior_logprob.sum() + log_det.sum()
                nll_all_loss += logprob

            else:
                zs = self.flow.sample(x.size(0), temporal_node_in)
                accel_out.append(zs[-1])

        if self.training:

            return -nll_all_loss

        else:
            return accel_out[-1]


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        Adapted from https://github.com/karpathy/pytorch-made

        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(
            self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(
                self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l-1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L-1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)


class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh, nh, nh, nh, nh, nh, nh], nout,
                        num_masks=2, natural_ordering=True)

    def forward(self, x):
        return self.net(x)


class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """

    def __init__(self, dim, parity, net_class=ARMLP, nh=16):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity
        self.context_s = nn.Linear(280, dim)
        self.context_t = nn.Linear(280, dim)

    def forward(self, x, h):
        # Evaluate Z in parallel, density estimation is fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        s = s + self.context_s(h)
        t = t + self.context_t(h)
        z = (x - t) * torch.exp(-0.5*s)
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = 0.5*torch.sum(-s, dim=1)
        return z, log_det

    def backward(self, z, h):
        # Decode X one at a time, sequentially, sampling will be slow
        x = torch.zeros_like(z).to(device)
        log_det = torch.zeros(z.size(0)).to(device)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone())
            s, t = st.split(self.dim, dim=1)
            s = s + self.context_s(h)
            t = t + self.context_t(h)
            x[:, i] = z[:, i] * torch.exp(0.5*s[:, i]) + t[:, i]
            log_det += s[:, i]
        return x, 0.5*log_det


class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, h):
        m, _ = x.shape
        log_det = torch.zeros(m).to(device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, h)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, h):
        m, _ = z.shape
        log_det = torch.zeros(m).to(device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, h)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x, h):
        zs, log_det = self.flow.forward(x, h)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, h):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples, h):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z, h)
        return xs

for test_sims in range(30):
    print("Running rollout #{}".format(test_sims))

    test_dataset = Particle_Dataset(simulation[test_sims*dataset_length:(test_sims+1)*dataset_length],current_states[test_sims*dataset_length:(test_sims+1)*dataset_length], updated_states[test_sims*dataset_length:(test_sims+1)*dataset_length], look_back_length)
    testloader = DataLoader([Data(x=test_dataset[i][0], edge_index=get_edge_index(nparticles), y=test_dataset[i][1]) for i in range(len(test_dataset))], batch_size=1, shuffle=False)
    
    aggr = 'add'
    hidden = 280
    message_passing_steps = 3

    if dim == 2:
        if look_back_length == 1:
            k_factor = 140

        elif look_back_length == 2:
            k_factor = 70

        elif look_back_length == 3:
            k_factor = 46

        elif look_back_length == 4:
            k_factor = 35

        elif look_back_length == 5:
            k_factor = 28

        elif look_back_length == 6:
            k_factor = 23

        elif look_back_length == 7:
            k_factor = 20

    elif dim == 3:
        if look_back_length == 1:
            k_factor = 93

        elif look_back_length == 2:
            k_factor = 46

        elif look_back_length == 3:
            k_factor = 31

        elif look_back_length == 4:
            k_factor = 23

        elif look_back_length == 5:
            k_factor = 18

        elif look_back_length == 6:
            k_factor = 15

        elif look_back_length == 7:
            k_factor = 13

    ogn = OGN(dim, device, look_back_length, k_factor, message_passing_steps, hidden=hidden, aggr=aggr)
    ogn.to(device)

    training_steps = int(2e7)
    lr_init = 7e-5 
    lr_min = 1e-6
    lr_decay = 0.1
    lr_decay_steps = int(5e6)
    lr_new = lr_init
    opt = torch.optim.Adam(ogn.parameters(), lr=lr_init, weight_decay=1e-8)

    checkpoint = torch.load('{}/Autoregress_GNF_G_Transformer_forward_{}_{}_{}_with_nll_final.pt'.format(
        root_dir, sim, dim, look_back_length))
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

        base_dir = f"{root_dir}/MSE_JCP/Autoregress_Gformer_GNF_{look_back_length}/{sim}"

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        np.save("{}/MSE_JCP/Autoregress_Gformer_GNF_{}/{}/Autoregress_Gformer_{}_{}_non_pos_no_ln_multi.npy".format(root_dir, look_back_length, sim,test_sims, dim), mse)
        np.save("{}/MSE_JCP/Autoregress_Gformer_GNF_{}/{}/Autoregress_Gformer_test_solution_{}_{}_non_pos_no_ln_multi.npy".format(root_dir, look_back_length, sim, test_sims, dim), nn_test_solution)
        np.save("{}/MSE_JCP/Autoregress_Gformer_GNF_{}/{}/Autoregress_Gformer_test_solved_{}_{}_non_pos_no_ln_multi.npy".format(root_dir, look_back_length, sim, test_sims, dim), nn_test_solved_long)
        plt.plot(range(figure_n_time_steps), mse[:figure_n_time_steps], c="blue")
        print("Done")
        plt.savefig("{}/MSE_JCP/Autoregress_Gformer_GNF_{}/{}/Autoregress_Gformer_{}_{}_non_pos_no_ln_multi.png".format(root_dir, look_back_length, sim, test_sims, dim))
        plt.close()
