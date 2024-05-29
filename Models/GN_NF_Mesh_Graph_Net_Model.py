from __future__ import print_function
from __future__ import division
import math
import numpy as np
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch import Tensor
import torch_scatter
from torch.distributions import MultivariateNormal
from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch.multiprocessing
from torch_geometric.nn import MessagePassing

device="cuda:1"
hidden = 128

def normalize(to_normalize, mean_vec, std_vec):
    return (to_normalize-mean_vec)/std_vec

def edge_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden)
    )


def node_block(n_f, hidden):
    return nn.Sequential(
        nn.Linear(n_f, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden)
    )

class InteractionNetwork(MessagePassing):
    def __init__(self, hidden=256, aggr='add'):
        super(InteractionNetwork, self).__init__(aggr=aggr)
        self.enc_edge_block = edge_block(3*hidden, hidden)
        self.enc_node_block = node_block(2*hidden, hidden)

    def forward(self, x, edge_index, e_features):
        x_residual = x
        aggr_out, e_features = self.propagate(edge_index, size=(x.size(0), x.size(0)), e_features=e_features, x=x)
        tmp = torch.cat([x, aggr_out], axis=1)
        x = self.enc_node_block(tmp)
        return x + x_residual, e_features

    def message(self, x_i, x_j, e_features):
        tmp = torch.cat([x_i, x_j, e_features], dim=1)
        edge_out = self.enc_edge_block(tmp) + e_features
        return edge_out
    
    def aggregate(self, edge_out, edge_index, dim_size = None):
        # The axis along which to index number of nodes.

        node_dim = 0
        out = torch_scatter.scatter(edge_out, edge_index[0, :], dim=node_dim, reduce = 'sum')

        return out, edge_out

class Encoder(MessagePassing):
    def __init__(self, dim, hidden=256):
        super(Encoder, self).__init__()
        self.enc_edge_block = edge_block(3, hidden)
        self.enc_node_block = node_block(dim, hidden)
        self.dim = dim

    def forward(self, node_in, node_edge):
        edge_out = self.enc_edge_block(node_edge)
        node_out = self.enc_node_block(node_in)
        return node_out, edge_out

class OGN(MessagePassing):
    def __init__(self, dim, device, message_passing_steps, aggr='add', hidden=128):

        super(OGN, self).__init__(aggr=aggr)
        self.device = device
        self.dim = dim
        # self._encoder = Encoder(dim+1+9, hidden=hidden) ## Uncomment for Airfoil
        self._encoder = Encoder(dim+9, hidden=hidden)
        self.gnn_stacks = nn.ModuleList([InteractionNetwork(hidden=hidden, aggr='add') for _ in range(message_passing_steps)])
        self.velocity_flow = NormalizingFlowModel(MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device)), [MAF(dim=dim, parity=i % 2) for i in range(2)])

    def just_derivative(self, g, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, losss_mask, mean_vec_y, std_vec_y):
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr

        # x = normalize(x, mean_vec_x, std_vec_x)
        # edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        ### Temporal Node Encoder
        x, e_features = self._encoder(x, edge_attr)

        ### Processor
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)

        ### Decoder
        if self.training:
            # labels_velocity = normalize(g.y, mean_vec_y, std_vec_y)
            z_vel, prior_logprob_vel, log_det_vel = self.velocity_flow(g.y, x)
            logprob_vel = prior_logprob_vel.view(-1,1)[losss_mask].sum() + log_det_vel.view(-1,1)[losss_mask].sum()
            return -logprob_vel

        else:
            vel_out = self.velocity_flow.sample(x.size(0), x)
            return vel_out[-1]

    
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
        self.net = MADE(nin, [nh, nh, nh, nh, nh, nh, nh, nh], nout, num_masks=2, natural_ordering=True)

    def forward(self, x):
        return self.net(x)


class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """

    def __init__(self, dim, parity, net_class=ARMLP, nh=8):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity
        self.context_s = nn.Linear(128, dim)
        self.context_t = nn.Linear(128, dim)

    def forward(self, x, h):
        # Evaluating all of Z in parallel, density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        s = s + self.context_s(h)
        t = t + self.context_t(h)
        z = (x - t) * torch.exp(-0.5*s)
        # z = x * torch.exp(0.5*s) + t
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = 0.5*torch.sum(-s, dim=1)
        return z, log_det

    def backward(self, z, h):
        # Decode the X one at a time, sequentially, sampling will be slow
        x = torch.zeros_like(z).to(device)
        log_det = torch.zeros(z.size(0)).to(device)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone())
            s, t = st.split(self.dim, dim=1)
            s = s + self.context_s(h)
            t = t + self.context_t(h)
            x[:, i] = z[:, i] * torch.exp(0.5*s[:, i]) + t[:, i]
            # x[:, i] = (z[:, i] - t[:, i]) * torch.exp(0.5*-s[:, i])
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
