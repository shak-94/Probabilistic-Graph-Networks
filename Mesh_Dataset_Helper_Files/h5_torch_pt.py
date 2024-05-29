import os
import numpy as np
import torch
import h5py
import tensorflow.compat.v1 as tf
import functools
import json
from torch_geometric.data import Data
import enum
import torch
import random
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader

import numpy as np
import time
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

root_dir = os.getcwd()
simulation = 'cylinder_flow'
mode = 'test'
dataset_dir = os.path.join(root_dir, simulation)
checkpoint_dir = os.path.join(root_dir, 'best_models')
postprocess_dir = os.path.join(root_dir, 'animations')

tf.enable_resource_variables()
tf.enable_eager_execution()

#Utility functions, provided in the release of the code from the original MeshGraphNets study:
#https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets

def triangles_to_edges(faces):
  """Computes mesh edges from triangles.
     Note that this triangles_to_edges method was provided as part of the
     code release for the MeshGraphNets paper by DeepMind, available here:
     https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
  """
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))



class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


#Define the data folder and data file name
datafile = os.path.join(root_dir + '/{}/{}{}.h5'.format(simulation, simulation, mode))
data = h5py.File(datafile, 'r')

#Define the list that will return the data graphs
data_list = []

#define the time difference between the graphs
dt=0.01   #A constant: do not change! (Cylinder Flow and SphereDynamic)
# dt=0.02   #A constant: do not change! (FlagSimple and FlagDynamic)
# dt=0.008  #A constant: do not change! (AirFoil)
#define the number of trajectories and time steps within each to process.
#note that here we only include 2 of each for a toy example.
# number_trajectories = len(data)
# number_ts = data['0']['cells'].shape[0]
number_trajectories = 30
number_ts = 600

# Inspect data shapes
for item in data['0'].keys():
  print('{} : {}'.format(item, data['0'][item].shape))

with h5py.File(datafile, 'r') as data:

    for i,trajectory in enumerate(data.keys()):
        if(i==number_trajectories):
            break
        print("Trajectory: ",i)

        #We iterate over all the time steps to produce an example graph except
        #for the last one, which does not have a following time step to produce
        #node output values
        for ts in range(len(data[trajectory]['velocity'])-1):

            if(ts==number_ts):
                break

            #Get node features

            #Note that it's faster to convert to numpy then to torch than to
            #import to torch from h5 format directly
            momentum = torch.tensor(np.array(data[trajectory]['velocity'][ts]))
            # rho_t=torch.tensor(np.array(data[trajectory]['density'][ts])) ## Uncomment only for airfoil dataset
            node_type = torch.tensor(np.array(data[trajectory]['node_type'][ts]))
            # node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['node_type'][0]), NodeType.SIZE))).squeeze(1)
            # x = torch.cat((momentum, rho_t, node_type),dim=-1).type(torch.float)
            x = torch.cat((momentum, node_type),dim=-1).type(torch.float)

            #Get edge indices in COO format
            edges = triangles_to_edges(tf.convert_to_tensor(np.array(data[trajectory]['cells'][ts])))

            edge_index = torch.cat((torch.tensor(edges[0].numpy()).unsqueeze(0) ,
                         torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

            #Get edge features
            u_i=torch.tensor(np.array(data[trajectory]['pos'][ts]))[edge_index[0]]
            u_j=torch.tensor(np.array(data[trajectory]['pos'][ts]))[edge_index[1]]
            u_ij=u_i-u_j
            u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
            edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)

            #Node outputs, for training (velocity)
            v_t=torch.tensor(np.array(data[trajectory]['velocity'][ts]))
            v_tp1=torch.tensor(np.array(data[trajectory]['velocity'][ts+1]))
            y=((v_tp1-v_t)/dt).type(torch.float)
            
            #Node outputs, for training (density) ## Only for airfoil dataset
            # rho_t=torch.tensor(np.array(data[trajectory]['density'][ts]))
            # rho_tp1=torch.tensor(np.array(data[trajectory]['density'][ts+1]))
            # rho_y=((rho_tp1-rho_t)/dt).type(torch.float)

            #Node outputs, for testing integrator (pressure)
            p=torch.tensor(np.array(data[trajectory]['pressure'][ts]))

            #Data needed for visualization code
            cells=torch.tensor(np.array(data[trajectory]['cells'][ts]))
            mesh_pos=torch.tensor(np.array(data[trajectory]['pos'][ts]))

            ## Uncomment for airfoil dataset
            # data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, rho_y=rho_y, p=p, cells=cells, mesh_pos=mesh_pos))
            
            ## Uncomment for cylinder flow dataset
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, p=p, cells=cells, mesh_pos=mesh_pos))

print("Done collecting data!")
torch.save(data_list,os.path.join('{}/{}/{}_processed_set_3.pt'.format(root_dir, simulation,mode)))
print("Done saving data!")
print("Output Location: ", dataset_dir+'/{}_processed_set_3.pt'.format(mode))