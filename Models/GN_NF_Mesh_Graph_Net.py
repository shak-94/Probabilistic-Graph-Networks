from __future__ import print_function
from __future__ import division
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from GN_NF_Mesh_Graph_Net_Model import OGN
import torch.multiprocessing
import os
import matplotlib.pyplot as plt
import argparse
import random
from matplotlib import tri as mtri
from torch_geometric.data import Data
import torch.nn.functional as F
import enum
from mpl_toolkits.axes_grid1 import make_axes_locatable
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('--sim', default='Cylinder_Flow', type=str, help='Choose a simulation to load, train and test')
parser.add_argument('--dim', default=2, type=int, help='Input simulation dimension')
parser.add_argument('--num_epochs', default=5000, type=int, help='No. of training/test epochs')
parser.add_argument('--num_train_test_sim', default=60000, type=int, help='No. of training/test simulations')
parser.add_argument('--batch', default=4, type=int, help='No. of batches')
parser.add_argument('--device', default='cuda:3', type=str, help='CUDA Device ID')
args = parser.parse_args()

# print(torch.get_num_threads())
torch.set_num_threads(16)

# Potential (see below for options)
sim = args.sim
# Dimension
dim = args.dim
# Number of simulations
train_test_sim = args.num_train_test_sim
exp_name = "GN_NF_{}".format(args.sim)
# Import simulation file
simulation = torch.load("cylinder_flow/train_processed_set.pt") # Import simulation file
print ("Simulation Length:{} \nSingle Simulation Trajectory Shape:{}".format(len(simulation), simulation[0].x.shape))
nparticles = simulation[0].x.shape[0]
total_epochs = args.num_epochs

np.random.seed(42)
torch.manual_seed(42)

## Select CUDA Device ID
device = torch.device(args.device)
print("Training the model on:{}".format(device))


def normalize(to_normalize,mean_vec,std_vec):
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

def get_stats(data_list):
    '''
    Method for normalizing processed datasets. Given  the processed data_list, 
    calculates the mean and standard deviation for the node features, edge features, 
    and node outputs, and normalizes these using the calculated statistics.
    '''

    #mean and std of the node features are calculated
    mean_vec_x=torch.zeros(data_list[0].x.shape[1:])
    std_vec_x=torch.zeros(data_list[0].x.shape[1:])

    #mean and std of the edge features are calculated
    mean_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])

    #mean and std of the output parameters are calculated
    mean_vec_y=torch.zeros(data_list[0].y.shape[1:])
    std_vec_y=torch.zeros(data_list[0].y.shape[1:])
    
    #mean and std of the output parameters are calculated
    mean_vec_p = torch.zeros(data_list[0].p.shape[1:])
    std_vec_p = torch.zeros(data_list[0].p.shape[1:])
    

    #Define the maximum number of accumulations to perform such that we do
    #not encounter memory issues
    max_accumulations = 10**6

    #Define a very small value for normalizing to 
    eps=torch.tensor(1e-8)

    #Define counters used in normalization
    num_accs_x = 0
    num_accs_edge=0
    num_accs_y=0
    num_accs_p=0

    #Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        #Add to the 
        mean_vec_x+=torch.sum(dp.x,dim=0)
        std_vec_x+=torch.sum(dp.x**2,dim=0)
        num_accs_x+=dp.x.shape[0]

        mean_vec_edge+=torch.sum(dp.edge_attr,dim=0)
        std_vec_edge+=torch.sum(dp.edge_attr**2,dim=0)
        num_accs_edge+=dp.edge_attr.shape[0]

        mean_vec_y+=torch.sum(dp.y,dim=0)
        std_vec_y+=torch.sum(dp.y**2,dim=0)
        num_accs_y+=dp.y.shape[0]
        
        mean_vec_p+=torch.sum(dp.p,dim=0)
        std_vec_p+=torch.sum(dp.p**2,dim=0)
        num_accs_p+=dp.p.shape[0]
    

        if(num_accs_x > max_accumulations or num_accs_edge > max_accumulations or num_accs_y > max_accumulations or num_accs_p > max_accumulations):
            break

    mean_vec_x = mean_vec_x/num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x/num_accs_x - mean_vec_x**2),eps)

    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2),eps)

    mean_vec_y = mean_vec_y/num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y/num_accs_y - mean_vec_y**2),eps)
    
    mean_vec_p = mean_vec_p/num_accs_p
    std_vec_p = torch.maximum(torch.sqrt(std_vec_p/num_accs_p - mean_vec_p**2),eps)
    

    mean_std_list = [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, mean_vec_p, std_vec_p]

    return mean_std_list

#torch_geometric DataLoaders are used for handling the data of lists of graphs
random.shuffle(simulation)
for sim in range(len(simulation)):
    ab = F.one_hot(simulation[sim].x[:, -1].long(), num_classes=9)
    simulation[sim].x = torch.cat([simulation[sim].x[:, :2],ab],1)
    # simulation[sim].x = torch.cat([simulation[sim].x[:, :3],ab],1) ## Uncomment for Airfoil

loader = DataLoader(simulation[:int(args.num_train_test_sim*0.7)],batch_size=args.batch, shuffle=False)
test_loader = DataLoader(simulation[int(args.num_train_test_sim*0.7):], batch_size=args.batch, shuffle=False)

#The statistics of the data are decomposed
stats_list = get_stats(simulation)
[mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y,mean_vec_p, std_vec_p] = stats_list


# build model
num_node_features = simulation[0].x.shape[1]
num_edge_features = simulation[0].edge_attr.shape[1]
num_classes = 2 # the dynamic variables have the shape of 2 (velocity)
aggr = 'add'
hidden = 128
message_passing_steps = 15
look_back_length = 1

ogn = OGN(dim, device, message_passing_steps, aggr=aggr, hidden=hidden)
ogn.to(device)
delta_t = 0.01 # For cylinder flow
lr_init = 4e-4
lr_min = 1e-6
lr_decay = 0.1
lr_decay_steps = int(2.3e6)
lr_new = lr_init
opt = torch.optim.Adam(ogn.parameters(), lr=lr_init, weight_decay=1e-8)
i = 0
min_test_loss = float('inf')


logs = {'train_loss': [], 'test_loss': [], 'train_pressure_loss': []}

for epoch in tqdm(range(0, total_epochs)):
    total_loss = 0.0
    total_velocity_loss = 0.0
    total_pressure_loss = 0.0
    
    total_test_loss = 0.0
    total_test_velocity_loss = 0.0
    total_test_pressure_loss = 0.0
    total_test_position_loss = 0.0

    j = 0
    num_items = 0
    test_num_items = 0
    test_pos_num_items = 0
    print (f"Gradient Steps Complete:{i}")

    ogn.train()
    for ginput in loader:

        opt.zero_grad()
        ginput = ginput.to(device)

        (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, mean_vec_p, std_vec_p) = (mean_vec_x.to(device),
                                                                                        std_vec_x.to(device), mean_vec_edge.to(device), std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device),
                                                                                        mean_vec_p.to(device), std_vec_p.to(device))
        #Define the node types that we calculate loss for
        normal = torch.tensor(0)
        outflow = torch.tensor(5)

        loss_mask=torch.logical_or((torch.argmax(ginput.x[:,2:],dim=1)==torch.tensor(0)),
                                   (torch.argmax(ginput.x[:,2:],dim=1)==torch.tensor(5)))

        nll_vel = ogn.just_derivative(ginput, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, loss_mask, mean_vec_y, std_vec_y)
        (nll_vel/int(ginput.x.shape[0])).backward()

        opt.step()
        lr_new = lr_init * (lr_decay ** (i/lr_decay_steps))
        for g in opt.param_groups:
            g['lr'] = lr_new

        total_loss += nll_vel.item()
        i += 1
        num_items += 1
    cur_loss = total_loss/num_items
    logs['train_loss'].extend([cur_loss])
    
    print ("Training Loss:{}".format(logs['train_loss'][-1]),)

    ogn.eval()
    with torch.no_grad():
        for test_ginput in test_loader:

            test_ginput = test_ginput.to(device)
            (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, mean_vec_p, std_vec_p) = (mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device), 
                                                                                                                  std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device),
                                                                                                                  mean_vec_p.to(device), std_vec_p.to(device))            
            #Define the node types that we calculate loss for
            normal = torch.tensor(0)
            outflow = torch.tensor(5)

            #Get the loss mask for the nodes of the types we calculate loss for
            test_loss_mask=torch.logical_or((torch.argmax(test_ginput.x[:,2:],dim=1)==torch.tensor(0)),
                                            (torch.argmax(test_ginput.x[:,2:],dim=1)==torch.tensor(5)))
            test_velocity = ogn.just_derivative(test_ginput, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, test_loss_mask, mean_vec_y, std_vec_y)

            test_labels_velocity = test_ginput.y

            eval_velo = test_ginput.x[:, 0:2] + test_velocity * delta_t
            gs_velo = test_ginput.x[:, 0:2] + test_ginput.y[:] * delta_t

            test_loss = torch.sum((eval_velo - gs_velo)**2, axis=1)
            test_loss = torch.sqrt(torch.mean(test_loss[test_loss_mask]))

            total_test_loss += test_loss.item()
            test_num_items += 1
        
        # torch.cuda.empty_cache()
        test_cur_loss = total_test_loss/test_num_items
        logs['test_loss'].extend([test_cur_loss])
        
        print("Testing Loss:{}".format(logs['test_loss'][-1]))

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': ogn.state_dict(),
        'optimizer': opt.state_dict()
    }
    current_test_loss = logs['test_loss'][-1]
    if current_test_loss < min_test_loss:
        min_test_loss = current_test_loss  # Update the minimum test loss found so far
        print (f"Better model found and saved! Current min: {min_test_loss}")
        torch.save(checkpoint, '{}_checkpoint_GNF_2_new_gtn_4e_4_2_cylinder_flow_15.pt'.format(exp_name))

