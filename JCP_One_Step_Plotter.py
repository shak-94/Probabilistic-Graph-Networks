import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import os, sys

plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['savefig.dpi'] = 300


def MSE_plotter(n, sim, dim, kk, lbk):
    mses = []
    mrse = []
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(n):
        if kk == 0:
            # lbk = 3
            model_mse = np.load("{}/LTS_{}_more/{}/LTS_{}_{}_non.npy".format(root_dir, lbk, sim, i, dim))
            test_sol = np.load("{}/LTS_{}_more/{}/LTS_test_solution_{}_{}_non.npy".format(root_dir, lbk, sim, i, dim))
            test_solved = np.load("{}/LTS_{}_more/{}/LTS_test_solved_{}_{}_non.npy".format(root_dir, lbk, sim, i, dim))
            errors = []
            relative_errors = []
            for ii in range(970):
                error = np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2, axis=(1))
                error = np.sqrt(np.mean(error))
                errors.append(error)
                aa = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                ba = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]])**2,axis=(1)))
                ca = np.sqrt(np.sum((test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                re = np.mean(aa/(ba+ca))
                relative_errors.append(re)
            errors = np.array(errors)
            relative_errors = np.array(relative_errors)

        elif kk == 1:
            # lbk = 3
            test_sol = np.load("{}/Autoregress_Vanilla_{}/{}/Autoregress_Vanilla_test_solution_{}_{}_non_pos_no_ln.npy".format(root_dir, lbk, sim, i, dim))
            test_solved = np.load("{}/Autoregress_Vanilla_{}/{}/Autoregress_Vanilla_test_solved_{}_{}_non_pos_no_ln.npy".format(root_dir, lbk, sim, i, dim))
            model_mse = np.load("{}/Autoregress_Vanilla_{}/{}/Autoregress_Vanilla_{}_{}_non_pos_no_ln.npy".format(root_dir, lbk, sim, i, dim))
            errors = []
            relative_errors = []
            for ii in range(970):
                error = np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2, axis=(1))
                error = np.sqrt(np.mean(error))
                errors.append(error)
                aa = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                ba = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]])**2,axis=(1)))
                ca = np.sqrt(np.sum((test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                re = np.mean(aa/(ba+ca))
                relative_errors.append(re)
            errors = np.array(errors)
            relative_errors = np.array(relative_errors)


        elif kk == 2:
            # lbk = 4
            test_sol = np.load("{}/Lagrangian_{}_more/{}/Lagrangian_test_solution_{}_{}_non_ln.npy".format(root_dir, lbk, sim, i, dim))
            test_solved = np.load("{}/Lagrangian_{}_more/{}/Lagrangian_test_solved_{}_{}_non_ln.npy".format(root_dir, lbk, sim, i, dim))
            model_mse = np.load("{}/Lagrangian_{}_more/{}/Lagrangian_{}_{}_non_ln.npy".format(root_dir, lbk, sim, i, dim))
            errors = []
            relative_errors = []
            for ii in range(970):
                error = np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2, axis=(1))
                error = np.sqrt(np.mean(error))
                errors.append(error)
                aa = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                ba = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]])**2,axis=(1)))
                ca = np.sqrt(np.sum((test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                re = np.mean(aa/(ba+ca))
                relative_errors.append(re)
            errors = np.array(errors)
            relative_errors = np.array(relative_errors)

        elif kk == 3:
            # lbk = 5
            test_sol = np.load("{}/Hamiltonian_{}_more/{}/Hamiltonian_test_solution_{}_{}_non_standard_layer_norm.npy".format(root_dir, lbk, sim, i, dim))
            test_solved = np.load("{}/Hamiltonian_{}_more/{}/Hamiltonian_test_solved_{}_{}_non_standard_layer_norm.npy".format(root_dir, lbk, sim, i, dim))
            model_mse = np.load("{}/Hamiltonian_{}_more/{}/Hamiltonian_{}_{}_non_standard_layer_norm.npy".format(root_dir, lbk, sim, i, dim))
            errors = []
            relative_errors = []
            for ii in range(970):
                error = np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2, axis=(1))
                error = np.sqrt(np.mean(error))
                errors.append(error)
                aa = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]] - test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                ba = np.sqrt(np.sum((test_sol[ii+1,:,:dimensions[0]])**2,axis=(1)))
                ca = np.sqrt(np.sum((test_solved[ii,:,:dimensions[0]])**2,axis=(1)))
                re = np.mean(aa/(ba+ca))
                relative_errors.append(re)
            errors = np.array(errors)
            relative_errors = np.array(relative_errors)

        mses.append(errors)
        mrse.append(relative_errors)
    #     axs[0].plot(model_mse)
    # axs[1].plot(np.mean(mses, axis=0), label="Mean")
    means = np.mean(mses, axis=0)
    rmeans = np.mean(mrse, axis=0)
    print(means[0])
    # print(rmeans[-1])
    stds = np.std(mses, axis=0, ddof=1)
    upper = means + (1.96*stds)/np.sqrt(n-1)
    lower = means - (1.96*stds)/np.sqrt(n-1)
    x = np.arange(len(means))
    return x, means, lower, upper


def combiner(params, cols, time, mnr, dimen, sim):
    plt.figure(figsize=(10, 10))

    for i in range(len(models)):

        if i == 0:
            plt.plot(params[i][1][:time], color=cols[i], label="Mean RMSE of GN", linewidth=3)
            plt.fill_between(params[i][0][:time], params[i][2][:time], params[i][3][:time], alpha=0.2, color=cols[i])

        elif i == 1:
            plt.plot(params[i][1][:time], color=cols[i], label="Mean RMSE of GN-NF", linewidth=3)
            plt.fill_between(params[i][0][:time], params[i][2][:time], params[i][3][:time], alpha=0.2, color=cols[i])
            
        elif i == 2:
            plt.plot(params[i][1][:time], color=cols[i], label="Mean RMSE of GLN", linewidth=3)
            plt.fill_between(params[i][0][:time], params[i][2][:time], params[i][3][:time], alpha=0.2, color=cols[i])

        elif i == 3:
            plt.plot(params[i][1][:time], color=cols[i], label="Mean RMSE of GHN", linewidth=3)
            plt.fill_between(params[i][0][:time], params[i][2][:time], params[i][3][:time], alpha=0.2, color=cols[i])
            

    plt.legend(loc="upper left")
    plt.xlabel("Timesteps")
    plt.ylabel("RMSE")
    if math.isnan(mnr[-1]) or math.isinf(mnr[-1]):
        up = mnr[-2]
    else:
        up = mnr[-1]
    plt.ylim(0.0, up)
    plt.savefig('{}/ONE_step_multi_solver_{}_{}.png'.format(save_dir, sim, dimen))
    plt.close()


s_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
root_dir = f'{s_dir}/MSE_JCP'
save_dir = f'{s_dir}/Solvers/Results'

models = ['Graph Network', 'Graph Network NF', 'Lagrangian', 'Hamiltonian']

colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'tab:brown', 'tab:purple']


### Change these parameters to generate required results
simulations = ['charge'] 
dimensions = [3]
look_back_length = [6]
num_simulations = 30
num_timesteps = 970

for simulation in simulations:
    for dimension in dimensions:
        plot_params = []
        meaner = []
        for kk, model in enumerate(models):
            x, m, l, u = MSE_plotter(num_simulations, simulation, dimension, kk, look_back_length[0])
            plot_params.append([x, m, l, u])
            meaner.append(m[-1])
            # print(model)
        meaner.sort()
        # print("##############")
        # combiner(plot_params, colors, num_timesteps, meaner, dimension, simulation)
