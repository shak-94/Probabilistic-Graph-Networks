# Probabilistic-Graph-Networks

This is the official PyTorch implementation of Probabilistic Graph Networks for Learning Physics Simulations.
Link to Paper: https://www.sciencedirect.com/science/article/pii/S0021999124003863.

## Running the Experiments
1. The models are located in the Models folder.
2. To train a model, open a specific model file in an editor of your choice. Then ensure that the root directory, simulation_x and simulation_y files are pointed to the right location. Then simply provide the arguments (batch size, dataset to train/test, epochs to train/test, cuda as well as the number of simulations to train on). The files are already populated with the parameters that were reported in the paper. Then run the python file using your terminal.
3. To perform roll-out inference on unseen trajectories, open the specific solver file in the Solvers folder. Then ensure that the root directory, simulation_x, simulation_y and the trained_model.pt files are pointed to the right location. The files are already populated with the parameters that were reported in the paper. Then run the python file using your terminal.
4. To train models on the mesh-based datasets, you will need to download the datasets from the official deepmind repository (https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) since these datasets are too huge to be uploaded to the repository directly.
5. Once downloaded, these files will be in a tfrecord format. You can use the "tfrecord_to_h5.py" script located in the "Mesh_Dataset_Helper_Files" directory to first convert them to be a h5 file and then consequently to Pytorch friendly files using "h5_torch_pt.py" script.
6. Then simply locate the Mesh Graph Net-NF models from the Models folder and ensure that the root directory and simulation directory are pointing to the right locations. Once complete, simply run the python file using your terminal.


## References
If you found part of this work helpful, please consider citing this article:

**Prakash, S.K.A., & Tucker, C.** (2024). *Probabilistic Graph Networks for Learning Physics Simulations*. Journal of Computational Physics, 113137. Elsevier. [https://doi.org/10.1016/j.jcp.2024.113137](https://doi.org/10.1016/j.jcp.2024.113137)


We thank the authors of (https://github.com/MilesCranmer/symbolic_deep_learning) for sharing their code as well as their particle-physics datasets. We also thank the authors of (https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) for sharing the code for Mesh Graph Net and mesh-based dataset (Cylinder Flow).
