# Principled data augmentation for learning to solve quadratic programming problems

Our work targets at generating more LP and LCQP data given some data instances, for supervised learning and contrastive pretraining to learn useful representations. Some useful materials [SSL survey](https://github.com/LirongWu/awesome-graph-self-supervised-learning), [PyGCL](https://github.com/PyGCL/PyGCL), [PyTorch metric learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)

## Environment setup

```angular2html
conda create -y -n ipmgnn python=3.11
conda activate da4l2o
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.5.3  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_cluster-1.6.3%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install wandb hydra-core pytorch-metric-learning

conda install -y -c conda-forge qpsolvers 
# for larger problems you need a license, please visit https://www.gurobi.com/ for more information
pip install gurobipy
```

## Data generation
For generating LP/QP data used in our supervised learning (Table 1) and contrastive pretraining-supervised finetuning (Table 2) experiments, as well as larger instances (Table 7) please see to `generate_lp.ipynb` and `generate_qp.ipynb`.

For out-of-distribution (OOD) dataset generation, please see to `generate_other_lp.ipynb` and `generate_other_qp.ipynb`. For the 4 LP problems, we adopt the code from https://arxiv.org/abs/1906.01629 and https://arxiv.org/abs/2310.10603. For the QP problems, we refer to https://arxiv.org/abs/2211.12443.

For QPLIB, first you need to install QPLIB on your own from https://qplib.zib.de/qplib.zip, then extract, and play around with our `QPLIB_enrich.ipynb`, which processes the raw data into our data form, and enrich the set. 

## Our proposed data augmentation

__Remove idle variable__: We can remove a variable whose optimal value is 0.

__Remove inactive constraint__: We can remove a constraint that is inactive, i.e., a constraint `i` such that `A_i x < b` strictly holds. 

__Scale variable coefficients__: We can scale the coordinates of the problem. For all the coefficients wrt variable `x_i`, we scale them by a scalar `alpha_i != 0`.

__Scale constraint coefficients__: For a constraint i, we can scale all the `A_ij` and `b_i` with a positive scalar `alpha_j > 0`.

__Add variable__: We can add a variable with some modification on `Q`, `A`, `c` such that the new variable takes an optimal value of 0.

__Add constraint__: We can add a constraint that is a convex combination of some existing constraints. 

## Deployment

### Supervised learning

#### Baselines
__Normal__: For normal training, just run 

Example code:  
`python run.py exp.datapath=PATH2DATA`

plus some customized arguments. Data scarcity is controlled by an argument `finetune.train_frac` and another controlling number of folds `finetune.folds`. In principle, if you use 10% of the training data, then you should run for 10 folds; if you use 20%, then 5 folds, etc.

__GraphCL__: GraphCL proposes 3 types of graph data augmentation: node dropping, edge flipping, node feature masking. 

Example code:  
`python run.py --config-name su_aug_dropn`  
`python run.py --config-name su_aug_edge`  
`python run.py --config-name su_aug_mask`

#### Ours
We provide each individual data augmentation as well as a combination of all. We introduce an interpolation factor on the augmentation strength, to make use of both the original data and augmented data. For the combination, we can randomly sample a subset of augmentations for each instance at each epoch. 

Example code:  
`python run.py --config-name su_aug_addc`  
`python run.py --config-name run_data_aug`

We provide various options for augmentations, including sampling from a pool of augmentations `data_aug.num_samples`, and repeats of augmentations so that each instance will be augmented into several new instances `data_aug.density`

### Contrastive pretraining
We pretrain an MPNN then use the labeled data for supervised finetuning. 

#### Baselines

__GraphCL__: GraphCL generates 2 views with their simple graph data augmentation. It performs graph level contrast, and uses NT-Xent loss. See the paper [GraphCL](https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf) and the official [repo](https://github.com/Shen-Lab/GraphCL), as well as the usage example in [PyGCL](https://github.com/PyGCL/PyGCL/blob/main/examples/GraphCL.py).

Example code: `python graphcl.py exp.datapath=PATH2DATA`

__GCC__: GCC generates views with random walk subgraphs, also graph level contrast and NT-Xent loss. See their paper [GCC](https://arxiv.org/abs/2006.09963) and their official code [repo](https://github.com/THUDM/GCC).

Example code: `python gcc.py`

__InfoGraph__: InfoGraph maximized global-local level mutual information (MI), without data augmentation. A GAN-like loss is applied. See their paper [InfoGraph](https://arxiv.org/abs/1908.01000) and usage example in [PyGCL](https://github.com/PyGCL/PyGCL/blob/main/examples/InfoGraph.py).

Example code: `python infograph.py`

__IGSD__: It is a distillation-based method with a teacher-student network, and uses PPR augmentation. The loss is consistence loss to minimize the corresponding L2 distance z1 <-> z2', z2 <-> z1'. See their paper [IGSD](https://arxiv.org/abs/2010.12609), their wrapped [code](https://openreview.net/forum?id=Z532uNJyG5y).

Example code: `python igsd.py`

__MVGRL__: MVGRL also uses PPR augmentation but performs graph-node level contrast. See their paper [MVGRL](https://arxiv.org/abs/2006.05582) and usage example in [PyGCL](https://github.com/PyGCL/PyGCL/blob/main/examples/MVGRL_graph.py).

Example code: `python mvgrl.py`

__GAE__: Graph autoencoder to reconstruct edges. See implementation in [PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAE.html#torch_geometric.nn.models.GAE).

Example code: `python gae.py`

#### Ours
We can use our data augmentation to generate different views for contrastive learning. We use a combination of our augmentations that do not without touching the solutions of the problems. 

Example code: `python pretrain_finetune.py --config-name pre_fine`

#### Picking the best model
For tuning the contrastive pretraining, we pick the best contrastive configuration, such that its finetuning performance is the best. Consider setting `finetune.whole=true` if you have enough computing resources, such that the whole pretrained models are finetuned, if `false`, then it performs linear probing, i.e., freeze the first layers and train a linear layer predictor head only. 

### Generalization
We provide more flexible finetuning pipeline with given pretrained model weights. Given a folder `PATH2MODEL` containing your pretrained models, just run

Example code: `python finetune.py exp.datapath=PATH2DATA finetune.modelpath=PATH2MODEL`

Example pretrained weights can be found in `pretrained_models`.

In `pretrained_models` we provide a number of pretrained model weights on random instances, including contrastive pretraining, supervised learning with/without augmentations. All pretrained with the full training set.

### QPLIB

We pick LCQP instances from QPLIB that feasible and fits into the memory. Due to the extreme scarce data and high size and distribution heterogeneity, it is infeasible to do normal train validation split on QPLIB. We provide two ways probing QPLIB.
1. We compare training QPLIB from scratch and using a contrastive pretrained model trained on the random QP instances, and show the training convergence. `python run_qplib.py`.
2. We enrich the dataset, by perturbing the coefficients. Then we compare supervised learning with/without augmentation. `python run_qplib_enrich.py`.
