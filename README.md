# SSL for L2O

useful materials [SSL survey](https://github.com/LirongWu/awesome-graph-self-supervised-learning), [PyGCL](https://github.com/PyGCL/PyGCL), [PyTorch metric learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)

## Environment setup

```angular2html
conda create -y -n ipmgnn python=3.11
conda activate ipmgnn
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.5.3  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_cluster-1.6.3%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install wandb seaborn matplotlib cplex ortools hydra-core
pip install pytorch-metric-learning

# the next are only if you want to evaluate solvers
conda install -y -c conda-forge qpsolvers 
# for larger problems you need a license, please visit https://www.gurobi.com/ for more information
pip install gurobipy
```

## Baselines

### GraphCL
Graph level contrast. See the [paper](https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf) and the official [repo](https://github.com/Shen-Lab/GraphCL), as well as the usage example [GraphCL](https://github.com/PyGCL/PyGCL/blob/main/examples/GraphCL.py)

Node dropping: `python graphcl.py exp.datapath=PATH2DATASET pretrain.method.GraphCLDropNode.strength=0.1 pretrain.method.GraphCLPerturbEdge.strength=0. pretrain.method.GraphCLMaskNode.strength=0.`  
Edge perturbation: `python graphcl.py exp.datapath=PATH2DATASET pretrain.method.GraphCLDropNode.strength=0. pretrain.method.GraphCLPerturbEdge.strength=0.1 pretrain.method.GraphCLMaskNode.strength=0.`  
Noe masking: `python graphcl.py exp.datapath=PATH2DATASET pretrain.method.GraphCLDropNode.strength=0. pretrain.method.GraphCLPerturbEdge.strength=0. pretrain.method.GraphCLMaskNode.strength=0.2`

Combination: `python graphcl.py exp.datapath=PATH2DATASET pretrain.method.GraphCLDropNode.strength=0.2 pretrain.method.GraphCLPerturbEdge.strength=0.15 pretrain.method.GraphCLMaskNode.strength=0.2`

GCC: rw subgraph, graph level CL

InfoGraph: global-local MI, no augmentation. GAN like loss

IGSD: distillation, teacher-student, PPR augmentation. the loss is consistence loss. There's a predictor, minimize the corresponding L2 distance z1 <-> z2', z2 <-> z1'

GAE: reconstruct edges
