import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from omegaconf import DictConfig, OmegaConf

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from models.hetero_encoder import TripartiteHeteroEncoder
from trainer import PlainGNNTrainer
from data.utils import save_run_config
from sklearn.linear_model import Ridge, Lasso, LinearRegression


@hydra.main(version_base=None, config_path='./config', config_name="finetune")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    train_set = LPDataset(args.datapath, 'train', transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.train_frac < 1:
        train_set = train_set[:int(len(train_set) * args.train_frac)]
    valid_set = LPDataset(args.datapath, 'valid', transform=GCNNorm() if 'gcn' in args.conv else None)
    test_set = LPDataset(args.datapath, 'test', transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_set,
                              batch_size=args.batchsize,
                              shuffle=False,
                              collate_fn=collate_fn_lp_base)
    val_loader = DataLoader(valid_set,
                            batch_size=args.val_batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base)
    test_loader = DataLoader(test_set,
                             batch_size=args.val_batchsize,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base)

    for run in range(args.runs):
        model = TripartiteHeteroEncoder(conv=args.conv,
                                        head=args.gat.heads,
                                        concat=args.gat.concat,
                                        hid_dim=args.hidden,
                                        num_encode_layers=args.num_encode_layers,
                                        num_conv_layers=args.num_conv_layers,
                                        num_pred_layers=args.num_pred_layers,
                                        num_mlp_layers=args.num_mlp_layers,
                                        norm=args.norm).to(device)

        model.load_state_dict(torch.load(args.modelpath, map_location=device))
        model.eval()

        features = []
        labels = []
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                obj_pred = model(data).cpu().numpy()
                label = data.obj_solution.cpu().numpy()
                features.append(obj_pred)
                labels.append(label)

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        val_features = []
        val_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                obj_pred = model(data).cpu().numpy()
                label = data.obj_solution.cpu().numpy()
                val_features.append(obj_pred)
                val_labels.append(label)

        val_features = np.concatenate(val_features, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        for weight_decay in [1.e-6, 1.e-5, 1.e-4, 1.e-3, 1.e-2, 0.1]:
            clf = Ridge(alpha=weight_decay)
            clf.fit(features, labels)

            preds = clf.predict(val_features)
            obj_gap = np.abs((preds - val_labels) / val_labels).mean()

            wandb.log({'alpha': weight_decay, 'gap': obj_gap})
            print(f'alpha: {weight_decay}, gap: {obj_gap}')

if __name__ == '__main__':
    main()
