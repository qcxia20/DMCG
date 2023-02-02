# Basic
import os
import copy
import re
import argparse
import shutil
import pickle
from tqdm import tqdm
from pathlib import Path
# DL
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader
from torch_sparse import SparseTensor
# rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
# DMCG
from confgen.molecule.gt import isomorphic_core
from confgen.molecule.graph import rdk2graph, rdk2graphedge
from confgen.model.gnn import GNN
from confgen.utils.utils import (
    WarmCosine,
    set_rdmol_positions,
    get_best_rmsd,
    evaluate_distance,
)
import multiprocessing

class PygGeomDatasetQC(InMemoryDataset):
    def __init__(
        self,
        mol_list,
        # smi_list,
        folder,
        rdk2graph=rdk2graph,
        transform=None,
        extend_edge=False,
        pre_transform=None,
        seed=None,
        remove_hs=False,
    ):
        self.mol_list = mol_list
        # self.smi_list = smi_list
        self.rdk2graph = rdk2graph
        if seed == None:
            self.seed = 2021
        else:
            self.seed = seed
        if extend_edge:
            self.rdk2graph = rdk2graphedge
        self.folder = folder
        self.remove_hs = remove_hs

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # no error, since download function will not download anything
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if os.path.exists(self.processed_paths[0]):
            return

    def process(self):
        valid_conformation = 0
        bad_case = 0
        data_list = []
        for i,mol in enumerate(tqdm(self.mol_list)):
            if self.remove_hs:
                try:
                    mol = RemoveHs(mol)
                except Exception:
                    continue
            if "." in Chem.MolToSmiles(mol):
                bad_case += 1
                continue
            if mol.GetNumBonds() < 1:
                bad_case += 1
                continue
            graph = self.rdk2graph(mol)
            assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data = CustomData()
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.n_nodes = graph["n_nodes"]
            data.n_edges = graph["n_edges"]
            data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)

            data.rd_mol = copy.deepcopy(mol)
            # data.smi = self.smi_list[i]
            data.isomorphisms = isomorphic_core(mol)

            data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
            data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
            data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
            valid_conformation += 1
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num confs {valid_conformation} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])

class CustomData(Data):
    def __cat_dim__(self, key, value):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask)", key)):
            return -1
        return 0

def generate_confs(model, device, loader, number, useff = False):
    model.eval()
    all_mol_preds = []
    mol_preds = []
    # batched_smis = []
    # mol_labels = []
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        # for smi in batch.smi:
        #     batched_smis.append(smi)
        for p in range(number):
            with torch.no_grad():
                pred, _ = model(batch, sample=False)
            pred = pred[-1]
            batch_size = batch.num_graphs
            n_nodes = batch.n_nodes.tolist()
            pre_nodes = 0
            for i in range(batch_size):
                # mol_labels.append(batch.rd_mol[i]) # The only difference compared with the following block
                mol_pred = set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
                pre_nodes += n_nodes[i]
                mol_preds.append(mol_pred)
        for j in range(batch_size):
            all_mol_preds.append([mol_preds[k] for k in range(j,batch_size*number,batch_size)])
    
    # use FF to perform FF optimization
    if useff:
        ff_all_mol_preds = []
        for gen_mols in all_mol_preds:
            tmplist = []
            for gen_mol in gen_mols:
                gen_mol_c = copy.deepcopy(gen_mol)
                MMFFOptimizeMolecule(gen_mol_c)
                tmplist.append(gen_mol_c)
            ff_all_mol_preds.append(tmplist)
        all_mol_preds = ff_all_mol_preds

    return all_mol_preds
    # return all_mol_preds, batched_smis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--folder", type=str, default="dataset/tmp", help="folder to save processed dataset")
    parser.add_argument("--outpkl", type=str, default="output.pkl", help="pkl file to save generated conformers")
    parser.add_argument("--use-ff", action="store_true", default=False)
    parser.add_argument("--number", type=int,default=10,help="number of conformers generated")
    parser.add_argument("--smitxt", type=str,help="SMILES text file, separated by LF")
    parser.add_argument("--sdffile", type=str,help="SDF file")
    parser.add_argument("--remove-hs", action="store_true", default=False)
    # dataset
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--extend-edge", action="store_true", default=False)
    # model settings
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--reuse-prior", action="store_true", default=False)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--vae-beta", type=float, default=1.0)
    parser.add_argument("--pred-pos-residual", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--shared-decoder", action="store_true", default=False)
    parser.add_argument("--shared-output", action="store_true", default=False)
    parser.add_argument("--sample-beta", type=float, default=1.0)

    parser.add_argument("--eval-from", type=str, default=None)
    args = parser.parse_args()
    # Default
    # os.chdir("/data/git-repo/DMCG")
    setattr(args, "dropout", 0.1)
    setattr(args, "use_bn", True)
    setattr(args, "num_layers", 6)
    setattr(args, "workers", 20)
    setattr(args, "batch_size", 128)
    setattr(args, "reuse_prior", True)
    setattr(args, "node_attn", True)
    setattr(args, "shared_output", True)
    setattr(args, "pred_pos_residual", True)
    setattr(args, "sample_beta", 1.2)
    # setattr(args, "remove_hs", True)
    # setattr(args, "eval_from", "/data/git-repo/DMCG/DMCG/Large_Drugs/checkpoint_94.pt")
    # setattr(args, "use_ff", True)
    # setattr(args, "number", 10)
    # setattr(args, "smitxt", "/data/git-repo/DMCG/smi.txt")
    # setattr(args, "outpkl", "output.pkl")
    # setattr(args, "folder", "dataset/tmp")
    if Path(args.folder).exists():
        shutil.rmtree(args.folder)

    #### From SMILES ####
    # smis = [ i for i in Path(args.smitxt).read_text().split("\n") if i ]
    # mol_list = [ Chem.MolFromSmiles(smi) for smi in smis ]
    # for mol in mol_list:
    #     AllChem.EmbedMolecule(mol)
    #####################

    #### From SDF ####
    supp = Chem.SDMolSupplier(args.sdffile, removeHs=True)
    mol_list = [ mol for mol in supp ]

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    dataset = PygGeomDatasetQC(
        mol_list = mol_list,
        # smi_list = smis,
        folder = args.folder,
        seed=args.seed,
        extend_edge=args.extend_edge,
        remove_hs=args.remove_hs,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # model setting #
    shared_params = {
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_layers": args.mlp_layers,
        "latent_size": args.latent_size,
        "use_layer_norm": args.use_layer_norm,
        "num_message_passing_steps": args.num_layers,
        "global_reducer": args.global_reducer,
        "node_reducer": args.node_reducer,
        "dropedge_rate": args.dropedge_rate,
        "dropnode_rate": args.dropnode_rate,
        "dropout": args.dropout,
        "layernorm_before": args.layernorm_before,
        "encoder_dropout": args.encoder_dropout,
        "use_bn": args.use_bn,
        "vae_beta": args.vae_beta,
        "decoder_layers": args.decoder_layers,
        "reuse_prior": args.reuse_prior,
        "cycle": args.cycle,
        "pred_pos_residual": args.pred_pos_residual,
        "node_attn": args.node_attn,
        "global_attn": args.global_attn,
        "shared_decoder": args.shared_decoder,
        "sample_beta": args.sample_beta,
        "shared_output": args.shared_output,
    }
    model = GNN(**shared_params).to(device)
    if args.eval_from is not None:
        assert os.path.exists(args.eval_from)
        checkpoint = torch.load(args.eval_from, map_location=device)["model_state_dict"]
        cur_state_dict = model.state_dict()
        del_keys = []
        for k in checkpoint.keys():
            if k not in cur_state_dict:
                del_keys.append(k)
        for k in del_keys:
            del checkpoint[k]
        model.load_state_dict(checkpoint)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")
    ################

    # generate confs
    tot_all_mol_preds = generate_confs(
        model = model,
        device = device,
        loader = test_loader,
        number = args.number,
        useff = args.use_ff
    )
    data_dic = dict(zip(list(range(len(tot_all_mol_preds))),tot_all_mol_preds))
    with open(args.outpkl, "wb") as f:
        pickle.dump(data_dic,f)
