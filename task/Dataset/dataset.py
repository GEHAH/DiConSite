import os
import sys

import numpy as np
import pandas as pd
import torch

# sys.path.append('')
import torch.nn.functional as F
import torch_geometric
sys.path.append('./task/Dataset')
from data_utils import get_pdb_xyz
from torch.utils.data import Dataset


class PPISDataset(Dataset):
    def __init__(self,esm_path,pdb_path,dataset_root):
        super(PPISDataset).__init__()
        self.esm_path = esm_path
        # self.dssp_path = dssp_path
        self.pdb_path = pdb_path
        self.dataset_root = dataset_root
        self.IDs,self.Sequences,self.Labels = self.get_isl(self.dataset_root)
    
    def __getitem__(self, index):
        pdb_ID = self.IDs[index]
        sequence = self.Sequences[index]
        label = torch.from_numpy(np.array(self.Labels[index]))

        X_coors = get_pdb_xyz(os.path.join(self.pdb_path,pdb_ID+".pdb"))
        X_ca = X_coors[:,1]

        edge_index = torch_geometric.nn.radius_graph(X_ca, r=14, loop=True, max_num_neighbors=1000)[[1,0]].long()
        # dssp_feat = torch.load(os.path.join(self.dssp_path,pdb_ID+".tensor"))
        esm_feat = pd.read_pickle(os.path.join(self.esm_path,pdb_ID+".pkl"))
        # esm_feat = torch.load(os.path.join(self.esm_path,pdb_ID+".pkl"),map_location='cpu')
        # node_angle = self._get_angle(X_coors)
        edge_feat = self._get_eage_features(X_coors,edge_index)
        # node_feat = torch.cat([dssp_feat,node_angle],dim=-1)
        data = torch_geometric.data.Data(name=pdb_ID,sequence=sequence,X_ca=X_ca,edge_index=edge_index,esm_feat=esm_feat,edge_feat=edge_feat,label=label)
        return data

    def __len__(self):
        return len(self.IDs)

    def get_isl(self,data_path):
        protein_data = pd.read_pickle(data_path)
        ids,sequneces,labels = [],[],[]
        for id in protein_data:
            # id_new = id.strip()
            if id not in  ['3j7y_S','3jb9_E','6exn_O','7b9v_h','1ned_ABCDEFGHIJKL','1n25_ABCDEF','1svm_ABCDEF','1xmf_ABCDEF','3gi0A','4oz6A']:
                ids.append(id)
                sequneces.append(protein_data[id][0])
                labels.append(list(map(int,protein_data[id][1])))
        return ids,sequneces,labels
    
    def _get_angle(self,X, eps=1e-7):
        """
        get the angle features
        """
        # psi, omega, phi
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
        D = torch.reshape(D, [-1, 3])
        dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

        # alpha, beta, gamma
        cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.acos(cosD)
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

        node_angles = torch.cat((dihedral, bond_angles), 1)
        return node_angles # dim = 12
    def _get_eage_features(self, X, edge_index):

        edge_dist = get_distance(X, edge_index)
        edge_direction, edge_orientation = get_direction_orientation(X, edge_index)
        edge_features = torch.cat([edge_dist, edge_direction, edge_orientation], dim=-1)
        return edge_features
    

def get_distance(X, edge_index):
        """
        get the distance features
        """
        atom_N = X[:, 0]  # [L, 3]
        atom_Ca = X[:, 1]
        atom_C = X[:, 2]
        atom_O = X[:, 3]
        atom_list = ["N", "Ca", "C", "O"]
        edge_dist = []
        for atom1 in atom_list:
            for atom2 in atom_list:
                E_vectors = eval(f'atom_{atom1}')[edge_index[0]] - eval(f'atom_{atom2}')[edge_index[1]]
                rbf = _rbf(E_vectors.norm(dim=-1))
                edge_dist.append(rbf)
        edge_dist = torch.cat(edge_dist, dim=-1)  # dim = [E, 16 * 16]

        return edge_dist
def _rbf(D, D_min=0., D_max=20., D_count=16):
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_direction_orientation(X, edge_index): 
    """
    get the direction features
    """
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return edge_direction, edge_orientation

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
        Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q
