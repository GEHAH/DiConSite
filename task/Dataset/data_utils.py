import numpy as np
import torch
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import os

def get_pdb_xyz1(pdb_file):
    """
    get the coordinates
    """
    with open(pdb_file,'r') as f:
        lines = f.readlines()
    current_pos = -1000
    X = []
    current_aa = {} # N, CA, C, O, R
    for line in lines:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                R_group = []
                for atom in current_aa:
                    if atom not in ["N", "CA", "C", "O"]:
                        R_group.append(current_aa[atom])
                if R_group == []:
                    R_group = [current_aa["CA"]]
                R_group = np.array(R_group).mean(0)
                X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return torch.tensor(X)

def get_pdb_xyz(pdb_file):
    rec_path = pdb_file
    biopython_parser = PDBParser()
    
    # 使用上下文管理器来捕获警告，忽略 PDBConstructionWarning 警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        # 通过 Biopython 的解析器加载 PDB 结构，'random_id' 是结构的临时 ID
        structure = biopython_parser.get_structure('random_id', rec_path)
        # 选择第一个模型，通常是PDB文件主要模型
        rec = structure[0]

    coords = []  # 存储所有链的原子坐标
    c_alpha_coords = []  # 存储每个残基的 Cα 原子坐标
    n_coords = []  # 存储每个残基的 N 原子坐标
    c_coords = []  # 存储每个残基的 C 原子坐标
    o_coords = [] #储存每个残基的O原子坐标
    r_coords = [] #存储每个侧链原子（Sidechain）的质心
    valid_chain_ids = []  # 存储有效链的 ID
    lengths = []  # 存储每条链中有效残基的数量
    sequences = []  # 存储每条链的氨基酸序列
    X = []  # 存储所有链的坐标数据
    # 遍历结构中的每一条链
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        chain_o_coords = []
        chain_r_coords = []
        chain_sequence = []  # 存储该链的氨基酸序列
        count = 0
        invalid_res_ids = []  # 储存无效残基的ID
        
        # 遍历链中每一个残基
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':  # 如果残基是水分子(HOH)，则跳过
                invalid_res_ids.append(residue.get_id())
                continue

            residue_coords = []  # 存储当前残基的原子坐标
            c_alpha, n, c,o = None, None, None,None  # 初始化 Cα、N 和 C 原子的坐标
            R_coord = []
            for atom in residue:  # 遍历残基中的每个原子
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                if atom.name == 'O':
                    o = list(atom.get_vector())
                if atom.name not in ["N", "CA", "C", "O"]:
                    R_coord.append(list(atom.get_vector()))
                residue_coords.append(list(atom.get_vector()))
            if R_coord:
                R_coords = np.array(R_coord).mean(axis=0)
            else:
                R_coords = c_alpha
            if o is None:
                o = c_alpha
            
            # 仅在残基有效（含 Cα、N 和 C 原子）时添加到链的坐标列表
            if c_alpha is not None and n is not None and c is not None:
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_o_coords.append(o)
                # if R_coord == []:
                #     chain_r_coords.append(c_alpha)
                # else:
                chain_r_coords.append(R_coords)
                # chain_r_coords.append(R_coord)
                chain_coords.append(np.array(residue_coords))
                chain_sequence.append(residue.get_resname())  # 记录当前残基的氨基酸名
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())  # 记录无效残基 ID

        # 从链中移除无效残基
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)

        lengths.append(count)  # 添加有效残基数量到 lengths 列表
        coords.append(chain_coords)  # 添加链的坐标列表
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        o_coords.append(np.array(chain_o_coords))
        r_coords.append(np.array(chain_r_coords))
        sequences.append(chain_sequence)  # 保存该链的氨基酸序列
        if len(chain_coords) > 0:
            valid_chain_ids.append(chain.get_id())  # 如果链中有有效坐标，则记录该链 ID

    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_o_coords = []
    valid_r_coords = []

    valid_lengths = []
    invalid_chain_ids = []
    valid_sequences = []  # 存储有效链的氨基酸序列

    # 仅保留有效链的坐标、序列和长度
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_o_coords.append(o_coords[i])
            valid_r_coords.append(r_coords[i])
            valid_lengths.append(lengths[i])
            valid_sequences.append(sequences[i])  # 添加有效链的序列
        else:
            invalid_chain_ids.append(chain.get_id())

    # list with n_residues arrays: [n_atoms, 3]
    coords = [item for sublist in valid_coords for item in sublist]
    # 将有效 Cα、N 和 C 坐标合并为数组
    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    o_coords = np.concatenate(valid_o_coords, axis=0)  # [n_residues, 3]
    r_coords = np.concatenate(valid_r_coords, axis=0)  # [n_residues, 3]

    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    X.append([n_coords, c_alpha_coords, c_coords, o_coords, r_coords])
    return torch.tensor(X).squeeze(0).permute(1, 0, 2)
