import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import sys
sys.path.append('./task/model_block/')
from EGNN import eg
import warnings

warnings.filterwarnings("ignore")


class KD_EGNN(nn.Module):
	def __init__(self, infeature_size, outfeature_size, nhidden_eg, edge_feature,
				n_eglayer, nclass):
		super(KD_EGNN,self).__init__()
		self.dropout = 0.3
		self.eg1 = eg(in_node_nf=infeature_size,
					nhidden=nhidden_eg,
					n_layers=n_eglayer,
					out_node_nf=outfeature_size,
					in_edge_nf=edge_feature,
					attention=True,
					normalize=False,
					tanh=True)
		self.eg2 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=outfeature_size,
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True)
		self.eg3 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 2),
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True)
		self.eg4 = eg(in_node_nf=int(outfeature_size / 2),
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 4),
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True)
		self.fc1 = nn.Sequential(
			nn.Linear(outfeature_size, int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc2 = nn.Sequential(
			nn.Linear(outfeature_size, int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc3 = nn.Sequential(
			nn.Linear(int(outfeature_size / 2), int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc4 = nn.Sequential(
			nn.Linear(int(outfeature_size / 4), int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)

	def forward(self,x_res,x_pos,edge_index):

		x_res = F.dropout(x_res, self.dropout, training=self.training)
		output_res, pre_pos_res = self.eg1(h=x_res,
										x=x_pos.float(),
										edges=edge_index,
										edge_attr=None)

		output_res2, pre_pos_res2 = self.eg2(h=output_res,
											x=pre_pos_res.float(),
											edges=edge_index,
											edge_attr=None)

		output_res3, pre_pos_res3 = self.eg3(h=output_res2,
											x=pre_pos_res2.float(),
											edges=edge_index,
											edge_attr=None)

		output_res4, pre_pos_res4 = self.eg4(h=output_res3,
											x=pre_pos_res3.float(),
											edges=edge_index,
											edge_attr=None)
		out1 = self.fc1(output_res)
		out2 = self.fc2(output_res2)
		out3 = self.fc3(output_res3)
		out4 = self.fc4(output_res4)

		return [out4,out3,out2,out1],[output_res4,output_res3,output_res2,output_res]

# class DenoiseLayer(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(dim+3, dim),
#             nn.SiLU(),
#             nn.Linear(dim, 3)  # 输出坐标修正量
#         )
    
#     def forward(self, x, pos):
#         delta_pos = self.mlp(torch.cat([x, pos], dim=-1))
#         return pos + delta_pos

class KD_EGNN_edge(nn.Module):
	def __init__(self, infeature_size, outfeature_size, nhidden_eg, edge_feature_size,
				n_eglayer, nclass):
		super(KD_EGNN_edge,self).__init__()
		self.dropout = 0.3
        
		self.eg1 = eg(in_node_nf=infeature_size,
					nhidden=nhidden_eg,
					n_layers=n_eglayer,
					out_node_nf=outfeature_size,
					in_edge_nf=int(outfeature_size / 4),
					attention=True,
					normalize=False,
					tanh=True)
		self.eg2 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=outfeature_size,
					  in_edge_nf=int(outfeature_size / 4),
					  attention=True,
					  normalize=False,
					  tanh=True)
		self.eg3 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 2),
					  in_edge_nf=int(outfeature_size / 4),
					  attention=True,
					  normalize=False,
					  tanh=True)
		self.eg4 = eg(in_node_nf=int(outfeature_size / 2),
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 4),
					  in_edge_nf=int(outfeature_size / 4),
					  attention=True,
					  normalize=False,
					  tanh=True)
          
		self.edge_fc = nn.Linear(edge_feature_size, int(outfeature_size/4))
          
		
		# self.denoise_layer1= DenoiseLayer(outfeature_size)
		# self.denoise_layer2= DenoiseLayer(outfeature_size)
		# self.denoise_layer3= DenoiseLayer(int(outfeature_size / 2))
        
		self.fc1 = nn.Sequential(
			nn.Linear(outfeature_size, int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc2 = nn.Sequential(
			nn.Linear(outfeature_size, int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc3 = nn.Sequential(
			nn.Linear(int(outfeature_size / 2), int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc4 = nn.Sequential(
			nn.Linear(int(outfeature_size / 4), int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)

	def forward(self,x_res,x_pos,edge_feat,edge_index):
		edge_feat = self.edge_fc(edge_feat)

		x_res = F.dropout(x_res, self.dropout, training=self.training)
		output_res, pre_pos_res = self.eg1(h=x_res,
										x=x_pos.float(),
										edges=edge_index,
										edge_attr=edge_feat)
		# pre_pos_res = self.denoise_layer1(output_res, pre_pos_res)

		output_res2, pre_pos_res2 = self.eg2(h=output_res,
											x=pre_pos_res.float(),
											edges=edge_index,
											edge_attr=edge_feat)
		# pre_pos_res2 = self.denoise_layer2(output_res2, pre_pos_res2)

		output_res3, pre_pos_res3 = self.eg3(h=output_res2,
											x=pre_pos_res2.float(),
											edges=edge_index,
											edge_attr=edge_feat)
		# pre_pos_res3 = self.denoise_layer3(output_res3, pre_pos_res3)

		output_res4, pre_pos_res4 = self.eg4(h=output_res3,
											x=pre_pos_res3.float(),
											edges=edge_index,
											edge_attr=edge_feat)
		out1 = self.fc1(output_res)
		out2 = self.fc2(output_res2)
		out3 = self.fc3(output_res3)
		out4 = self.fc4(output_res4)

		return [out4,out3,out2,out1],[output_res4,output_res3,output_res2,output_res]



class Expert(nn.Module):
    """单专家网络模块"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    """混合专家模块（含负载均衡损失实现）"""
    def __init__(self, n_experts, input_dim, output_dim, k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(n_experts)])
        self.gate = nn.Linear(input_dim, n_experts)
        self.k = k
        self.n_experts = n_experts
        self.output_dim = output_dim
        self.aux_loss = torch.tensor(0.0)  # 初始化辅助损失
        
    def _load_balancing_loss(self, gate_scores, topk_idx):
        """
        负载均衡损失计算
        Args:
            gate_scores: 原始门控分数 [batch_size, n_experts]
            topk_idx: 选中的专家索引 [batch_size, k]
        Returns:
            loss: 标量损失值
        """
        # 1. 计算专家使用频率
        batch_size = gate_scores.size(0)
        mask = torch.zeros_like(gate_scores).scatter(-1, topk_idx, 1)  # [batch_size, n_experts]
        expert_usage = mask.float().mean(dim=0)  # [n_experts]
        
        # 2. 计算理想均匀分布
        uniform_dist = torch.ones_like(expert_usage) / self.n_experts
        
        # 3. 计算分布差异（使用均方差）
        load_balance_loss = F.mse_loss(expert_usage, uniform_dist)
        
        # 4. 添加门控分数多样性正则项
        gate_entropy = torch.distributions.Categorical(logits=gate_scores).entropy().mean()
        diversity_loss = -0.1 * gate_entropy  # 鼓励门控分布多样性
        
        return load_balance_loss + diversity_loss

    def forward(self, x):
        # 门控计算
        gate_scores = self.gate(x)  # [batch_size, n_experts]
        topk_val, topk_idx = torch.topk(gate_scores, self.k, dim=-1)
        
        # 计算负载均衡损失
        aux_loss = self._load_balancing_loss(gate_scores, topk_idx)
        
        # 稀疏化处理
        mask = torch.zeros_like(gate_scores).scatter(-1, topk_idx, 1)
        gate_scores = gate_scores * mask
        gate_weights = F.softmax(topk_val, dim=-1)
        
        # 专家结果融合
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, n_experts, output_dim]
        selected_experts = torch.gather(expert_outputs, 1, 
                                      topk_idx.unsqueeze(-1).expand(-1, -1, self.output_dim))
        return (selected_experts * gate_weights.unsqueeze(-1)).sum(dim=1),aux_loss


class kd_egnn(nn.Module):
    def __init__(self, infeature_size, outfeature_size, nhidden_eg, edge_feature_size, n_eglayer, nclass,n_experts=4):
        super(kd_egnn, self).__init__()
        self.eg1 = eg(in_node_nf=infeature_size,
					nhidden=nhidden_eg,
					n_layers=n_eglayer,
					out_node_nf=outfeature_size,
					in_edge_nf=int(outfeature_size / 4),
					attention=True,
					normalize=False,
					tanh=True)
        self.eg2 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=outfeature_size,
					  in_edge_nf=int(outfeature_size / 4),
					  attention=True,
					  normalize=False,
					  tanh=True)
        self.eg3 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 2),
					  in_edge_nf=int(outfeature_size / 4),
					  attention=True,
					  normalize=False,
					  tanh=True)
        self.eg4 = eg(in_node_nf=int(outfeature_size / 2),
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 4),
					  in_edge_nf=int(outfeature_size / 4),
					  attention=True,
					  normalize=False,
					  tanh=True)
        self.edge_fc = nn.Linear(edge_feature_size, int(outfeature_size/4))
        
        self.moe1 = MoE(n_experts, outfeature_size, int(outfeature_size/4))
        self.moe2 = MoE(n_experts, outfeature_size, int(outfeature_size/4))
        self.moe3 = MoE(n_experts, int(outfeature_size/2), int(outfeature_size/4))
        self.moe4 = MoE(n_experts, int(outfeature_size/4), int(outfeature_size/4))
        self.fc1 = nn.Linear(int(outfeature_size/4), nclass)
        self.fc2 = nn.Linear(int(outfeature_size/4), nclass)
        self.fc3 = nn.Linear(int(outfeature_size/4), nclass)
        self.fc4 = nn.Linear(int(outfeature_size/4), nclass)
    def forward(self,x_res,x_pos,edge_feat,edge_index):
        x_res = F.dropout(x_res, 0.2, training=self.training)
        edge_feat = self.edge_fc(edge_feat)
        # gate_probs = self.moe_gate(x_res)
        output_res, pre_pos_res = self.eg1(h=x_res,
                                    x=x_pos.float(),
                                    edges=edge_index,
                                    edge_attr=edge_feat)
        output_res2, pre_pos_res2 = self.eg2(h=output_res,
                                            x=pre_pos_res.float(),
                                            edges=edge_index,
                                            edge_attr=edge_feat)
        output_res3, pre_pos_res3 = self.eg3(h=output_res2,
                                            x=pre_pos_res2.float(),
                                            edges=edge_index,
                                            edge_attr=edge_feat)
        output_res4, pre_pos_res4 = self.eg4(h=output_res3,
                                            x=pre_pos_res3.float(),
                                            edges=edge_index,
                                            edge_attr=edge_feat)
        out1,aux_loss1 = self.moe1(output_res)
        out2,aux_loss2 = self.moe2(output_res2)
        out3,aux_loss3 = self.moe3(output_res3)
        out4,aux_loss4 = self.moe4(output_res4)
        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        out3 = self.fc3(out3)
        out4 = self.fc4(out4)
		# 计算负载均衡损失
        balance_loss = sum([aux_loss1,aux_loss2,aux_loss3,aux_loss4])/4


        return [out4,out3,out2,out1],[output_res4,output_res3,output_res2,output_res,balance_loss]

