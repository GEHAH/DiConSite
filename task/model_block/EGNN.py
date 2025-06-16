import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import warnings
warnings.filterwarnings("ignore")
# E_GCL 类是 PyTorch 中的神经网络模块。
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    input_nf:输入点特征的大小
    edges_in_d: 边特征的大小
    hidden_nf: 隐藏层的特征长度
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),#节点特征n*2+径向距离额+外边特征
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        """
        :param source: source 参数表示图中边的起始节点。它是边的起源节点。
        :param target: target参数表示边的目标节点。在图中，边连接两个节点，目标节点是边所指向的节点。
        :param radial: “径向”参数可能是表示边缘的距离或大小的数值。它可用于确定图形或网络中源节点和目标节点之间的边的强度或权重。
        :param edge_attr: `edge_attr` 参数用于指定与源节点和目标节点之间的边关联的任何附加属性或特征。这些属性可以提供有关节点之间关系的附加信息，并且可以用作边缘模型的输入。
        """
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index #具有联系的边的索引
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        # 函数的主要目的是计算给定坐标之间的径向距离和归一化后的坐标差
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h,edge_index,coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        #给定坐标之间的径向距离和归一化后的坐标差

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        #得到新的边的特征
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        # 不使用coord更新，注释掉coord

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_feat


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

class eg(nn.Module):
    def __init__(self,in_node_nf, nhidden, out_node_nf, in_edge_nf=0,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, attention=True,
                 normalize=False, tanh=False):
        super(eg, self).__init__()
        self.nhidden = nhidden
        # self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.nhidden)
        self.embedding_out = nn.Linear(self.nhidden, out_node_nf)
        for i in range(0,n_layers):
            self.add_module('eg_%d' % i,E_GCL(self.nhidden, self.nhidden,self.nhidden,edges_in_d=in_edge_nf,
                                              act_fn=act_fn, residual=residual, attention=attention,
                                              normalize=normalize,tanh=tanh))
        # self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["eg_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)

        return h, x