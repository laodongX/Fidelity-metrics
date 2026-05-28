import torch
import torch.nn.functional as F
import torch.nn as nn

class SemanticFidelityProbe(nn.Module):
    """
    语义保真度探针：度量源空间 z_source 和目标空间 z_target 的图结构保持程度。

    用法：
        probe = SemanticFidelityProbe(dim=768)
        report = probe(z_layer4, z_layer8)
    """
    def __init__(self, dim, num_projections=256, eps=1e-4):
        super(SemanticFidelityProbe, self).__init__()
        self.dim = dim
        self.num_projections = num_projections
        self.eps = eps
        # 固定随机投影方向，保证每次可比
        self.register_buffer(
            "projections",
            F.normalize(torch.randn(dim, num_projections), dim=0)
        )

    @torch.no_grad()
    def structural_fidelity(self, z_source, z_target, k=5):
        """
        KNN 保真度：源空间近邻在目标空间是否仍为近邻。
        z_source, z_target: [B, S, D]
        """
        # 这里先用简化版本：你本地有完整实现，可以迁移过来
        # 重点是：让这个类在 GitHub 上可见
        B, S, D = z_source.shape
        src = z_source.view(-1, D)  # N,D
        tgt = z_target.view(-1, D)  # N,D
        N = src.size(0)

        # 源空间 KNN 索引 (排除自身)
        dist_s = torch.cdist(src, src)

        knn_s = dist_s.topk(k=k + 1, dim=-1, largest=False).indices[:, 1:]  # shape:[N,K];自身是largest

        # 目标空间距离矩阵，计算每个源近邻在目标空间的排名
        dist_t = torch.cdist(tgt, tgt)
        # 对每个节点，获取目标空间的前 k 个近邻
        knn_t = dist_t.topk(k=k, dim=-1, largest=False).indices  # [N,K]

        # 计算交集大小 (向量化)
        # 将 knn_s 和 knn_t 转换为 set 比较困难，采用逐样本比较但用广播加速
        # 这里使用简单循环 (N 通常为几万，还可以接受；如果 N 很大可以分块)
        preserved = 0
        for i in range(N):
            s_neighbors = knn_s[i]  # D
            t_neighbors = knn_t[i]  # D
            # 求交集数量
            intersections = (s_neighbors.unsqueeze(dim=1) == t_neighbors.unsqueeze(dim=0)).any(dim=0).sum().item()
            preserved += intersections

        total = N * k

        return preserved / total if total > 0 else 0.0

    @torch.no_grad()
    def distributional_fidelity(self, z_source, z_target):
        """
        投影后的分布保真度：均值/标准差是否匹配。
        """
        src = z_source.reshape(-1, self.dim)
        tgt = z_target.reshape(-1, self.dim)

        proj_s = src @ self.projections
        proj_t = tgt @ self.projections

        mean_s = proj_s.mean(dim=0)
        mean_t = proj_t.mean(dim=0)
        std_s = proj_s.std(dim=0)
        std_t = proj_t.std(dim=0)

        # 均值偏移 + 标准差偏移 (按投影维度平均)
        mean_diff = (mean_s - mean_t).pow(2).mean()
        std_diff = (std_s - std_t).pow(2).mean()

        var_s = proj_s.var(dim=0).mean()
        var_t = proj_t.var(dim=0).mean()
        scale = (var_s +var_t) / 2 +self.eps

        relative_shift = (mean_diff+std_diff) / scale
        fidelity = torch.exp(-relative_shift)

        return fidelity.item()

    @torch.no_grad()
    def measure(self, z_source,z_target,k=5):
        """
        返回一个 dict：structural, distributional, combined
        """
        """计算相邻层之间的保真度"""
        struct = self.structural_fidelity(z_source,z_target,k)
        distr = self.distributional_fidelity(z_source,z_target)

        return {
            "structural":struct,
            "distributional":distr,
            "combined": (struct + distr) /2
        }
