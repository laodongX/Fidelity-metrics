import torch
from .probe import  SemanticFidelityProbe
import torch.nn as nn
class LazywiseFidelityHook:
    """
    逐层保真度钩子 — 自动提取模型中间层特征并计算保真度
    """
    def __init__(self,model:nn.Module,probe:SemanticFidelityProbe):
        self.model = model
        self.probe = probe
        self._outputs = {}
        self._hooks = []