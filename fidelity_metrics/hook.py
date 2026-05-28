import torch
from fidelity_metrics.probe import  SemanticFidelityProbe
import torch.nn as nn
from typing import List,Any,Dict
class LayerwiseFidelityHook:
    """

    逐层保真度钩子 — 自动提取模型中间层特征并计算保真度

    用法:
        probe = SemanticFidelityProbe(dim=768)
        hook = LayerwiseFidelityHook(model, probe)
        hook.register_hooks(["blocks.0", "blocks.4", "blocks.8"])

        with torch.no_grad():
            _ = model(val_ids)

        report = hook.get_report()
        hook.remove_hooks()
    """

    def __init__(self,model:nn.Module,probe:SemanticFidelityProbe):
        self.model = model
        self.probe = probe
        self.outputs:Dict[str:Any]= {}
        self.hooks:List[Any] = []
        self._registered_names:List[str] = []

    # ✅ 新版：工厂函数，每个 hook 绑定自己的层名
    def _hook_fn(self, name):  # name 是工厂参数
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.outputs[name] = out.detach().cpu()  # 按名字存！
            elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                self.outputs[name] = out[0].detach().cpu()

        return fn

    def register_hooks(self,layer_names:List[str]):
        """注册 forward hook 到指定层（自动清理旧 hook）"""
        # 1. 安全措施：先移除旧 hook，防止重复注册
        self.remove_hooks()
        self.outputs = {}
        self._registered_names = [ ]
        modules_dict = dict(self.model.named_modules())
        for name in layer_names:
            module = modules_dict.get(name)
            if module is not None:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
                self._registered_names.append(name)

            else:
                print(f"[LayerwiseFidelityHooK Warning] 层 '{name}' 未在模型中找到，已跳过")

    def get_report(self) -> List[Dict]:
        """
            计算所有相邻层的保真度并生成报告
            返回格式: [{'layers': '0->4', 'structural': 0.85, ...}, ...]
        """
        if len(self.outputs)<2:
            print("[LayerwiseFidelityHooK Warning] 捕获的层输出不足2层，无法计算保真度")
            return []

        report = []
        ## 遍历相邻层对，调用 probe.measure
        active_names = [n for n in self._registered_names if n in self.outputs]
        for i in range(len(active_names) - 1) :

            src_name = active_names[i]
            tgt_name = active_names[i+1]
            z_src = self.outputs[src_name]
            z_tgt = self.outputs[tgt_name]

            # 维度检查：跳过不匹配的层对
            if z_src.size(-1) != z_tgt.size(-1):
                print(f"[Warning] {src_name}(dim={z_src.size(-1)}) 和 {tgt_name}(dim={z_tgt.size(-1)}) 维度不同，跳过")
                continue
            if z_src.size(-1) != self.probe.dim:
                print(f"[Warning] {src_name}(dim={z_src.size(-1)}) 与 probe.dim={self.probe.dim} 不匹配，跳过")
                continue

            #处理序列长度不一致的情况，如pooling层导致长度变化
            min_seq = min(z_src.size(1),z_tgt.size(1))
            z_src_aligned = z_src[:,:min_seq,:]
            z_tgt_aligned = z_tgt[:,:min_seq,:]

            #调用核心度量方法
            fidelity_dict = self.probe.measure(z_src_aligned,z_tgt_aligned)
            #组装报告
            src_name = self._registered_names[i] if i < len(self._registered_names) else str(i)
            tgt_name = self._registered_names[i+1] if i+1 < len(self._registered_names) else str(i+1)

            report.append(
                {
                    "layers":f"{src_name}->{tgt_name}",
                    **fidelity_dict
                }
            )
            self.outputs = {}

        return report

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

        self.hooks = []
        self._registered_names = []