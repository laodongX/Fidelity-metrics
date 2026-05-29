import torch
import torch.nn as nn
from fidelity_metrics.probe import SemanticFidelityProbe
from fidelity_metrics.hook import LayerwiseFidelityHook

x = torch.randn((2,64,128))
class Modeltest(nn.Module):
    def __init__(self):
        super(Modeltest, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.lineart = nn.Linear(128, 128, bias=False)

    def forward(self,x):
        x = self.linears(x)
        output = self.lineart(x)
        return output

model = Modeltest()
se_probe = SemanticFidelityProbe(dim=128,num_projections=64)
layerhook = LayerwiseFidelityHook(model=model,probe=se_probe)
names= []
for name,m in model.named_modules():
    names.append(name)

layerhook.register_hooks(layer_names=[names[2],names[3]])
with torch.no_grad():
    output = model(x)

report = layerhook.get_report()
for r in report:
    print(f"  {r['layers']}: structural={r['structural']:.4f}, distributional={r['distributional']:.4f}, combined={r['combined']:.4f}")

layerhook.remove_hooks()
