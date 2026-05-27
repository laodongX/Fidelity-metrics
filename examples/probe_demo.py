from fidelity_metrics.probe import SemanticFidelityProbe
import torch
import torch.nn as nn

def linear_example_one():
    print("""linear_example++++++++++""")
    B,S,D = 2,64,128
    x = torch.randn(B,S,D)
    z_s_layer = nn.Linear(D,D*2)
    z_t_layer = nn.Linear(D*2,D)
    z = z_s_layer(x)
    x_t = z_t_layer(z)
    # z_s_layer.weight = torch.randn(D,D*2,dtype=torch.long)
    # z_t_layer.weight = torch.randn(D*2,D,dtype=torch.long)


    z_probe = SemanticFidelityProbe(dim=128,num_projections=64)
    result_fidelity = z_probe.measure(z_layers=[x,x_t])
    print(result_fidelity)
    for i in result_fidelity:
        # 诊断结论
        if i['combined'] < 0.6:
            print("🚨 诊断结果: 场景2发生严重信息断流，建议检查该层的 Engram 门控或 MoE 路由！")
        else:
            print("✅ 诊断结果: 场景2保真度正常。")
    return result_fidelity


def main():
    print("🚀 启动语义保真度探针演示...\n")

    # 模拟隐空间特征: Batch=2, Seq=64, Dim=512
    B, S, D = 2, 64, 512
    z_source = torch.randn(B, S, D)

    # 场景1: 正常传播 (目标层与源层相似，仅有微小非线性变换)
    z_target_healthy = z_source + torch.randn_like(z_source) * 0.1

    # 场景2: 信息断裂 (目标层加入巨大噪声，模拟表征坍塌/断流)
    z_target_broken = z_source + torch.randn_like(z_source) * 5.0

    # 初始化探针
    probe = SemanticFidelityProbe(dim=D)

    # 测量健康层
    print("📊 场景1: 正常传播 (微小噪声)")
    report_healthy = probe.measure([z_source, z_target_healthy])[0]
    print(f"  结构保真度: {report_healthy['structural']:.4f}")
    print(f"  分布保真度: {report_healthy['distributional']:.4f}")
    print(f"  综合保真度: {report_healthy['combined']:.4f}\n")

    # 测量断裂层
    print("⚠️ 场景2: 信息断裂 (巨大噪声)")
    report_broken = probe.measure([z_source, z_target_broken])[0]
    print(f"  结构保真度: {report_broken['structural']:.4f}")
    print(f"  分布保真度: {report_broken['distributional']:.4f}")
    print(f"  综合保真度: {report_broken['combined']:.4f}\n")

    # 诊断结论
    if report_broken['combined'] < 0.6:
        print("🚨 诊断结果: 场景2发生严重信息断流，建议检查该层的 Engram 门控或 MoE 路由！")
    else:
        print("✅ 诊断结果: 场景2保真度正常。")

if __name__ == "__main__":
    linear_example_one()
    main()