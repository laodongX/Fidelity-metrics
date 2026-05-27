from setuptools import setup, find_packages

setup(
    name="fidelity-metrics",
    version="0.1.0",
    author="laodongX",         # 换成你的名字
    description="Semantic Fidelity Metrics for LLM/VLM Training: Measuring Information Loss Across Layers",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)