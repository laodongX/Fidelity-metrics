from setuptools import setup, find_packages
with open(file="README.md",mode='r',encoding='utf-8') as fh:
    long_description = fh.read()
setup(
    name="fidelity-metrics",
    version="0.1.0",
    author="laodongX",         # 换成你的名字
    long_description=long_description,
    #description="Semantic Fidelity Metrics for LLM/VLM Training: Measuring Information Loss Across Layers",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)