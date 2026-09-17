from setuptools import setup, find_packages

setup(
    name='defmuon',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'transformers',
        'datasets',
        'tiktoken',
        'tqdm',
        'wandb',
        'hydra-core',
    ],
)