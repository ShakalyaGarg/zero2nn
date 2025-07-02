from setuptools import setup, find_packages

setup(
    name='zero2nn',
    version='0.1.0',
    description='A from-scratch deep learning journey through autodiff, character-level models, and transformers.',
    author='Shakalya Garg',
    packages=find_packages(),  # Automatically includes micrograd, makemore, gpt, shared, etc.
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'graphviz',
        'tqdm',
        'torch'
    ],
    python_requires='>=3.7',
)
