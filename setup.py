from setuptools import setup, find_packages

setup(
    name='transpropy',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "setuptools",
        "scikit-learn",
        "tqdm"
    ],
    url='https://github.com/SSSYDYSSS/TransProPy',
    author='Yu Dongyue',
    author_email='yudongyue@mail.nankai.edu.cn',
    description='A collection of deep learning models that integrate algorithms and various machine learning approaches to extract features (genes) effective for classification and attribute them accordingly.'
)