#! /bin/bash

# Required by rdkit
sudo apt-get install -y libxrender-dev

conda env create -n pubchem-sklearn
source activate pubchem-sklearn
conda install -y -q -c rdkit rdkit=2017.03.3
conda install -y -q -c glemaitre imbalanced-learn
conda install -y -q -c conda-forge keras
conda install -y -q -c anaconda tensorflow-gpu
