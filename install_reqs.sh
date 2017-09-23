#! /bin/bash

# Required by rdkit
sudo apt-get install -y libxmlrender-dev

conda env create -n pubchem-sklearn
source activate pubchem-sklearn
conda install -y -q -c rdkit rdkit=2017.03.3
conda install -c glemaitre imbalanced-learn
conda install -c conda-forge keras
conda install -c anaconda tensorflow-gpu
