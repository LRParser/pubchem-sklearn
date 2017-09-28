#! /bin/bash

cd ~
source activate pubchem-sklearn

# Required by rdkit
sudo apt-get install -y libxrender-dev

conda env create -n pubchem-sklearn
source activate pubchem-sklearn
conda install -y -q -c rdkit rdkit=2017.03.3
conda install -y -q -c glemaitre imbalanced-learn
conda install -y -q -c conda-forge keras
conda install -y -q -c anaconda tensorflow-gpu

# Uncomment if you want to try using hyperopt
#cd ~
#git clone https://github.com/hyperopt/hyperopt-sklearn.git
#cd hyperopt-sklearn
#pip install -e .

cd ~
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
