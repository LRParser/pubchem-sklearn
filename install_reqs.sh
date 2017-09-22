#! /bin/bash
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
chmod +x Anaconda3-4.4.0-Linux-x86_64.sh
./Anaconda3-4.4.0-Linux-x86_64.sh
conda install -y -c mcs07 pubchempy
conda install -y -q -c rdkit rdkit=2017.03.3
conda install -c glemaitre imbalanced-learn
conda install -c conda-forge keras
conda install -c anaconda tensorflow-gpu
