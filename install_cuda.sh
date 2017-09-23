#! /bin/bash

echo "Installing CUDA requirements"
cd ~

sudo apt-get install -y build-essential

~/NVIDIA-Linux-x86_64-384.69.run --silent

echo "Installing CUDA"

~/cuda_8.0.61_375.26_linux.run --toolkit --silent

echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc


echo 'Installing cuDNN'
cp ~/cuda/include/* /usr/local/cuda-8.0/include/
cp ~/cuda/lib64/* /usr/local/cuda-8.0/lib64

source ~/.bashrc
