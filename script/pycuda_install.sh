export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

apt install libcanberra-gtk-module libcanberra-gtk3-module

apt install  -y python3-pip
apt install -y libpython3.6-dev
apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

source ~/.bashrc
nvcc --version
pip3 install pycuda
# onnx
apt install -y protobuf-compiler libprotoc-dev
pip3 install protobuf==3.9
pip3 install Cython

apt update
apt upgrade python3-numpy
pip3 install 'https://github.com/jetson-nano-wheels/python3.6-numpy-1.19.4/releases/download/v0.0.2/numpy-1.19.4-cp36-cp36m-linux_aarch64.whl'
pip3 install onnx==1.9.0
# pillow
apt install -y libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
pip3 install pillow
