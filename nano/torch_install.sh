wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl 
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
sudo pip3 install Cython
sudo pip3 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
sudo python3 setup.py install --user
cd ../
sudo pip3 install 'pillow<7'