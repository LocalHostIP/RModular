# RModular
Robot manipulador

# Apuntes
1234

# Instalar darknet

git clone https://github.com/AlexeyAB/darknet

cd darknet

mkdir build_release

cd build_release

cmake ..
cmake --build . --target install --parallel 8

## reinstalar cmake
sudo apt remove cmake

Visit https://cmake.org/download/ and download the latest bash script.

chmod +x /opt/cmake-3.*your_version*.sh

sudo bash /opt/cmake-3.*your_version*.sh

You will need to press y twice.

The script installs the binary to /opt/cmake-3.*your_version* so in order to get the cmake command, make a symbolic link:

sudo ln -s /opt/cmake-3.*your_version*/bin/* /usr/local/bin
Test your results with:

cmake --version

https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu

# Links
Usar cualquier versión de YOLO con Darknet

cfgs
https://github.com/AlexeyAB/darknet/tree/master/cfg

Pesos
https://github.com/AlexeyAB/darknet/releases/tag/yolov4


