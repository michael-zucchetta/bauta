#!/bin/bash
#
# INSTALL CONDA ENVIRONMENT
#
if ! [ -x "$(command -v conda)" ]; then
  echo 'Error: it was not possible to find anaconda ( conda command ). Please install anaconda or miniconda for your system ( https://conda.io/miniconda.html ). Remember to source .bashrc to enable anaconda' >&2
  exit 1
fi
BAUTA_HOME="$(pwd)"
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    *)          machine="UNKNOWN:${unameOut}"
esac
conda update -n base conda --yes
echo "Installing anaconda environment 'bauta' with all the required dependencies..."
conda create --name bauta python=3.6 --yes
source activate bauta
if ! [ -x "$(command -v nvcc)" ];
then
  echo "CUDA not detected"
  if [ "$machine" == "Mac" ];
  then
    conda install pytorch=0.3.1 torchvision -c pytorch
  else
    conda install pytorch-cpu=0.3.1 torchvision -c pytorch
  fi
else
  CUDA_VERSION="$(nvcc --version --disable-warnings | grep -Po "V(\d)[.0*]*" | awk '{print substr($0,2,1)}')"
  case "$CUDA_VERSION" in
        8)
          conda install --yes pytorch=0.3.1 torchvision -c pytorch
            ;;
        9)
          conda install --yes pytorch=0.3.1 torchvision cuda90 -c pytorch
            ;;
        *)
          conda install --yes pytorch=0.3.1 torchvision -c pytorch
  esac
fi
conda install --yes click
conda install --yes pyyaml
conda install --yes requests
conda install --yes coverage
conda install --yes mock
if [ "$machine" == "Mac" ];
then
  #
  # INSTALL ESSENTIAL DEPENDENCIES
  #
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  sudo xcodebuild -license accept
  brew install wget git cmake pkg-config jpeg libpng libtiff openexr eigen tbb
  #
  # INSTALL PROTOBUF FROM SOURCES TO AVOID PROBLEMS WITH DEPENDENCIES
  #
  cd ~
  git clone git@github.com:google/protobuf.git
  cd protobuf
  git checkout 3.1.x
  ./autogen.sh
  ./configure
  make
  make check
  make install
  sudo update_dyld_shared_cache
  #
  # INSTALL OPENCV3 FROM SOURCES TO AVOID PROBLEMS WITH DEPENDENCIES
  #
  cd ~ && wget -qO- https://github.com/opencv/opencv/archive/3.4.1.tar.gz | tar xvz
  cd opencv-3.4.1
  mkdir release
  cd release
  cmake -DBUILD_TIFF=ON \
      -DBUILD_opencv_java=OFF \
      -DWITH_CUDA=OFF \
      -DBUILD_PROTOBUF=OFF \
      -DPROTOBUF_UPDATE_FILES=ON \
      -DWITH_OPENGL=ON \
      -DWITH_OPENCL=ON \
      -DWITH_IPP=ON \
      -DWITH_TBB=ON \
      -DWITH_EIGEN=ON \
      -DBUILD_PROTOBUF=OFF \
      -DWITH_V4L=ON \
      -DWITH_VTK=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=RELEASE  \
      -DPYTHON3_PACKAGES_PATH=$CONDA_PREFIX/lib/python3.6/site-packages \
      -DPYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.6m \
      -DPYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=ON \
      -DINSTALL_PYTHON_EXAMPLES=ON \
      -DINSTALL_C_EXAMPLES=OFF ..
  make -j4
  make install
else
  conda install --yes -c menpo opencv3
fi
#
# INSTALL BAUTA ITSELF
#
cd $BAUTA_HOME
python setup.py install
