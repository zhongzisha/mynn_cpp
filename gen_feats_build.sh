#!/bin/bash
#
#$ -V
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

# For Intel libraries
source ${PUBLICHOME}/intel/tbb/bin/tbbvars.sh intel64
source ${PUBLICHOME}/intel/ipp/bin/ippvars.sh intel64
source ${PUBLICHOME}/intel/mkl/bin/mklvars.sh intel64
source ${PUBLICHOME}/intel/bin/compilervars.sh intel64
# For CUDA libraries
export CUDA_INSTALL_PATH=/usr/local/cuda-6.5
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$CUDA_INSTALL_PATH/lib:$LD_LIBRARY_PATH
# For OpenCV libraries
export OPENCV_ROOT=${MYHOME}/softwares/
# Python include directory. This should contain the file Python.h, among others.
export PYTHON_INCLUDE_PATH=${MYHOME}/softwares/include/python2.7
# Numpy include directory. This should contain the file arrayobject.h, among others.
export NUMPY_INCLUDE_PATH=${MYHOME}/softwares/lib/python2.7/site-packages/numpy/core/include/numpy/
# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=${MYHOME}/softwares/lib
# You don't have to change these:
export CUDA_SDK_PATH=$CUDA_INSTALL_PATH/samples
export CUDA_SAMPLES_PATH=${MYHOME}/NVIDIA_CUDA-6.5_Samples
export CPLUS_INCLUDE_PATH=
# Common libraries
export COMMON_LIBRARY_PATH=${MYHOME}/softwares/lib
export PKG_CONFIG_PATH=${MYHOME}/softwares/lib/pkgconfig:${OPENCV_ROOT}/lib/pkgconfig:${PKG_CONFIG_PATH}

# Output GPU status
OutputGPUStatus() {
	# nvidia-smi -a | grep -A 1 Temperature
	nvidia-smi -a
	if [ ! -e "$CUDA_SAMPLES_PATH/1_Utilities/deviceQuery/deviceQuery" ];then
		current_dir=`pwd`
		cd $CUDA_SAMPLES_PATH/1_Utilities/deviceQuery
		make clean
		make
		cd $current_dir
	fi
	$CUDA_SAMPLES_PATH/1_Utilities/deviceQuery/deviceQuery
}

# Compile all GPU Samples
CompileAllGPUSamples() {
	current_dir=`pwd`
	cd "$CUDA_SAMPLES_PATH"
	make
	cd $current_dir
}

# Check ATLAS
CheckATLAS() {
	find /usr/include -name "cblas*"
	find /usr/lib -name "*atlas*"
	find /usr/lib64 -name "*cblas*"
	find /usr/local/lib -name "*atlas*"
	find /usr/local/lib -name "*cblas*"
	find /usr/local/lib64 -name "*atlas*"
	find /usr/local/lib64 -name "*cblas*"
	ls -alt /usr
	ls -alt /usr/local
	ls -alt /opt
}

# me="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
# me=${0##*/}
# extension="${me##*.}"
# filename="${me%.*}"

# Check valid nodes
CheckNodeIsValid() {
	nodename="$(hostname)"
	declare -a invalidnodename=("g0102.local")
	for name in "${invalidnodename[@]}"
	do
		if [ $nodename == $name ];then
			return 0;
		fi
	done
	return 1;
}

# 
CheckNodeIsValid;
is_valid_node=$? #get the result of last command
echo ${is_valid_node}
if [ ${is_valid_node} -eq 1 ];then
	echo "$(hostname) is a valid node"
else
	echo "$(hostname) is not a invalid node"
	exit
fi

gpuCount=`GPUCount`
if [ "${gpuCount}" = "1" ] || [ "${gpuCount}" = "2" ];then
	echo "Total ${gpuCount} GPUs are availabe!"
	if [ -e "${MYFLAG}_job_is_running.txt" ];then
		echo "The job is running! Do not submit it again!"		
	else
		echo "Job ${MYFLAG} is started at `hostname`!" > ${MYFLAG}_job_is_running.txt
                OutputGPUStatus
		# CompileAllGPUSamples
# ============= Add the main programs from here ============
free -g
top -b -n 1

pkg-config --libs python
pkg-config --libs opencv


echo "start the job"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH

export CAFFE_ROOT=/NLPRMNT_old/zhongzisha/deep/code/caffe

export LD_LIBRARY_PATH=$CAFFE_ROOT/build/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=${GCC463_ROOT}/lib64:${GCC463_ROOT}/lib:$LD_LIBRARY_PATH
export PATH=${GCC463_ROOT}/bin:$PATH

echo "compile ..."
rm -rf ptrn
protoc -I=./ --cpp_out=./ ./myproto.proto
nvcc -ccbin=/opt/mpich2/gnu/bin/mpicxx gen_feats.cu myproto.pb.cc \
  -I/opt/mpich2/gnu/include \
  -I${MYHOME}/softwares2/include \
  -I/usr/local/cuda-6.5/include \
  -I$CAFFE_ROOT/include -I$CAFFE_ROOT/src -I$CAFFE_ROOT/build/src \
  -L/opt/mpich2/gnu/lib -lmpich -lmpichcxx -lmpl -lopa -lfmpich -lmpichf90 \
  -L${MYHOME}/softwares2/lib -lboost_system -lboost_filesystem \
  -L/usr/local/cuda-6.5/lib64 -lcudart -lcurand -lcublas \
  -lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
  -L$CAFFE_ROOT/build/lib -lcaffe \
  -gencode arch=compute_35,code=sm_35 \
  -o ptrn

# Load the libraries
export LD_LIBRARY_PATH=${MYHOME}/softwares2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/mpich2/gnu/lib:$LD_LIBRARY_PATH

# mpirun
if [ -f $(pwd)/ptrn ]; then
echo "mpirun ..."
/opt/mpich2/gnu/bin/mpirun -np 4 -hostfile $(pwd)/nodes_gpu $(pwd)/ptrn \
	$(pwd)/p/trn/images_list.txt \
	$(pwd)/caffenet_train_iter_40000.caffemodel \
	$(pwd)/deploy.prototxt \
	output 64 64 32
fi

echo "end the job"





rm -rf *_job_is_running*
# ==========================================================
	fi
else
	echo "GPUs are not available!"
fi
