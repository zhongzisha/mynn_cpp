CUDA_ROOT=/usr/local/cuda-6.5
CAFFE_ROOT=/media/slave1temp/cuda/caffe-rc2

export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUDA_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CAFFE_ROOT/build/lib:/home/ubuntu/Documents/3rdparty/lib:$LD_LIBRARY_PATH

EXE=test.exe
rm -rf $EXE

g++ \
	-I${CUDA_ROOT}/include \
	-I${CAFFE_ROOT}/include -I${CAFFE_ROOT}/build/src \
	-I/usr/include/mpi \
	-I/home/ubuntu/Documents/3rdparty/include \
    test_cudnn.cpp \
    -L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas \
    -L/home/ubuntu/Documents/3rdparty/lib -lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
    -lboost_system -lboost_filesystem \
    -lmpi -lmpi++ -fopenmp \
    -L${CAFFE_ROOT}/build/lib -lcaffe \
	-lmatio \
    -o $EXE

# valgrind --tool=memcheck --leak-check=full ./test_read_imageset cifar10_train_lmdb
# valgrind --tool=memcheck --track-origins=yes ./test_read_imageset cifar10_train_lmdb


./$EXE cifar10_train_lmdb cifar10_test_lmdb mean.binaryproto 0.001 20 0.9 0.005 100 200 120 0,1










