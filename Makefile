CUDA_ROOT=/usr/local/cuda-6.5
HOSTNAME=$(shell hostname)
EXE=test.exe

all:
	@echo ${HOSTNAME}
	rm -rf ${EXE}
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ -gencode arch=compute_35,code=sm_35 \
	-Xcompiler -fopenmp \
	-I${CUDNN_ROOT}/ \
	-I${CUDA_ROOT}/include \
	-I/usr/include/mpi \
	-I/home/ubuntu/Documents/3rdparty/include \
	-I${GCC463_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp test_cudnn_v2.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas \
	-L${GCC463_ROOT}/lib64 -L${GCC463_ROOT}/lib \
	-L/home/ubuntu/Documents/3rdparty/lib -lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lboost_system -lboost_filesystem \
	-lboost_thread \
	-lleveldb -llmdb \
	-lcudnn \
	-lmatio \
	-lhdf5 -lhdf5_hl \
	-o ${EXE}
	
clean:
	rm -rf ${EXE}

# valgrind --tool=memcheck --leak-check=full ./test_read_imageset cifar10_train_lmdb
# valgrind --tool=memcheck --track-origins=yes ./test_read_imageset cifar10_train_lmdb
# ./$EXE cifar10_train_lmdb cifar10_test_lmdb mean.binaryproto 0.001 20 0.9 0.005 100 200 120 0,1,2,3









