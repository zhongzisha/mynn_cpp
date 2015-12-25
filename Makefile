CUDA_ROOT=/usr/local/cuda-6.5
HOSTNAME=$(shell hostname)
OUTDIR=./tools
EXE=caffe

all:
	mkdir -p ${OUTDIR}
	@echo ${HOSTNAME}
	rm -rf ${OUTDIR}/${EXE}
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp \
	-I${CUDA_ROOT}/include \
	-I/usr/include/mpi \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp test_cudnn_v2.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${OUTDIR}/${EXE}
	
clean:
	rm -rf ${EXE}










