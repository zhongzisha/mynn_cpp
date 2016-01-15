CUDA_ROOT=/usr/local/cuda-6.5
HOSTNAME=$(shell hostname)
BIN_DIR = bin/
OBJ_DIR = obj/

NVCC          := $(CUDA_ROOT)/bin/nvcc -ccbin=g++
NVCC_MPI	  := $(CUDA_ROOT)/bin/nvcc -ccbin=$(MPIHOME)/bin/mpicxx

# internal flags
NVCC_FLAGS += -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 \
			  -Xcompiler -fopenmp -std=c++11
			  		
# Common includes and paths for CUDA
INCLUDES  := -I${GCC484_ROOT}/include -I${CUDA_ROOT}/include -I${MPIHOME}/include
LIBRARIES := -L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
		-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
		-lleveldb -llmdb -lrocksdb \
		-lmatio -lhdf5 -lhdf5_hl \
		-lboost_thread -lboost_filesystem -lboost_system -lgomp \
		-L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs \
		-L$(CUDA_ROOT)/lib -L$(CUDA_ROOT)/lib/stubs \
		-lcudart -lcublas -lcurand -lcudnn \
		-L${MPIHOME}/lib -lmpich -lmpichcxx -lmpl -lopa -lfmpich -lmpichf90

myproto.pb.o:myproto.pb.cc
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
io.o:io.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<

db.o:db.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<

internal_thread.o:internal_thread.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
common.o:common.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<

blob.o:blob.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
data_layer.o:data_layer.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
common_layer.o:common_layer.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
conv_layer.o:conv_layer.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
loss_layer.o:loss_layer.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
network_cifar10.o:network_cifar10.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
network_alex.o:network_alex.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<

convert_cifar_data.o:convert_cifar_data.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<

convert_cifar_data.exe: myproto.pb.o io.o db.o convert_cifar_data.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

convert_imageset.o:convert_imageset.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
convert_imageset.exe: myproto.pb.o io.o db.o convert_imageset.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	
compute_image_mean.o:compute_image_mean.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
compute_image_mean.exe: myproto.pb.o io.o db.o compute_image_mean.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

test.o:test_cudnn_v2.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
test.exe:myproto.pb.o io.o db.o internal_thread.o common.o blob.o data_layer.o common_layer.o conv_layer.o loss_layer.o network_cifar10.o network_alex.o test.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

test_mpi.o:test_mpi.cu
	$(NVCC_MPI) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
test_mpi.exe: myproto.pb.o io.o db.o internal_thread.o common.o blob.o data_layer.o common_layer.o conv_layer.o loss_layer.o network_cifar10.o network_alex.o test_mpi.o
	$(NVCC_MPI) $(NVCC_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

main_cifar10net_1gpu_notstnet.o:main_cifar10net_1gpu_notstnet.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
main_cifar10net_1gpu_notstnet.exe: myproto.pb.o io.o db.o internal_thread.o common.o blob.o data_layer.o common_layer.o conv_layer.o loss_layer.o network_cifar10.o network_alex.o main_cifar10net_1gpu_notstnet.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

main_cifar10net_1gpu_mpi.o:main_cifar10net_1gpu_mpi.cu
	$(NVCC_MPI) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
main_cifar10net_1gpu_mpi.exe:myproto.pb.o io.o db.o internal_thread.o common.o blob.o data_layer.o common_layer.o conv_layer.o loss_layer.o network_cifar10.o network_alex.o main_cifar10net_1gpu_mpi.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

main_cifar10net_mgpu_mpi.o:main_cifar10net_mgpu_mpi.cu
	$(NVCC_MPI) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
main_cifar10net_mgpu_mpi.exe:myproto.pb.o io.o db.o internal_thread.o common.o blob.o data_layer.o common_layer.o conv_layer.o loss_layer.o network_cifar10.o network_alex.o main_cifar10net_mgpu_mpi.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

main_alexnet_mgpu_mpi.o:main_alexnet_mgpu_mpi.cu
	$(NVCC_MPI) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
main_alexnet_mgpu_mpi.exe:myproto.pb.o io.o db.o internal_thread.o common.o blob.o data_layer.o common_layer.o conv_layer.o loss_layer.o network_cifar10.o network_alex.o main_alexnet_mgpu_mpi.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)
	
main_alexnet_mgpu_notstnet.o:main_alexnet_mgpu_notstnet.cu
	$(NVCC_MPI) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<
	
main_alexnet_mgpu_notstnet.exe:myproto.pb.o io.o db.o internal_thread.o common.o blob.o data_layer.o common_layer.o conv_layer.o loss_layer.o network_cifar10.o network_alex.o main_alexnet_mgpu_notstnet.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

tools: convert_cifar_data.exe convert_imageset.exe compute_image_mean.exe
tests: test.exe test_mpi.exe

all: tools tests

clean:
	rm -rf *.o *.exe
	
main_cifar10net_1gpu:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_cifar10.cu \
	main_cifar10net_1gpu.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_cifar10net_1gpu
	
main_cifar10net_1gpu_notstnet:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_cifar10.cu \
	main_cifar10net_1gpu_notstnet.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_cifar10net_1gpu_notstnet
	
main_cifar10net_1gpu_convg:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_cifar10_convg.cu \
	main_cifar10net_1gpu_convg.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_cifar10net_1gpu_convg
	
main_cifar10net_1gpu_convg_notstnet:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_cifar10_convg.cu \
	main_cifar10net_1gpu_convg_notstnet.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_cifar10net_1gpu_convg_notstnet
	
main_cifar10net_mgpu:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_cifar10.cu \
	main_cifar10net_mgpu.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_cifar10net_mgpu
	
main_cifar10net_mgpu_notstnet:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_cifar10.cu \
	main_cifar10net_mgpu_notstnet.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_cifar10net_mgpu_notstnet
	
main_alexnet_1gpu:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_alex.cu \
	main_alexnet_1gpu.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_alexnet_1gpu
	
main_alexnet_1gpu_notstnet:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_alex.cu \
	main_alexnet_1gpu_notstnet.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_alexnet_1gpu_notstnet
	
main_alexnet_mgpu:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_alex.cu \
	main_alexnet_mgpu.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_alexnet_mgpu
	
main_alexnet_mgpu_notstnet:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_alex.cu \
	main_alexnet_mgpu_notstnet.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${BIN_DIR}/main_alexnet_mgpu_notstnet








