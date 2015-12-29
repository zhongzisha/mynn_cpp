CUDA_ROOT=/usr/local/cuda-6.5
HOSTNAME=$(shell hostname)
OUTDIR=./tools

all: caffe

convert_imageset:
	g++ -std=c++11 \
		-I${CUDA_ROOT}/include \
		-I${GCC484_ROOT}/include \
		myproto.pb.cc io.cpp db.cpp convert_imageset.cpp \
		-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
		-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
		-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
		-lleveldb -llmdb -lrocksdb \
		-lmatio -lhdf5 -lhdf5_hl \
		-lboost_thread -lboost_filesystem -lboost_system \
		-o ${OUTDIR}/convert_imageset
		
compute_image_mean:
	g++ -std=c++11 \
		-I${CUDA_ROOT}/include \
		-I${GCC484_ROOT}/include \
		myproto.pb.cc io.cpp db.cpp compute_image_mean.cpp \
		-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
		-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
		-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
		-lleveldb -llmdb -lrocksdb \
		-lmatio -lhdf5 -lhdf5_hl -lpthread \
		-lboost_thread -lboost_filesystem -lboost_system \
		-o ${OUTDIR}/compute_image_mean

test:
	protoc -I=./ --cpp_out=./ ./myproto.proto
	nvcc -m64 -ccbin=g++ \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-Xcompiler -fopenmp -std=c++11 \
	-I${CUDA_ROOT}/include \
	-I${GCC484_ROOT}/include \
	myproto.pb.cc io.cpp db.cpp internal_thread.cpp common.cu blob.cu data_layer.cu common_layer.cu conv_layer.cu loss_layer.cu \
	network_cifar10.cu network_alex.cu \
	test_cudnn_v2.cu \
	-L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib -lcudart -lcurand -lcublas -lcudnn \
	-L${GCC484_ROOT}/lib64 -L${GCC484_ROOT}/lib \
	-lprotobuf -lglog -lgflags -lopencv_core -lopencv_imgproc -lopencv_highgui \
	-lleveldb -llmdb -lrocksdb \
	-lmatio -lhdf5 -lhdf5_hl \
	-lboost_thread -lboost_filesystem -lboost_system \
	-o ${OUTDIR}/test
	
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
	-o ${OUTDIR}/main_cifar10net_1gpu
	
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
	-o ${OUTDIR}/main_cifar10net_1gpu_notstnet
	
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
	-o ${OUTDIR}/main_cifar10net_1gpu_convg
	
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
	-o ${OUTDIR}/main_cifar10net_1gpu_convg_notstnet
	
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
	-o ${OUTDIR}/main_cifar10net_mgpu
	
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
	-o ${OUTDIR}/main_cifar10net_mgpu_notstnet
	
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
	-o ${OUTDIR}/main_alexnet_1gpu
	
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
	-o ${OUTDIR}/main_alexnet_1gpu_notstnet
	
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
	-o ${OUTDIR}/main_alexnet_mgpu
	
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
	-o ${OUTDIR}/main_alexnet_mgpu_notstnet








