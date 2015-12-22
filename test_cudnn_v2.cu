
#include <glog/logging.h>
#include <pthread.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
using namespace std;

#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <boost/filesystem.hpp>
using namespace boost;

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

extern "C" {
#include <cblas.h>
}


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include "matio.h"

#include "myproto.pb.h"
#include "io.hpp"
#include "db.hpp"
#include "internal_thread.hpp"


template <typename Dtype> enum matio_types matio_type_map();
template <> enum matio_types matio_type_map<float>() { return MAT_T_SINGLE; }
template <> enum matio_types matio_type_map<double>() { return MAT_T_DOUBLE; }
template <> enum matio_types matio_type_map<int>() { return MAT_T_INT32; }
template <> enum matio_types matio_type_map<unsigned int>() { return MAT_T_UINT32; }

template <typename Dtype> enum matio_classes matio_class_map();
template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

const char* curandGetErrorString(curandStatus_t error) {
	switch (error) {
	case CURAND_STATUS_SUCCESS:
		return "CURAND_STATUS_SUCCESS";
	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";
	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	}
	return "Unknown curand status";
}

const char* cublasGetErrorString(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
	}
	return "Unknown cublas status";
}

/********************************************************
 * Prints the error message, and exits
 * ******************************************************/


// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
		/* Code block avoids redefinition of cudaError_t error */ \
		do { \
			cudaError_t error = condition; \
			if (error != cudaSuccess) \
			printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
		} while (0)

#define CUBLAS_CHECK(condition) \
		do { \
			cublasStatus_t status = condition; \
			if (status != CUBLAS_STATUS_SUCCESS) \
			printf("CUBLAS Error: %s\n", cublasGetErrorString(status)); \
		} while (0)

#define CURAND_CHECK(condition) \
		do { \
			curandStatus_t status = condition; \
			if (status != CURAND_STATUS_SUCCESS) \
			printf("CURAND Error: %s\n", curandGetErrorString(status)); \
		} while (0)

#define CUDNN_CHECK(status) \
		do { \
			if (status != CUDNN_STATUS_SUCCESS) \
			printf("CUDNN Error: %s\n", cudnnGetErrorString(status)); \
		} while (0)

#define EXIT_WAIVED 0
#define FatalError(s) {                                                \
		std::stringstream _where, _message;                                \
		_where << __FILE__ << ':' << __LINE__;                             \
		_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
		std::cerr << _message.str() << "\nAborting...\n";                  \
		cudaDeviceReset();                                                 \
		exit(EXIT_FAILURE);                                                \
		}


void EnableP2P(vector<int> gpus)
{
	// check p2p access
	cudaDeviceProp prop[gpus.size()];
	for(int i = 0; i < gpus.size(); i++) {
		cudaGetDeviceProperties(&prop[i], gpus[i]);
	}

	for(int i = 0; i < gpus.size(); i++) {
		for(int j = 0; j < gpus.size(); j++) {
			if(i==j)
				continue;
			int can_access_peer;
			cudaDeviceCanAccessPeer(&can_access_peer, gpus[i], gpus[j]);
			if(can_access_peer) {
				cudaSetDevice(gpus[i]);
				cudaDeviceEnablePeerAccess(gpus[j], 0);
				cudaSetDevice(gpus[j]);
				cudaDeviceEnablePeerAccess(gpus[i], 0);
				const bool has_uva = (prop[gpus[i]].unifiedAddressing && prop[gpus[j]].unifiedAddressing);
				if(has_uva) {
					printf("(%d <--> %d): YES!\n", gpus[i], gpus[j]);
				}
			} else {
				printf("(%d <--> %d): NO!\n", gpus[i], gpus[j]);
			}
		}
	}
}

void DisableP2P(vector<int> gpus)
{
	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		cudaDeviceDisablePeerAccess(gpus[i]);
	}
}

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int GPU_CUDA_NUM_THREADS = 1024;
#else
    const int GPU_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int GPU_GET_BLOCKS(const int N) {
  return (N + GPU_CUDA_NUM_THREADS - 1) / GPU_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void add_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

void cpu_add(const int N, const float *a, const float *b, float *y) {
#pragma omp parallel
	for(int i = 0; i < N; i++) {
		y[i] = a[i] + b[i];
	}
}

void gpu_add(const int N, const float* a, const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<<<GPU_GET_BLOCKS(N), GPU_CUDA_NUM_THREADS>>>(N, a, b, y);
}

void gpu_copy(const int N, const float *X, float *Y) {
	CUDA_CHECK( cudaMemcpy(Y, X, sizeof(float) * N, cudaMemcpyDefault) );
}

void gpu_scal(cublasHandle_t cublashandle, const int N, const float alpha, float *X) {
	CUBLAS_CHECK( cublasSscal(cublashandle, N, &alpha, X, 1) );
}

void gpu_axpy(cublasHandle_t cublashandle, const int N, const float alpha,
		const float* X, float* Y) {
	CUBLAS_CHECK( cublasSaxpy(cublashandle, N, &alpha, X, 1, Y, 1) );
}

void gpu_axpby(cublasHandle_t cublashandle, const int N, const float alpha,
		const float* X, const float beta, float* Y) {
	gpu_scal(cublashandle, N, beta, Y);
	gpu_axpy(cublashandle, N, alpha, X, Y);
}

void gpu_gemv(cublasHandle_t cublashandle,
		const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const float alpha, const float* A, const float* x,
		const float beta, float* y) {
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	CUBLAS_CHECK( cublasSgemv(cublashandle, cuTransA, N, M, &alpha, A, N, x, 1, &beta, y, 1) );
}

void gpu_gemm(cublasHandle_t cublashandle,
		const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta,
		float* C) {
	// Note that cublas follows fortran order.
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	CUBLAS_CHECK( cublasSgemm(cublashandle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N) );
}

class Blob_t
{
public:
	int N;
	int C;
	int H;
	int W;
	float *data_cpu, *data_gpu;
	float *diff_cpu, *diff_gpu;

	Blob_t() : data_cpu(NULL), data_gpu(NULL), diff_cpu(NULL), diff_gpu(NULL), N(0), C(0), H(0), W(0) {};

	Blob_t(int N_, int C_, int H_, int W_) : data_cpu(NULL), data_gpu(NULL), diff_cpu(NULL), diff_gpu(NULL)
	{
		N = N_;
		C = C_;
		H = H_;
		W = W_;
	}

	~Blob_t()
	{
		if(data_cpu != NULL)
		{
			CUDA_CHECK( cudaFreeHost(data_cpu) );
			data_cpu = NULL;
		}
		if(data_gpu != NULL)
		{
			CUDA_CHECK( cudaFree(data_gpu) );
			data_gpu = NULL;
		}
		if(diff_cpu != NULL)
		{
			CUDA_CHECK( cudaFreeHost(diff_cpu) );
			diff_cpu = NULL;
		}
		if(diff_gpu != NULL)
		{
			CUDA_CHECK( cudaFree(diff_gpu) );
			diff_gpu = NULL;
		}
	}

	inline int count() const {
		return N * C * H * W;
	}
	inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const {
		return ((n * C + c) * H + h) * W + w;
	}

	void print_gpu_data() {
		if(data_gpu == NULL)
			printf("gpu data is NULL.\n");
		data_to_cpu();
		for(int n = 0; n < N; n++) {
			for(int c = 0; c < C; c++) {
				for(int h = 0; h < H; h++) {
					for(int w = 0; w < W; w++) {
						int index = (((n)*C + c)*H + h)*W+w;
						printf("(%d, %d, %d, %d) : %.6f\n", n, c, h, w, data_cpu[index]);
					}
				}
			}
		}

	}

	void print_gpu_data(int howmany) {
		if(data_gpu == NULL)
			printf("gpu data is NULL.\n");
		data_to_cpu();
		for(int n = 0; n < 1; n++) {
			for(int c = 0; c < 1; c++) {
				for(int h = 0; h < 1; h++) {
					for(int w = 0; w < W; w++) {
						int index = (((n)*C + c)*H + h)*W+w;
						printf("(%d, %d, %d, %d) : %.6f\n", n, c, h, w, data_cpu[index]);
					}
				}
			}
		}
	}

	void print_cpu_data(int howmany) {
		if(data_cpu == NULL)
			printf("data_cpu is NULL.\n");
		for(int n = 0; n < 1; n++) {
			for(int c = 0; c < 1; c++) {
				for(int h = 0; h < 1; h++) {
					for(int w = 0; w < W; w++) {
						int index = (((n)*C + c)*H + h)*W+w;
						printf("(%d, %d, %d, %d) : %.6f\n", n, c, h, w, data_cpu[index]);
					}
				}
			}
		}
	}

	void save_cpu_data_and_diff_to_mat(const char *fname)
	{
		// save results into matlab format
		mat_t *matfp = Mat_Create(fname, 0);
		//matfp = Mat_CreateVer(fname, 0, MAT_FT_MAT73);
		size_t dims[4];
		dims[0] = W;
		dims[1] = H;
		dims[2] = C;
		dims[3] = N;
		matvar_t *matvar, *matvar2;
		// save data
		{
			matvar = Mat_VarCreate("data", matio_class_map<float>(), matio_type_map<float>(), 4, dims, data_cpu, 0);
			if(matvar == NULL)
				cout << "Error creating 'data' variable";
			matvar2 = Mat_VarCreate("diff", matio_class_map<float>(), matio_type_map<float>(), 4, dims, diff_cpu, 0);
			if(matvar2 == NULL)
				cout << "Error creating 'diff' variable";
			if(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0)
				cout << "Error saving array 'data' into MAT file " << fname;

			if(Mat_VarWrite(matfp, matvar2, MAT_COMPRESSION_NONE) != 0)
				cout << "Error saving array 'diff' into MAT file " << fname;

			Mat_VarFree(matvar);
			Mat_VarFree(matvar2);
		}
		Mat_Close(matfp);
	}

	/*
	 *  data allocate
	 */
	void allocate_gpu_data()
	{
		int count = N * C * H * W;
		if(data_gpu != NULL)
			CUDA_CHECK( cudaFree(data_gpu) );
		CUDA_CHECK( cudaMalloc((void**)&data_gpu, count * sizeof(float)) );
		CUDA_CHECK( cudaMemset(data_gpu, 0, count * sizeof(float)) );
	}

	void allocate_gpu_diff()
	{
		int count = N * C * H * W;
		if(diff_gpu != NULL)
			CUDA_CHECK( cudaFree(diff_gpu) );
		CUDA_CHECK( cudaMalloc((void**)&diff_gpu, count * sizeof(float)) );
		CUDA_CHECK( cudaMemset(diff_gpu, 0, count * sizeof(float)) );
	}

	void allocate_cpu_data()
	{
		int count = N * C * H * W;
		if(data_cpu != NULL)
			CUDA_CHECK( cudaFreeHost(data_cpu) );
		CUDA_CHECK( cudaMallocHost((void**)&data_cpu, count * sizeof(float)) );
		CUDA_CHECK( cudaMemset(data_cpu, 0, count * sizeof(float)) );
	}

	void allocate_cpu_diff()
	{
		int count = N * C * H * W;
		if(diff_cpu != NULL)
			CUDA_CHECK( cudaFreeHost(diff_cpu) );
		CUDA_CHECK( cudaMallocHost((void**)&diff_cpu, count * sizeof(float)) );
		CUDA_CHECK( cudaMemset(diff_cpu, 0, count * sizeof(float)) );
	}

	/*
	 * data copy
	 */
	void data_to_gpu()
	{
		int count = N * C * H * W;
		if(data_gpu == NULL)
			CUDA_CHECK( cudaMalloc((void**)&data_gpu, count * sizeof(float)) );
		if(data_cpu != NULL)
			CUDA_CHECK( cudaMemcpy(data_gpu, data_cpu, count * sizeof(float), cudaMemcpyHostToDevice) );
	}

	void diff_to_gpu()
	{
		int count = N * C * H * W;
		if(diff_gpu == NULL)
			CUDA_CHECK( cudaMalloc((void**)&diff_gpu, count * sizeof(float)) );
		if(diff_cpu != NULL)
			CUDA_CHECK( cudaMemcpy(diff_gpu, diff_cpu, count * sizeof(float), cudaMemcpyHostToDevice) );
	}

	void data_to_cpu()
	{
		int count = N * C * H * W;
		if(data_cpu == NULL)
			CUDA_CHECK( cudaMallocHost((void**)&data_cpu, count * sizeof(float)) );
		if(data_gpu != NULL)
			CUDA_CHECK( cudaMemcpy(data_cpu, data_gpu, count * sizeof(float), cudaMemcpyDeviceToHost) );
	}

	void diff_to_cpu()
	{
		int count = N * C * H * W;
		if(diff_cpu == NULL)
			CUDA_CHECK( cudaMallocHost((void**)&diff_cpu, count * sizeof(float)) );
		if(diff_gpu != NULL)
			CUDA_CHECK( cudaMemcpy(diff_cpu, diff_gpu, count * sizeof(float), cudaMemcpyDeviceToHost) );
	}
};

void cpu_fill(float *ptr, int count, float value) {
	memset(ptr, value, count * sizeof(float));
}

void gpu_fill(curandGenerator_t curand_generator, float *ptr, int count, float mu, float std = 0.0f) {
	if(std == 0.0f) {
		CUDA_CHECK( cudaMemset(ptr, mu, count * sizeof(float)) );
	} else {
		CURAND_CHECK( curandGenerateNormal(curand_generator, ptr, count, mu, std) );
	}
}

void CopyBlobData_gpu(const Blob_t *src, int src_gpu_id, Blob_t *dst, int dst_gpu_id)
{
	int count = src->count();
	if(src_gpu_id == dst_gpu_id) {
		cudaSetDevice(src_gpu_id);
		cudaMemcpy(dst->data_gpu, src->data_gpu, count * sizeof(float), cudaMemcpyDefault);
	} else {
		cudaDeviceProp prop[2];
		cudaGetDeviceProperties(&prop[0], src_gpu_id);
		cudaGetDeviceProperties(&prop[1], dst_gpu_id);
		int can_access_peer;
		cudaDeviceCanAccessPeer(&can_access_peer, src_gpu_id, dst_gpu_id);
		if(can_access_peer) {
			const bool has_uva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
			if(has_uva) {
				cudaMemcpy(dst->data_gpu, src->data_gpu, count * sizeof(float), cudaMemcpyDefault);
				return;
			}
		}

		float *temp_data = NULL;
		for(int i = 0; i <= 1; i++) {
			cudaSetDevice(src_gpu_id);
			cudaMallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->data_gpu, count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMemcpy(dst->data_gpu, temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			cudaFreeHost(temp_data);
		}
	}
}

void AddBlobDiff_gpu(const Blob_t *src, int src_gpu_id, Blob_t *dst, int dst_gpu_id)
{
	int count = src->count();
	if(src_gpu_id == dst_gpu_id) {
		cudaSetDevice(src_gpu_id);
		gpu_add(count, src->diff_gpu, dst->diff_gpu, dst->diff_gpu);
	} else {
		cudaDeviceProp prop[2];
		cudaGetDeviceProperties(&prop[0], src_gpu_id);
		cudaGetDeviceProperties(&prop[1], dst_gpu_id);
		int can_access_peer;
		cudaDeviceCanAccessPeer(&can_access_peer, src_gpu_id, dst_gpu_id);
		if(can_access_peer) {
			const bool has_uva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
			if(has_uva) {
				gpu_add(count, src->diff_gpu, dst->diff_gpu, dst->diff_gpu);
				return;
			}
		}

		float *temp_data = NULL;
		float *dst_temp_data = NULL;
		for(int i = 0; i <= 1; i++) {
			cudaSetDevice(src_gpu_id);
			cudaMallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->diff_gpu, count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMalloc((void **)&dst_temp_data, count * sizeof(float));
			cudaMemcpy(dst_temp_data, temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			gpu_add(count, dst_temp_data, dst->diff_gpu, dst->diff_gpu);
			cudaFreeHost(temp_data);
			cudaFree(dst_temp_data);
		}
	}
}

class DataLayerParameter_t
{
public:
	string backend;
	string source;
	string mean_file;
	int batch_size;
};

class DataLayer_t : public InternalThread
{
public:
	DataLayerParameter_t *data_params;
	Blob_t *prefetch_data_;
	Blob_t *prefetch_label_;
	Blob_t *mean_;
	int datum_size_;

	shared_ptr<db::DB> db_;
	shared_ptr<db::Cursor> cursor_;

	DataLayer_t(const DataLayerParameter_t *data_params_) {
		data_params = const_cast<DataLayerParameter_t *>(data_params_);

		prefetch_data_ = NULL;
		prefetch_label_ = NULL;
		mean_ = NULL;
		datum_size_ = 0;
	}

	~DataLayer_t() {
		JoinPrefetchThread(); // here, we should stop the final thread, when we delete the class instance
		delete prefetch_data_;
		delete prefetch_label_;
		delete mean_;
	}

	void Setup() {
		// Initialize DB
		db_.reset(db::GetDB(data_params->backend));
		db_->Open(data_params->source, db::READ);
		cursor_.reset(db_->NewCursor());

		// Read a data point, and use it to initialize the top blob.
		Datum datum;
		datum.ParseFromString(cursor_->value());
		datum_size_ = datum.channels() * datum.height() * datum.width();

		// printf("datum: (%d, %d, %d, %d)\n", data_params->batch_size, datum.channels(), datum.height(), datum.width());

		prefetch_data_ = new Blob_t(data_params->batch_size, datum.channels(), datum.height(), datum.width());
		prefetch_label_ = new Blob_t(data_params->batch_size, 1, 1, 1);
		prefetch_data_->allocate_cpu_data();
		prefetch_label_->allocate_cpu_data();

		mean_ = new Blob_t(1, datum.channels(), datum.height(), datum.width());
		mean_->allocate_cpu_data();
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(data_params->mean_file.c_str(), &blob_proto);
		for (int i = 0; i < mean_->count(); ++i) {
			mean_->data_cpu[i] = (float)blob_proto.data(i);
		}
		// printf("mean_: (%d, %d, %d, %d)\n", mean_->N, mean_->C, mean_->H, mean_->W);

		CreatePrefetchThread();
	}

	void Forward_cpu(Blob_t *top_data, Blob_t *top_label) {
		// printf("First, join the thread.\n");
		JoinPrefetchThread();

		// printf("copy data to top_data.\n");
		top_data->N = prefetch_data_->N;
		top_data->C = prefetch_data_->C;
		top_data->H = prefetch_data_->H;
		top_data->W = prefetch_data_->W;
		top_data->allocate_cpu_data();
		memcpy(top_data->data_cpu, prefetch_data_->data_cpu, prefetch_data_->count() * sizeof(float));

		// printf("copy label to top_label.\n");
		top_label->N = prefetch_label_->N;
		top_label->C = prefetch_label_->C;
		top_label->H = prefetch_label_->H;
		top_label->W = prefetch_label_->W;
		top_label->allocate_cpu_data();
		memcpy(top_label->data_cpu, prefetch_label_->data_cpu, prefetch_label_->count() * sizeof(float));

		// printf("Start a new prefetch thread.\n");
		CreatePrefetchThread();
	}

	void Forward_cpu_multi(vector<Blob_t *> &top_data, vector<Blob_t *> &top_label, vector<int> &batch_sizes) {
		// printf("First, join the thread.\n");
		JoinPrefetchThread();

		for(int i = 0; i < batch_sizes.size(); i++) {
			int start_index = 0;
			for(int j = 0; j < i; j++) {
				start_index += batch_sizes[j];
			}
			// printf("copy data to top_data.\n");
			top_data[i]->N = batch_sizes[i];
			top_data[i]->C = prefetch_data_->C;
			top_data[i]->H = prefetch_data_->H;
			top_data[i]->W = prefetch_data_->W;
			top_data[i]->allocate_cpu_data();
			memcpy(top_data[i]->data_cpu,
					prefetch_data_->data_cpu + start_index * top_data[i]->C * top_data[i]->H * top_data[i]->W,
					top_data[i]->count() * sizeof(float));

			// printf("copy label to top_label.\n");
			top_label[i]->N = batch_sizes[i];
			top_label[i]->C = prefetch_label_->C;
			top_label[i]->H = prefetch_label_->H;
			top_label[i]->W = prefetch_label_->W;
			top_label[i]->allocate_cpu_data();
			memcpy(top_label[i]->data_cpu,
					prefetch_label_->data_cpu + start_index * top_data[i]->C * top_data[i]->H * top_data[i]->W,
					top_label[i]->count() * sizeof(float));
		}
		// printf("Start a new prefetch thread.\n");
		CreatePrefetchThread();
	}

protected:
	void CreatePrefetchThread() {
		CHECK(StartInternalThread()) << "Thread execution failed";
	}
	void JoinPrefetchThread() {
		CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
	}
	void InternalThreadEntry(){

		float *top_data = prefetch_data_->data_cpu;
		float *top_label = prefetch_label_->data_cpu;
		float *mean_data = mean_->data_cpu;
		for (int item_id = 0; item_id < data_params->batch_size; ++item_id) {

			// get a blob
			Datum datum;
			datum.ParseFromString(cursor_->value());

			// read one data
			const string& data = datum.data();
			if (data.size()) {
				for (int j = 0; j < datum_size_; ++j) {
					top_data[item_id * datum_size_ + j] = (static_cast<float>((uint8_t)data[j])) - mean_data[j];
				}
			} else {
				for (int j = 0; j < datum_size_; ++j) {
					top_data[item_id * datum_size_ + j] = (datum.float_data(j)) - mean_data[j];
				}
			}

			// read the label
			top_label[item_id] = datum.label();

			// go to the next iter
			cursor_->Next();
			if (!cursor_->valid()) {
				cursor_->SeekToFirst();
			}
		}
	}
};

class ConvolutionParameter_t
{
public:
	int filter_N;
	int filter_C;
	int filter_H;
	int filter_W;
	int pad_h, pad_w;
	int stride_h, stride_w;
	int upscale_h, upscale_w;
	cudnnConvolutionMode_t cudnn_conv_mode;
};

class Layer_t
{
public:
	cudnnDataType_t dataType;
	cudnnTensorFormat_t tensorFormat;

	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t bottomTensorDesc;
	cudnnTensorDescriptor_t topTensorDesc;
	Layer_t()
	{
		dataType = CUDNN_DATA_FLOAT;
		tensorFormat = CUDNN_TENSOR_NCHW;
		cudnnHandle = NULL;

		CUDNN_CHECK( cudnnCreate(&cudnnHandle) );
		CUDNN_CHECK( cudnnCreateTensorDescriptor(&bottomTensorDesc) );
		CUDNN_CHECK( cudnnCreateTensorDescriptor(&topTensorDesc) );
	}

	~Layer_t()
	{
		CUDNN_CHECK( cudnnDestroyTensorDescriptor(bottomTensorDesc) );
		CUDNN_CHECK( cudnnDestroyTensorDescriptor(topTensorDesc) );
		CUDNN_CHECK( cudnnDestroy(cudnnHandle) );
	}

	void Setup(const Blob_t *bottom, Blob_t *top) {};
	void Forward(const Blob_t *bottom, Blob_t *top) {};
	void Backward(const Blob_t *top, Blob_t *bottom) {};
};

class ConvolutionLayer_t : public Layer_t
{
public:
	Blob_t *filtersBlob;
	Blob_t *biasBlob;

	cudnnFilterDescriptor_t filterDesc;
	cudnnTensorDescriptor_t biasTensorDesc;
	cudnnConvolutionDescriptor_t convDesc;
	ConvolutionParameter_t *conv_params;


	ConvolutionLayer_t(const ConvolutionParameter_t *conv_params_)
	{
		conv_params = const_cast<ConvolutionParameter_t*>(conv_params_);
		filtersBlob = new Blob_t(conv_params->filter_N, conv_params->filter_C, conv_params->filter_H, conv_params->filter_W);
		biasBlob = new Blob_t(1, conv_params->filter_C, 1, 1);

		filtersBlob->allocate_gpu_data();
		filtersBlob->allocate_gpu_diff();
		biasBlob->allocate_gpu_data();
		biasBlob->allocate_gpu_diff();

		CUDNN_CHECK( cudnnCreateFilterDescriptor(&filterDesc) );
		CUDNN_CHECK( cudnnCreateTensorDescriptor(&biasTensorDesc) );
		CUDNN_CHECK( cudnnCreateConvolutionDescriptor(&convDesc) );
	};

	~ConvolutionLayer_t()
	{
		delete filtersBlob; filtersBlob = NULL;
		delete biasBlob; biasBlob = NULL;

		CUDNN_CHECK( cudnnDestroyConvolutionDescriptor(convDesc) );
		CUDNN_CHECK( cudnnDestroyFilterDescriptor(filterDesc) );
		CUDNN_CHECK( cudnnDestroyTensorDescriptor(biasTensorDesc) );
	}


	void Setup(const Blob_t *bottom, Blob_t *top)
	{

		CUDNN_CHECK( cudnnSetTensor4dDescriptor(bottomTensorDesc,
				tensorFormat,
				dataType,
				bottom->N,
				bottom->C,
				bottom->H,
				bottom->W) );

		CUDNN_CHECK( cudnnSetFilter4dDescriptor(filterDesc,
				dataType,
				filtersBlob->C,
				filtersBlob->N,
				filtersBlob->H,
				filtersBlob->W) );

		CUDNN_CHECK( cudnnSetConvolution2dDescriptor(convDesc,
				conv_params->pad_h, // padding
				conv_params->pad_w,
				conv_params->stride_h, // stride
				conv_params->stride_w,
				conv_params->upscale_h, // upscale
				conv_params->upscale_w,
				conv_params->cudnn_conv_mode) );

		// find dimension of convolution output
		CUDNN_CHECK( cudnnGetConvolution2dForwardOutputDim(convDesc,
				bottomTensorDesc,
				filterDesc,
				&(top->N),
				&(top->C),
				&(top->H),
				&(top->W)) );

		CUDNN_CHECK( cudnnSetTensor4dDescriptor(topTensorDesc,
				tensorFormat,
				dataType,
				top->N,
				top->C,
				top->H,
				top->W) );

		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}

	void Forward(const Blob_t *bottom, Blob_t *top)
	{
		cudnnConvolutionFwdAlgo_t algo;
		CUDNN_CHECK( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				bottomTensorDesc,
				filterDesc,
				convDesc,
				topTensorDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				0,
				&algo ) );

		size_t sizeInBytes=0;
		void* workSpace=NULL;
		CUDNN_CHECK( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				bottomTensorDesc,
				filterDesc,
				convDesc,
				topTensorDesc,
				algo,
				&sizeInBytes) );
		if (sizeInBytes!=0)
		{
			CUDA_CHECK( cudaMalloc(&workSpace,sizeInBytes) );
		}
		float alpha = float(1);
		float beta  = float(0);
		CUDNN_CHECK( cudnnConvolutionForward(cudnnHandle,
				&alpha,
				bottomTensorDesc,
				bottom->data_gpu,
				filterDesc,
				filtersBlob->data_gpu,
				convDesc,
				algo,
				workSpace,
				sizeInBytes,
				&beta,
				topTensorDesc,
				top->data_gpu) );

		// add bias
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(biasTensorDesc,
				tensorFormat,
				dataType,
				1,
				top->C,
				1,
				1) );
		alpha = float(1);
		beta  = float(1);
		CUDNN_CHECK( cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C,
				&alpha,
				biasTensorDesc,
				biasBlob->data_gpu,
				&beta,
				topTensorDesc,
				top->data_gpu) );

		// free buffer
		if (sizeInBytes!=0)
		{
			CUDA_CHECK( cudaFree(workSpace) );
		}

	}

	void Backward(const Blob_t *top, Blob_t *bottom)
	{

		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnnHandle,
				&alpha,
				topTensorDesc,
				top->diff_gpu,
				&beta,
				biasTensorDesc,
				biasBlob->diff_gpu));

		CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnnHandle,
				&alpha,
				bottomTensorDesc,
				bottom->data_gpu,
				topTensorDesc,
				top->diff_gpu,
				convDesc,
				&beta,
				filterDesc,
				filtersBlob->diff_gpu));

		CUDNN_CHECK(cudnnConvolutionBackwardData(cudnnHandle,
				&alpha,
				filterDesc,
				filtersBlob->data_gpu,
				topTensorDesc,
				top->diff_gpu,
				convDesc,
				&beta,
				bottomTensorDesc,
				bottom->diff_gpu));

	}
};

class ActivationParameter_t
{
public:
	cudnnActivationMode_t cudnn_activation_mode;
};

class ActivationLayer_t : public Layer_t
{
public:
	ActivationParameter_t *cudnn_activation_params;

	ActivationLayer_t(const ActivationParameter_t *cudnn_activation_params_) {
		cudnn_activation_params = const_cast<ActivationParameter_t *>(cudnn_activation_params_);
	}

	~ActivationLayer_t() {

	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(bottomTensorDesc,
				tensorFormat,
				dataType,
				bottom->N,
				bottom->C,
				bottom->H,
				bottom->W) );

		top->N = bottom->N;
		top->C = bottom->C;
		top->H = bottom->H;
		top->W = bottom->W;

		CUDNN_CHECK( cudnnSetTensor4dDescriptor(topTensorDesc,
				tensorFormat,
				dataType,
				top->N,
				top->C,
				top->H,
				top->W) );

		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}

	void Forward(const Blob_t *bottom, Blob_t *top) {
		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK( cudnnActivationForward(cudnnHandle,
				cudnn_activation_params->cudnn_activation_mode,
				&alpha,
				bottomTensorDesc,
				bottom->data_gpu,
				&beta,
				topTensorDesc,
				top->data_gpu) );
	}

	void Backward(const Blob_t *top, Blob_t *bottom) {
		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK( cudnnActivationBackward( cudnnHandle,
				cudnn_activation_params->cudnn_activation_mode,
				&alpha,
				topTensorDesc,
				top->data_gpu,
				topTensorDesc,
				top->diff_gpu,
				bottomTensorDesc,
				bottom->data_gpu,
				&beta,
				bottomTensorDesc,
				bottom->diff_gpu) );
	}
};

class PoolingParameter_t
{
public:
	cudnnPoolingMode_t cudnn_pooling_mode;
	int poolsize_h;
	int poolsize_w;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
};

class PoolingLayer_t : public Layer_t
{
public:
	PoolingParameter_t *cudnn_pooling_params;
	cudnnPoolingDescriptor_t poolingDesc;

	PoolingLayer_t(const PoolingParameter_t *cudnn_pooling_params_) {
		cudnn_pooling_params = const_cast<PoolingParameter_t *>(cudnn_pooling_params_);

		CUDNN_CHECK( cudnnCreatePoolingDescriptor(&poolingDesc) );
	}

	~PoolingLayer_t() {
		CUDNN_CHECK( cudnnDestroyPoolingDescriptor(poolingDesc) );
	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		CUDNN_CHECK( cudnnSetPooling2dDescriptor(poolingDesc,
				cudnn_pooling_params->cudnn_pooling_mode,
				cudnn_pooling_params->poolsize_h, // window
				cudnn_pooling_params->poolsize_w,
				cudnn_pooling_params->pad_h,
				cudnn_pooling_params->pad_w,
				cudnn_pooling_params->stride_h,
				cudnn_pooling_params->stride_w) );
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(bottomTensorDesc,
				tensorFormat,
				dataType,
				bottom->N,
				bottom->C,
				bottom->H,
				bottom->W) );
		/*
		CUDNN_CHECK( cudnnGetPooling2dForwardOutputDim(poolingDesc,
				bottomTensorDesc,
				&(top->N),
				&(top->C),
				&(top->H),
				&(top->W)) );
		 */
		top->N = bottom->N;
		top->C = bottom->C;
		top->H = 1 + ceil((bottom->H + 2 * cudnn_pooling_params->pad_h - cudnn_pooling_params->poolsize_h) / cudnn_pooling_params->stride_h);
		top->W = 1 + ceil((bottom->W + 2 * cudnn_pooling_params->pad_w - cudnn_pooling_params->poolsize_w) / cudnn_pooling_params->stride_w);

		CUDNN_CHECK( cudnnSetTensor4dDescriptor(topTensorDesc,
				tensorFormat,
				dataType,
				top->N,
				top->C,
				top->H,
				top->W) );

		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}

	void Forward(const Blob_t *bottom, Blob_t *top) {
		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK( cudnnPoolingForward(cudnnHandle,
				poolingDesc,
				&alpha,
				bottomTensorDesc,
				bottom->data_gpu,
				&beta,
				topTensorDesc,
				top->data_gpu) );
	}

	void Backward(const Blob_t *top, Blob_t *bottom) {
		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK( cudnnPoolingBackward( cudnnHandle,
				poolingDesc,
				&alpha,
				topTensorDesc,
				top->data_gpu,
				topTensorDesc,
				top->diff_gpu,
				bottomTensorDesc,
				bottom->data_gpu,
				&beta,
				bottomTensorDesc,
				bottom->diff_gpu) );
	}
};

class FullyConnectedParameter_t
{
public:
	int hidden_size;
};

class FullyConnectedLayer_t
{
public:
	cublasHandle_t cublashandle;
	FullyConnectedParameter_t *fc_params;
	Blob_t *filtersBlob;
	Blob_t *biasBlob;
	Blob_t *bias_multiplier;
	int M_;
	int N_;
	int K_;
	FullyConnectedLayer_t(const FullyConnectedParameter_t *fc_params_) {
		fc_params = const_cast<FullyConnectedParameter_t *>(fc_params_);

		filtersBlob = NULL;
		biasBlob = NULL;
		bias_multiplier = NULL;

		M_ = 0;
		N_ = 0;
		K_ = 0;

		cublashandle = NULL;

		CUBLAS_CHECK( cublasCreate(&cublashandle) );
	}

	~FullyConnectedLayer_t() {
		delete filtersBlob; filtersBlob = NULL;
		delete biasBlob; biasBlob = NULL;
		delete bias_multiplier; bias_multiplier = NULL;

		CUBLAS_CHECK( cublasDestroy(cublashandle) );
	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		N_ = fc_params->hidden_size;
		K_ = bottom->C * bottom->H * bottom->W;
		M_ = bottom->N;
		filtersBlob = new Blob_t(1, 1, N_, K_);
		biasBlob = new Blob_t(1,1,1,N_);
		bias_multiplier = new Blob_t(1,1,1,M_);

		filtersBlob->allocate_gpu_data();
		filtersBlob->allocate_gpu_diff();
		biasBlob->allocate_gpu_data();
		biasBlob->allocate_gpu_diff();

		bias_multiplier->allocate_gpu_data();
		CUDA_CHECK( cudaMemset(bias_multiplier->data_gpu, (float)1.0f, M_ * sizeof(float)) );
		//bias_multiplier->allocate_gpu_diff();

		top->N = bottom->N;
		top->C = N_;
		top->H = 1;
		top->W = 1;
		top->allocate_gpu_data();
		top->allocate_gpu_diff();

	}

	void Forward(const Blob_t *bottom, Blob_t *top) {
		gpu_gemm(cublashandle, CblasNoTrans, CblasTrans, M_, N_, K_, (float)1.,
				bottom->data_gpu, filtersBlob->data_gpu, (float)0., top->data_gpu);
		gpu_gemm(cublashandle, CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
				bias_multiplier->data_gpu, biasBlob->data_gpu, (float)1., top->data_gpu);

	}

	void Backward(const Blob_t *top, Blob_t *bottom) {
		// Gradient with respect to weight
		gpu_gemm(cublashandle, CblasTrans, CblasNoTrans, N_, K_, M_, (float)1.,
				top->diff_gpu, bottom->data_gpu, (float)0., filtersBlob->diff_gpu);
		// Gradient with respect to bias
		gpu_gemv(cublashandle, CblasTrans, M_, N_, (float)1.,
				top->diff_gpu, bias_multiplier->data_gpu, (float)0., biasBlob->diff_gpu);
		// Gradient with respect to bottom data
		gpu_gemm(cublashandle, CblasNoTrans, CblasNoTrans, M_, K_, N_, (float)1.,
				top->diff_gpu, filtersBlob->data_gpu, (float)0., bottom->diff_gpu);
	}
};

class SoftmaxParameter_t
{
public:
	cudnnSoftmaxAlgorithm_t cudnn_softmax_algo;
	cudnnSoftmaxMode_t cudnn_softmax_mode;
};

class SoftmaxLayer_t : public Layer_t
{
public:
	SoftmaxParameter_t *cudnn_softmax_params;
	SoftmaxLayer_t(const SoftmaxParameter_t *cudnn_softmax_params_) {
		cudnn_softmax_params = const_cast<SoftmaxParameter_t *>(cudnn_softmax_params_);
	}

	~SoftmaxLayer_t() {

	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(bottomTensorDesc,
				tensorFormat,
				dataType,
				bottom->N,
				bottom->C,
				bottom->H,
				bottom->W) );

		top->N = bottom->N;
		top->C = bottom->C;
		top->H = bottom->H;
		top->W = bottom->W;
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(topTensorDesc,
				tensorFormat,
				dataType,
				top->N,
				top->C,
				top->H,
				top->W) );

		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}

	void Forward(const Blob_t *bottom, Blob_t *top) {
		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK( cudnnSoftmaxForward(cudnnHandle,
				cudnn_softmax_params->cudnn_softmax_algo ,
				cudnn_softmax_params->cudnn_softmax_mode,
				&alpha,
				bottomTensorDesc,
				bottom->data_gpu,
				&beta,
				topTensorDesc,
				top->data_gpu) );
		top->data_to_cpu();
	}

	void Backward(const Blob_t *top, Blob_t *bottom) {
		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK( cudnnSoftmaxBackward( cudnnHandle,
				cudnn_softmax_params->cudnn_softmax_algo ,
				cudnn_softmax_params->cudnn_softmax_mode,
				&alpha,
				topTensorDesc,
				top->data_gpu,
				topTensorDesc,
				top->diff_gpu,
				&beta,
				bottomTensorDesc,
				bottom->diff_gpu) );
	}
};

class MultinomialLogisticLossParameter_t
{
public:
	std::vector<int> ignore_labels;
	bool normalize;
};

class MultinomialLogisticLossLayer_t
{
public:
	MultinomialLogisticLossParameter_t *mlr_params;


	MultinomialLogisticLossLayer_t(const MultinomialLogisticLossParameter_t *mlr_params_) {
		mlr_params = const_cast<MultinomialLogisticLossParameter_t *>(mlr_params_);
	}

	~MultinomialLogisticLossLayer_t() {

	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		top->N = 1;
		top->C = 1;
		top->H = 1;
		top->W = 1;
		top->allocate_cpu_data();
		top->allocate_cpu_diff();
		top->data_cpu[0] = 1.0f;
	}

	void Forward(const Blob_t *bottom, const Blob_t *label, Blob_t *top) {
		int num = bottom->N;
		int dim = bottom->count() / bottom->N;
		float loss = 0.0f;
		for (int i = 0; i < num; ++i) {
			int truelabel = static_cast<int>(label->data_cpu[i]);
			float prob = std::max(bottom->data_cpu[i * dim + truelabel], (float)1e-20);
			loss -= log(prob);
		}
		top->data_cpu[0] = (loss / num);
	}

	void Backward(const Blob_t *top, const Blob_t *label, Blob_t *bottom) {
		if(bottom->diff_cpu == NULL) {
			bottom->allocate_cpu_diff();
		}
		int num = bottom->N;
		int dim = bottom->count() / bottom->N;
		const float scale = - top->diff_cpu[0] / num;
		for (int i = 0; i < num; ++i) {
			int truelabel = static_cast<int>(label->data_cpu[i]);
			float prob = std::max(bottom->data_cpu[i * dim + truelabel], (float)1e-20);
			bottom->diff_cpu[i * dim + truelabel] = scale / prob;
		}
		bottom->diff_to_gpu();
	}
};

class Network_t
{
public:
	string net_name;
	int gpu_id;
	cudaStream_t curand_stream;
	curandGenerator_t curand_generator;
	curandRngType_t curand_rngtype;
	cublasHandle_t cublas_handle;

	Blob_t *batch_samples;
	Blob_t *batch_labels;

	ConvolutionParameter_t *conv1_params;
	ConvolutionLayer_t *conv1;
	Blob_t *conv1_top;

	ActivationParameter_t *relu1_params;
	ActivationLayer_t *relu1;
	Blob_t *relu1_top;

	PoolingParameter_t *mp1_params;
	PoolingLayer_t *mp1;
	Blob_t *mp1_top;

	FullyConnectedParameter_t *ip1_params;
	FullyConnectedLayer_t *ip1;
	Blob_t *ip1_top;

	SoftmaxParameter_t *sm1_params;
	SoftmaxLayer_t *sm1;
	Blob_t *sm1_top;

	MultinomialLogisticLossParameter_t *mlr1_params;
	MultinomialLogisticLossLayer_t *mlr1;
	Blob_t *mlr1_top;


	Blob_t *conv1_filtersBlob_old;
	Blob_t *conv1_biasBlob_old;
	Blob_t *ip1_filtersBlob_old;
	Blob_t *ip1_biasBlob_old;


	Network_t(string net_name_, int gpu_id_ = 0) {
		net_name = net_name_;
		gpu_id = gpu_id_;
		curand_stream = NULL;
		curand_generator = NULL;
		curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
		cublas_handle = NULL;

		batch_samples = NULL;
		batch_labels = NULL;

		conv1 = NULL;
		conv1_top = NULL;
		conv1_params = NULL;
		relu1 = NULL;
		relu1_top = NULL;
		relu1_params = NULL;
		mp1 = NULL;
		mp1_top = NULL;
		mp1_params = NULL;
		ip1 = NULL;
		ip1_top = NULL;
		ip1_params = NULL;
		sm1 = NULL;
		sm1_top = NULL;
		sm1_params = NULL;
		mlr1 = NULL;
		mlr1_top = NULL;
		mlr1_params = NULL;


		conv1_filtersBlob_old = NULL;
		conv1_biasBlob_old = NULL;
		ip1_filtersBlob_old = NULL;
		ip1_biasBlob_old = NULL;

	}

	~Network_t() {
	}

	void DestroyNet() {

		cudaSetDevice(gpu_id);

		delete batch_samples; batch_samples = NULL;
		delete batch_labels; batch_labels = NULL;

		delete conv1; conv1 = NULL;
		delete relu1; relu1 = NULL;
		delete mp1; mp1 = NULL;
		delete ip1; ip1 = NULL;
		delete sm1; sm1 = NULL;
		delete mlr1; mlr1 = NULL;

		delete conv1_top; conv1_top = NULL;
		delete relu1_top; relu1_top = NULL;
		delete mp1_top; mp1_top = NULL;
		delete ip1_top; ip1_top = NULL;
		delete sm1_top; sm1_top = NULL;
		delete mlr1_top; mlr1_top = NULL;

		delete conv1_params; conv1_params = NULL;
		delete relu1_params; relu1_params = NULL;
		delete mp1_params; mp1_params = NULL;
		delete ip1_params; ip1_params = NULL;
		delete sm1_params; sm1_params = NULL;
		delete mlr1_params; mlr1_params = NULL;

		delete conv1_filtersBlob_old; conv1_filtersBlob_old = NULL;
		delete conv1_biasBlob_old; conv1_biasBlob_old = NULL;
		delete ip1_filtersBlob_old; ip1_filtersBlob_old = NULL;
		delete ip1_biasBlob_old; ip1_biasBlob_old = NULL;

		CURAND_CHECK( curandDestroyGenerator(curand_generator) );
		CUDA_CHECK( cudaStreamDestroy(curand_stream) );
		CUBLAS_CHECK( cublasDestroy(cublas_handle) );
	}

	void BuildNet(int batch_size_, const string &net_params_file = "") {
		cudaSetDevice(gpu_id);
		CUDA_CHECK( cudaStreamCreate(&curand_stream) );
		curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
		CURAND_CHECK( curandCreateGenerator(&curand_generator, curand_rngtype) );
		CURAND_CHECK( curandSetStream(curand_generator, curand_stream) );
		CUBLAS_CHECK( cublasCreate(&cublas_handle) );

		batch_samples = new Blob_t(batch_size_, 3, 32, 32);
		batch_labels = new Blob_t(batch_size_, 1, 1, 1);
		batch_samples->allocate_gpu_data();
		batch_samples->allocate_gpu_diff();
		batch_labels->allocate_gpu_data();

		printf("conv1 setup.\n");
		conv1_top = new Blob_t();
		conv1_params = new ConvolutionParameter_t();
		conv1_params->filter_N = 3;
		conv1_params->filter_C = 32;
		conv1_params->filter_H = 5;
		conv1_params->filter_W = 5;
		conv1_params->pad_h = 0;
		conv1_params->pad_w = 0;
		conv1_params->stride_h = 1;
		conv1_params->stride_w = 1;
		conv1_params->upscale_h = 1;
		conv1_params->upscale_w = 1;
		conv1_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv1 = new ConvolutionLayer_t(conv1_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->filtersBlob->data_gpu, conv1->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->biasBlob->data_gpu, conv1->biasBlob->count(), (float)0.0f, (float)0.01f) );
		conv1->Setup(batch_samples, conv1_top);


		printf("relu1 setup.\n");
		relu1_top = new Blob_t();
		relu1_params = new ActivationParameter_t();
		relu1_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu1 = new ActivationLayer_t(relu1_params);
		relu1->Setup(conv1_top, relu1_top);

		printf("mp1 setup.\n");
		mp1_top = new Blob_t();
		mp1_params = new PoolingParameter_t();
		mp1_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
		mp1_params->poolsize_h = 2;
		mp1_params->poolsize_w = 2;
		mp1_params->pad_h = 0;
		mp1_params->pad_w = 0;
		mp1_params->stride_h = 2;
		mp1_params->stride_w = 2;
		mp1 = new PoolingLayer_t(mp1_params);
		mp1->Setup(relu1_top, mp1_top);

		printf("ip1 setup.\n");
		ip1_top = new Blob_t();
		ip1_params = new FullyConnectedParameter_t();
		ip1_params->hidden_size = 10;
		ip1 = new FullyConnectedLayer_t(ip1_params);
		ip1->Setup(mp1_top, ip1_top);
		CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->filtersBlob->data_gpu, ip1->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );


		printf("sm1 setup.\n");
		sm1_top = new Blob_t();
		sm1_params = new SoftmaxParameter_t();
		sm1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
		sm1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
		sm1 = new SoftmaxLayer_t(sm1_params);
		sm1->Setup(ip1_top, sm1_top);

		printf("mlr1 setup (in cpu).\n");
		mlr1_top = new Blob_t();
		mlr1_params = new MultinomialLogisticLossParameter_t();
		mlr1 = new MultinomialLogisticLossLayer_t(mlr1_params);
		mlr1->Setup(sm1_top, mlr1_top);

		printf("initialize old net params.\n");
		conv1_filtersBlob_old = new Blob_t(conv1->filtersBlob->N, conv1->filtersBlob->C, conv1->filtersBlob->H, conv1->filtersBlob->W);
		conv1_biasBlob_old = new Blob_t(conv1->biasBlob->N, conv1->biasBlob->C, conv1->biasBlob->H, conv1->biasBlob->W);
		conv1_filtersBlob_old->allocate_gpu_data();
		conv1_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv1_filtersBlob_old->data_gpu, conv1_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv1_biasBlob_old->data_gpu, conv1_biasBlob_old->count(), 0.0f, 0.0f);

		ip1_filtersBlob_old = new Blob_t(ip1->filtersBlob->N, ip1->filtersBlob->C, ip1->filtersBlob->H, ip1->filtersBlob->W);
		ip1_biasBlob_old = new Blob_t(ip1->biasBlob->N, ip1->biasBlob->C, ip1->biasBlob->H, ip1->biasBlob->W);
		ip1_filtersBlob_old->allocate_gpu_data();
		ip1_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, ip1_filtersBlob_old->data_gpu, ip1_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, ip1_biasBlob_old->data_gpu, ip1_biasBlob_old->count(), 0.0f, 0.0f);

		printf("build net (done).\n");
	}

	float Forward() {
		cudaSetDevice(gpu_id);

		float loss = 0.0f;

		// printf("conv1 forward.\n");
		conv1->Forward(batch_samples, conv1_top);

		// printf("relu1 forward.\n");
		relu1->Forward(conv1_top, relu1_top);

		// printf("mp1 forward.\n");
		mp1->Forward(relu1_top, mp1_top);

		// printf("ip1 forward.\n");
		ip1->Forward(mp1_top, ip1_top);

		// printf("sm1 forward.\n");
		sm1->Forward(ip1_top, sm1_top);

		// printf("mlr1 forward.\n");
		mlr1->Forward(sm1_top, batch_labels, mlr1_top);

		loss = mlr1_top->data_cpu[0];

		return loss;
	}

	void Backward() {
		cudaSetDevice(gpu_id);

		// printf("mlr1 backward.\n");
		mlr1->Backward(mlr1_top, batch_labels, sm1_top);

		// printf("sm1 backward.\n");
		sm1->Backward(sm1_top, ip1_top);

		// printf("ip1 backward.\n");
		ip1->Backward(ip1_top, mp1_top);

		// printf("mp1 backward.\n");
		mp1->Backward(mp1_top, relu1_top);

		// printf("relu1 backward.\n");
		relu1->Backward(relu1_top, conv1_top);

		// printf("conv1 backward.\n");
		conv1->Backward(conv1_top, batch_samples);
	}

	float ForwardBackward() {
		float loss = Forward();
		Backward();
		return loss;
	}

	void ComputeUpdateValueSingle(Blob_t *param_gradient_blob, Blob_t *param_blob_old,
			float lr_rate, float momentum, float weight_decay) {
		gpu_axpy(cublas_handle,
				param_gradient_blob->count(), weight_decay,
				param_gradient_blob->data_gpu,
				param_gradient_blob->diff_gpu);

		gpu_axpby(cublas_handle,
				param_gradient_blob->count(), lr_rate,
				param_gradient_blob->diff_gpu, momentum,
				param_blob_old->data_gpu);
		// copy
		gpu_copy(param_gradient_blob->count(),
				param_blob_old->data_gpu,
				param_gradient_blob->diff_gpu);
	}
	void ComputeUpdateValue(float lr_rate, float momentum, float weight_decay) {
		cudaSetDevice(gpu_id);
		ComputeUpdateValueSingle(conv1->filtersBlob, conv1_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv1->biasBlob, conv1_biasBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(ip1->filtersBlob, ip1_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(ip1->biasBlob, ip1_biasBlob_old, lr_rate, momentum, weight_decay);
	}

	void UpdateNet() {
		cudaSetDevice(gpu_id);
		gpu_axpy(cublas_handle, conv1->filtersBlob->count(), float(-1), conv1->filtersBlob->diff_gpu, conv1->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv1->biasBlob->count(), float(-1), conv1->biasBlob->diff_gpu, conv1->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, ip1->filtersBlob->count(), float(-1), ip1->filtersBlob->diff_gpu, ip1->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, ip1->biasBlob->count(), float(-1), ip1->biasBlob->diff_gpu, ip1->biasBlob->data_gpu);
	}

	void CopyNetParamsFrom(const Network_t *other) {
		CopyBlobData_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv1->biasBlob, other->gpu_id, conv1->biasBlob, gpu_id);
		CopyBlobData_gpu(other->ip1->filtersBlob, other->gpu_id, ip1->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->ip1->biasBlob, other->gpu_id, ip1->biasBlob, gpu_id);
	}

	void AddNetParamsDiffFrom(const Network_t *other) {
		AddBlobDiff_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv1->biasBlob, other->gpu_id, conv1->biasBlob, gpu_id);
		AddBlobDiff_gpu(other->ip1->filtersBlob, other->gpu_id, ip1->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->ip1->biasBlob, other->gpu_id, ip1->biasBlob, gpu_id);
	}

	void ClearNetParamsDiff() {
		cudaSetDevice(gpu_id);
		CUDA_CHECK( cudaMemset(conv1->filtersBlob->diff_gpu, 0, conv1->filtersBlob->count() * sizeof(float)) );
		CUDA_CHECK( cudaMemset(conv1->biasBlob->diff_gpu, 0, conv1->biasBlob->count() * sizeof(float)) );
		CUDA_CHECK( cudaMemset(ip1->filtersBlob->diff_gpu, 0, ip1->filtersBlob->count() * sizeof(float)) );
		CUDA_CHECK( cudaMemset(ip1->biasBlob->diff_gpu, 0, ip1->biasBlob->count() * sizeof(float)) );
	}

};

struct thread_data_t
{
public:
	Blob_t *batch_samples;
	Blob_t *batch_labels;
	Network_t *net;
	int current_gpu_id;
	int net_gpu_id;
	float lr_rate;
	float momentum;
	float weight_decay;
};

void do_slave(void *data_)
{
	thread_data_t *data = (thread_data_t *)data_;
	cudaSetDevice(data->net_gpu_id);
	CUDA_CHECK( cudaMemcpy(data->net->batch_samples->data_gpu, data->batch_samples->data_cpu, data->batch_samples->count() * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(data->net->batch_labels->data_gpu, data->batch_labels->data_cpu, data->batch_labels->count() * sizeof(float), cudaMemcpyHostToDevice) );
	float trn_loss = data->net->ForwardBackward();
	printf("trn_loss: %.6f\n", trn_loss);
	data->net->ComputeUpdateValue(data->lr_rate, data->momentum, data->weight_decay);
}


int main_test_data_layer_ok(int argc, char *argv[]) {
	if(argc != 12) {
		printf("Usage: <filename> trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n");
		return -1;
	}
	string trn_db_filename = string(argv[1]);
	string tst_db_filename = string(argv[2]);
	string mean_file = string(argv[3]);
	float lr_rate = atof(argv[4]);
	int lr_stepsize = atoi(argv[5]);
	float momentum = atof(argv[6]);
	float weight_decay = atof(argv[7]);
	int trn_batch_size = atoi(argv[8]);
	int tst_batch_size = atoi(argv[9]);
	int max_epoch_num = atoi(argv[10]);
	string gpu_ids_str = string(argv[11]);

	Blob_t *batch_samples = new Blob_t();
	Blob_t *batch_labels = new Blob_t();
	DataLayerParameter_t *data_param = new DataLayerParameter_t();
	data_param->backend = "lmdb";
	data_param->batch_size = trn_batch_size;
	data_param->source = trn_db_filename;
	data_param->mean_file = mean_file;
	DataLayer_t *trn_data_layer = new DataLayer_t(data_param);
	trn_data_layer->Setup();
	printf("forward datalayer.\n");
	trn_data_layer->Forward_cpu(batch_samples, batch_labels);
	printf("forward datalayer(done).\n");

	batch_samples->print_cpu_data(100);
	batch_labels->print_cpu_data(100);

	delete data_param; data_param = NULL;
	delete batch_samples; batch_samples = NULL;
	delete batch_labels; batch_labels = NULL;
	delete trn_data_layer; trn_data_layer = NULL;
	return 0;
}

int main_test_multigpu_ok(int argc, char *argv[]) {
	if(argc != 12) {
		printf("Usage: <filename> trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n");
		return -1;
	}
	string trn_db_filename = string(argv[1]);
	string tst_db_filename = string(argv[2]);
	string mean_file = string(argv[3]);
	float lr_rate = atof(argv[4]);
	int lr_stepsize = atoi(argv[5]);
	float momentum = atof(argv[6]);
	float weight_decay = atof(argv[7]);
	int trn_batch_size = atoi(argv[8]);
	int tst_batch_size = atoi(argv[9]);
	int max_epoch_num = atoi(argv[10]);
	string gpu_ids_str = string(argv[11]);


	int current_gpu_id;
	cudaGetDevice(&current_gpu_id);
	printf("current gpu id: %d\n", current_gpu_id);

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	printf("number of manually-set gpus: %ld, total %d gpus.\n", gpus.size(), num_gpus);

	if(num_gpus >= gpus.size()) {
		printf("enable P2P: ");
		EnableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	} else {
		gpus.clear();
		gpus.push_back(current_gpu_id);
	}

	cudaSetDevice(current_gpu_id);

	vector<Network_t *> trn_nets(gpus.size());
	for(int i = 0; i < gpus.size(); i++) {
		trn_nets[i] = NULL;
	}
	printf("initialize nets for each gpu ...\n");
	for(int i = 0; i < gpus.size(); i++)
	{
		printf("=========== gpu [%d] ==============\n", gpus[i]);
		cudaSetDevice(gpus[i]);
		trn_nets[i] = new Network_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(trn_batch_size, "");
		trn_nets[i]->batch_labels->allocate_cpu_data();
	}
	printf("initialize nets for each gpu (done) ...\n");

	cudaSetDevice(current_gpu_id);

	pthread_t *threads;
	pthread_attr_t pta;
	threads = (pthread_t *) malloc(sizeof(pthread_t) * gpus.size());
	int ret_count = pthread_attr_init(&pta);
	thread_data_t thread_data[gpus.size()];

	// prepare batch data, should use blocking queue
	Blob_t *batch_samples = new Blob_t(trn_batch_size, 3, 32, 32);
	Blob_t *batch_labels  = new Blob_t(trn_batch_size, 1, 1, 1);
	batch_samples->allocate_cpu_data();
	batch_labels->allocate_cpu_data();
	for(int n = 0; n < batch_samples->N; n++) {
		for(int c = 0; c < batch_samples->C; c++) {
			for(int h = 0; h < batch_samples->H; h++) {
				for(int w = 0; w < batch_samples->W; w++) {
					int index = (((n)*batch_samples->C+c)*batch_samples->H+h)*batch_samples->W + w;
					batch_samples->data_cpu[index] = (float)rand() / (float)RAND_MAX;
				}
			}
		}
		batch_labels->data_cpu[n] = n;
	}

	for(int i = 0; i < gpus.size(); i++) {
		thread_data[i].lr_rate = lr_rate;
		thread_data[i].momentum = momentum;
		thread_data[i].weight_decay = weight_decay;
		thread_data[i].current_gpu_id = current_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples;
		thread_data[i].batch_labels = batch_labels;

		ret_count = pthread_create(&threads[i], &pta, (void*(*)(void*))do_slave, (void*)(&(thread_data[i])));
	}

	for(int i = 0; i < gpus.size(); i++) {
		ret_count = pthread_join(threads[i], NULL);
	}

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(current_gpu_id);
	delete batch_samples;
	delete batch_labels;

	if(num_gpus >= gpus.size()) {
		printf("disable P2P: ");
		DisableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	}
	cudaDeviceReset();
	return 0;
}

int main(int argc, char *argv[]) {
	if(argc != 12) {
		printf("Usage: <filename> trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n");
		return -1;
	}
	string trn_db_filename = string(argv[1]);
	string tst_db_filename = string(argv[2]);
	string mean_file = string(argv[3]);
	float lr_rate = atof(argv[4]);
	int lr_stepsize = atoi(argv[5]);
	float momentum = atof(argv[6]);
	float weight_decay = atof(argv[7]);
	int trn_batch_size = atoi(argv[8]);
	int tst_batch_size = atoi(argv[9]);
	int max_epoch_num = atoi(argv[10]);
	string gpu_ids_str = string(argv[11]);


	int current_gpu_id;
	cudaGetDevice(&current_gpu_id);
	printf("current gpu id: %d\n", current_gpu_id);

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	printf("number of manually-set gpus: %ld, total %d gpus.\n", gpus.size(), num_gpus);

	if(num_gpus >= gpus.size()) {
		printf("enable P2P: ");
		EnableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	} else {
		gpus.clear();
		gpus.push_back(current_gpu_id);
	}

	cudaSetDevice(current_gpu_id);

	vector<Network_t *> trn_nets(gpus.size());
	vector<Blob_t *> batch_samples_slices(gpus.size());
	vector<Blob_t *> batch_labels_slices(gpus.size());
	vector<int> batch_sizes(gpus.size());
	for(int i = 0; i < gpus.size(); i++) {
		trn_nets[i] = NULL;
		batch_samples_slices[i] = NULL;
		batch_labels_slices[i] = NULL;
		batch_sizes[i] = 0;
	}
	printf("initialize nets for each gpu ...\n");
	for(int i = 0; i < gpus.size(); i++)
	{
		printf("=========== gpu [%d] ==============\n", gpus[i]);
		cudaSetDevice(gpus[i]);

		batch_samples_slices[i] = new Blob_t();
		batch_labels_slices[i] = new Blob_t();
		batch_sizes[i] = trn_batch_size / gpus.size();

		trn_nets[i] = new Network_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(batch_sizes[i], "");
		trn_nets[i]->batch_labels->allocate_cpu_data();
	}
	printf("initialize nets for each gpu (done) ...\n");

	cudaSetDevice(current_gpu_id);

	Blob_t *trn_batch_samples = new Blob_t();
	Blob_t *trn_batch_labels = new Blob_t();
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	Blob_t *tst_batch_samples = new Blob_t();
	Blob_t *tst_batch_labels = new Blob_t();
	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	Network_t *trn_net = new Network_t("trn_net", current_gpu_id);
	trn_net->BuildNet(trn_batch_size, "");
	trn_net->batch_labels->allocate_cpu_data();
	Network_t *tst_net = new Network_t("tst_net", current_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");
	tst_net->batch_labels->allocate_cpu_data();

	pthread_t *threads;
	pthread_attr_t pta;
	threads = (pthread_t *) malloc(sizeof(pthread_t) * gpus.size());
	int ret_count = pthread_attr_init(&pta);
	thread_data_t thread_data[gpus.size()];
	for(int i = 0; i < gpus.size(); i++) {
		thread_data[i].lr_rate = lr_rate;
		thread_data[i].momentum = momentum;
		thread_data[i].weight_decay = weight_decay;
		thread_data[i].current_gpu_id = current_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples_slices[i];
		thread_data[i].batch_labels = batch_labels_slices[i];
	}

	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f;
		tst_net->CopyNetParamsFrom(trn_net);
		for(int iter = 0; iter < floor(10000 / tst_batch_size); iter++) {
			tst_data_layer->Forward_cpu(tst_batch_samples, tst_batch_labels);
			tst_loss += tst_net->Forward();
		}

		// training net
		for(int iter = 0; iter < floor(50000 / trn_batch_size); iter++) {
			trn_data_layer->Forward_cpu_multi(batch_samples_slices, batch_labels_slices, batch_sizes);

			trn_net->ClearNetParamsDiff();

			// copy trn_net params into trn_nets_i
			for(int i = 0; i < gpus.size(); i++) {
				trn_nets[i]->CopyNetParamsFrom(trn_net);
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_create(&threads[i], &pta, (void*(*)(void*))do_slave, (void*)(&(thread_data[i])));
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_join(threads[i], NULL);
			}

			cudaDeviceSynchronize();
			cudaSetDevice(current_gpu_id);
			// copy update values from each sub nets to the main trn_net
			for(int i = 0; i < gpus.size(); i++) {
				trn_net->AddNetParamsDiffFrom(trn_nets[i]);
			}
			trn_net->UpdateNet();
		}
	}

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(current_gpu_id);
	delete trn_batch_samples;
	delete trn_batch_labels;
	delete tst_batch_samples;
	delete tst_batch_labels;
	delete trn_net;
	delete tst_net;

	delete trn_data_param; trn_data_param = NULL;
	delete trn_data_layer; trn_data_layer = NULL;
	delete tst_data_param; tst_data_param = NULL;
	delete tst_data_layer; tst_data_layer = NULL;

	if(num_gpus >= gpus.size()) {
		printf("disable P2P: ");
		DisableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	}
	free(threads); threads = NULL;
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}
