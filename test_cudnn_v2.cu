
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
			LOG(FATAL) << "CUDA ERROR: " << cudaGetErrorString(error); \
		} while (0)

#define CUBLAS_CHECK(condition) \
		do { \
			cublasStatus_t status = condition; \
			if (status != CUBLAS_STATUS_SUCCESS) \
			LOG(FATAL) << "CUBLAS ERROR: " << cublasGetErrorString(status); \
		} while (0)

#define CURAND_CHECK(condition) \
		do { \
			curandStatus_t status = condition; \
			if (status != CURAND_STATUS_SUCCESS) \
			LOG(FATAL) << "CURAND ERROR: " << curandGetErrorString(status); \
		} while (0)

#define CUDNN_CHECK(status) \
		do { \
			if (status != CUDNN_STATUS_SUCCESS) \
			LOG(FATAL) << "CUDNN ERROR: " << cudnnGetErrorString(status); \
		} while (0)


inline void MallocHost(void** ptr, size_t size) {
  *ptr = malloc(size);
}

inline void FreeHost(void* ptr) {
  free(ptr);
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
					// printf("(%d <--> %d): YES!\n", gpus[i], gpus[j]);
					LOG(INFO) << gpus[i] << " <--> " << gpus[j] << ": UVA YES!\n";
				}
			} else {
				// printf("(%d <--> %d): NO!\n", gpus[i], gpus[j]);
				LOG(INFO) << gpus[i] << " <--> " << gpus[j] << ": UVA NO!\n";
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

__global__ void set_kernel(const int n, const float alpha, float* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = alpha;
	}
}
void cpu_set(const int N, const float alpha, float *Y) {
#pragma omp parallel
	for(int i = 0; i < N; i++) {
		Y[i] = alpha;
	}
}
void gpu_set(const int N, const float alpha, float* Y) {
	set_kernel<<<GPU_GET_BLOCKS(N), GPU_CUDA_NUM_THREADS>>>(N, alpha, Y);
}

void gpu_copy(const int N, const float *X, float *Y) {
	CUDA_CHECK( cudaMemcpy(Y, X, sizeof(float) * N, cudaMemcpyDefault) );
}

void gpu_asum(cublasHandle_t cublashandle, const int n, const float* x, float* y) {
	CUBLAS_CHECK(cublasSasum(cublashandle, n, x, 1, y));
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
			FreeHost(data_cpu);
			data_cpu = NULL;
		}
		if(data_gpu != NULL)
		{
			CUDA_CHECK( cudaFree(data_gpu) );
			data_gpu = NULL;
		}
		if(diff_cpu != NULL)
		{
			FreeHost(diff_cpu);
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
			return;
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
			return;

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
			return;
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

	void save_cpu_data_and_diff_to_mat(const char *fname, bool is_save_diff = false)
	{
		data_to_cpu();

		mat_t *matfp = Mat_Create(fname, 0);
		//matfp = Mat_CreateVer(fname, 0, MAT_FT_MAT73);
		size_t dims[4];
		dims[0] = W;
		dims[1] = H;
		dims[2] = C;
		dims[3] = N;
		matvar_t *matvar;
		// save data
		matvar = Mat_VarCreate("data", matio_class_map<float>(), matio_type_map<float>(), 4, dims, data_cpu, 0);
		if(matvar == NULL)
			LOG(FATAL) << "Error creating 'data' variable";
		if(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0)
			LOG(FATAL) << "Error saving array 'data' into MAT file " << fname;
		Mat_VarFree(matvar);

		// save diff
		if(is_save_diff) {
			diff_to_cpu();

			matvar_t *matvar2;
			matvar2 = Mat_VarCreate("diff", matio_class_map<float>(), matio_type_map<float>(), 4, dims, diff_cpu, 0);
			if(matvar2 == NULL)
				LOG(FATAL) << "Error creating 'diff' variable";
			if(Mat_VarWrite(matfp, matvar2, MAT_COMPRESSION_NONE) != 0)
				LOG(FATAL) << "Error saving array 'diff' into MAT file " << fname;
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
			FreeHost(data_cpu);
		MallocHost((void**)&data_cpu, count * sizeof(float));
		cpu_set(count, 0, data_cpu);
	}

	void allocate_cpu_diff()
	{
		int count = N * C * H * W;
		if(diff_cpu != NULL)
			FreeHost(diff_cpu);
		MallocHost((void**)&diff_cpu, count * sizeof(float));
		cpu_set(count, 0, data_cpu);
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
			MallocHost((void**)&data_cpu, count * sizeof(float));
		if(data_gpu != NULL)
			CUDA_CHECK( cudaMemcpy(data_cpu, data_gpu, count * sizeof(float), cudaMemcpyDeviceToHost) );
	}

	void diff_to_cpu()
	{
		int count = N * C * H * W;
		if(diff_cpu == NULL)
			MallocHost((void**)&diff_cpu, count * sizeof(float));
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
		const bool has_uva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
		if(can_access_peer || has_uva) {
			cudaSetDevice(src_gpu_id);
			cudaMemcpy(dst->data_gpu, src->data_gpu, count * sizeof(float), cudaMemcpyDefault);
		} else {
			float *temp_data = NULL;
			cudaSetDevice(src_gpu_id);
			MallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->data_gpu, count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMemcpy(dst->data_gpu, temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			FreeHost(temp_data);
		}
	}
}

void AddBlobDiff_gpu(const Blob_t *src, int src_gpu_id, Blob_t *dst, int dst_gpu_id)
{
	int count = src->count();
	if(src_gpu_id == dst_gpu_id) {
		cudaSetDevice(dst_gpu_id);
		gpu_add(count, src->diff_gpu, dst->diff_gpu, dst->diff_gpu);
	} else {
		cudaDeviceProp prop[2];
		cudaGetDeviceProperties(&prop[0], src_gpu_id);
		cudaGetDeviceProperties(&prop[1], dst_gpu_id);
		int can_access_peer;
		cudaDeviceCanAccessPeer(&can_access_peer, src_gpu_id, dst_gpu_id);
		const bool has_uva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
		if(can_access_peer || has_uva) {
			cudaSetDevice(dst_gpu_id);
			float *temp_data_in_dst = NULL;
			CUDA_CHECK( cudaMalloc((void**)&temp_data_in_dst, count * sizeof(float)) );
			CUDA_CHECK( cudaMemcpy(temp_data_in_dst, src->diff_gpu, count * sizeof(float), cudaMemcpyDefault) );
			gpu_add(count, temp_data_in_dst, dst->diff_gpu, dst->diff_gpu);
			CUDA_CHECK( cudaFree(temp_data_in_dst) );
		} else {
			float *temp_data = NULL;
			float *dst_temp_data = NULL;
			cudaSetDevice(src_gpu_id);
			MallocHost((void **)&temp_data, count * sizeof(float));
			CUDA_CHECK( cudaMemcpy(temp_data, src->diff_gpu, count * sizeof(float), cudaMemcpyDeviceToHost) );
			cudaSetDevice(dst_gpu_id);
			CUDA_CHECK( cudaMalloc((void **)&dst_temp_data, count * sizeof(float)) );
			CUDA_CHECK( cudaMemcpy(dst_temp_data, temp_data, count * sizeof(float), cudaMemcpyHostToDevice) );
			gpu_add(count, dst_temp_data, dst->diff_gpu, dst->diff_gpu);
			FreeHost(temp_data);
			CUDA_CHECK( cudaFree(dst_temp_data) );
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
	int crop_size;
};

class DataLayer_t : public InternalThread
{
public:
	DataLayerParameter_t *data_params;
	Blob_t *prefetch_data_;
	Blob_t *prefetch_label_;
	Blob_t *mean_;
	int datum_size_;
	int datum_channels_;
	int datum_height_;
	int datum_width_;

	shared_ptr<db::DB> db_;
	shared_ptr<db::Cursor> cursor_;

	DataLayer_t(const DataLayerParameter_t *data_params_) {
		data_params = const_cast<DataLayerParameter_t *>(data_params_);

		prefetch_data_ = NULL;
		prefetch_label_ = NULL;
		mean_ = NULL;
		datum_size_ = 0;
		datum_channels_ = 0;
		datum_height_ = 0;
		datum_width_ = 0;
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
		datum_channels_ = datum.channels();
		datum_height_ = datum.height();
		datum_width_ = datum.width();

		if(data_params->crop_size && data_params->crop_size < datum_height_ && data_params->crop_size < datum_width_) {
			datum_size_ = datum.channels() * data_params->crop_size * data_params->crop_size;
			prefetch_data_ = new Blob_t(data_params->batch_size, datum_channels_, data_params->crop_size, data_params->crop_size);
			prefetch_label_ = new Blob_t(data_params->batch_size, 1, 1, 1);

		} else {
			datum_size_ = datum_channels_ * datum_height_ * datum_width_;
			prefetch_data_ = new Blob_t(data_params->batch_size, datum_channels_, datum_height_, datum_width_);
			prefetch_label_ = new Blob_t(data_params->batch_size, 1, 1, 1);
		}
		prefetch_data_->allocate_cpu_data();
		prefetch_label_->allocate_cpu_data();
		LOG(INFO) << "prefetch_data_ size: "
				<< prefetch_data_->N << ", "
				<< prefetch_data_->C << ", "
				<< prefetch_data_->H << ", "
				<< prefetch_data_->W;

		mean_ = new Blob_t(1, datum_channels_, datum_height_, datum_width_);
		mean_->allocate_cpu_data();
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(data_params->mean_file.c_str(), &blob_proto);
		for (int i = 0; i < mean_->count(); ++i) {
			mean_->data_cpu[i] = (float)blob_proto.data(i);
		}
		LOG(INFO) << "mean_ size: "
				<< mean_->N << ", "
				<< mean_->C << ", "
				<< mean_->H << ", "
				<< mean_->W;

		CreatePrefetchThread();
	}

	void Forward_cpu(Blob_t *top_data, Blob_t *top_label) {
		// printf("First, join the thread.\n");
		JoinPrefetchThread();

		// printf("copy data to top_data.\n");
		memcpy(top_data->data_cpu, prefetch_data_->data_cpu, prefetch_data_->count() * sizeof(float));

		// printf("copy label to top_label.\n");
		memcpy(top_label->data_cpu, prefetch_label_->data_cpu, prefetch_label_->count() * sizeof(float));

		// printf("Start a new prefetch thread.\n");
		CreatePrefetchThread();
	}

	void Forward_to_Network(Blob_t *top_data, Blob_t *top_label) {

		JoinPrefetchThread();

		CUDA_CHECK( cudaMemcpy(top_data->data_gpu, prefetch_data_->data_cpu, prefetch_data_->count() * sizeof(float), cudaMemcpyDefault) );

		CUDA_CHECK( cudaMemcpy(top_label->data_gpu, prefetch_label_->data_cpu, prefetch_label_->count() * sizeof(float), cudaMemcpyDefault) );

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
			memcpy(top_data[i]->data_cpu,
					prefetch_data_->data_cpu + start_index * top_data[i]->C * top_data[i]->H * top_data[i]->W,
					top_data[i]->count() * sizeof(float));

			// printf("copy label to top_label.\n");
			memcpy(top_label[i]->data_cpu,
					prefetch_label_->data_cpu + start_index * top_label[i]->C * top_label[i]->H * top_label[i]->W,
					top_label[i]->count() * sizeof(float));
		}
		// printf("Start a new prefetch thread.\n");
		CreatePrefetchThread();
	}

	void Forward_to_Network_multi(vector<int> &gpus, vector<int> &batch_sizes, vector<Blob_t *> &top_data, vector<Blob_t *> &top_label) {
		JoinPrefetchThread();

		for(int i = 0; i < batch_sizes.size(); i++) {
			int start_index = 0;
			for(int j = 0; j < i; j++) {
				start_index += batch_sizes[j];
			}

			cudaSetDevice(gpus[i]);
			CUDA_CHECK( cudaMemcpy(top_data[i]->data_gpu,
					prefetch_data_->data_cpu + start_index * top_data[i]->C * top_data[i]->H * top_data[i]->W,
					top_data[i]->count() * sizeof(float), cudaMemcpyDefault) );

			CUDA_CHECK( cudaMemcpy(top_label[i]->data_gpu,
					prefetch_label_->data_cpu + start_index * top_label[i]->C * top_label[i]->H * top_label[i]->W,
					top_label[i]->count() * sizeof(float), cudaMemcpyDefault) );
		}
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
			if(data_params->crop_size && data_params->crop_size < datum_height_ && data_params->crop_size < datum_width_) {
				int h_offset = rand() % (datum_height_ - data_params->crop_size);
				int w_offset = rand() % (datum_width_ - data_params->crop_size);
				if(data.size()) {
					for (int c = 0; c < datum_channels_; ++c) {
						for (int h = 0; h < data_params->crop_size; ++h) {
							for (int w = 0; w < data_params->crop_size; ++w) {
								int index = (c * datum_height_ + h + h_offset) * datum_width_ + w + w_offset;
								top_data[((item_id * datum_channels_ + c) * data_params->crop_size + h) * data_params->crop_size + w] =
										static_cast<float>((uint8_t)data[index]) - mean_data[index];
							}
						}
					}
				} else {
					for (int c = 0; c < datum_channels_; ++c) {
						for (int h = 0; h < data_params->crop_size; ++h) {
							for (int w = 0; w < data_params->crop_size; ++w) {
								int index = (c * datum_height_ + h + h_offset) * datum_width_ + w + w_offset;
								top_data[((item_id * datum_channels_ + c) * data_params->crop_size + h) * data_params->crop_size + w] =
										data[index] - mean_data[index];
							}
						}
					}
				}
			} else {
				if (data.size()) {
					for (int j = 0; j < datum_size_; ++j) {
						top_data[item_id * datum_size_ + j] = (static_cast<float>((uint8_t)data[j])) - mean_data[j];
					}
				} else {
					for (int j = 0; j < datum_size_; ++j) {
						top_data[item_id * datum_size_ + j] = (datum.float_data(j)) - mean_data[j];
					}
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
	float filter_lr_mult;
	float filter_weight_decay_mult;
	float bias_lr_mult;
	float bias_weight_decay_mult;
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

		// add bias
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(biasTensorDesc,
				tensorFormat,
				dataType,
				1,
				top->C,
				1,
				1) );

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


__global__ void sync_conv_groups() { }

class ConvolutionWithGroupParameter_t
{
public:
	int group;
	int filter_N;
	int filter_C;
	int filter_H;
	int filter_W;
	int pad_h, pad_w;
	int stride_h, stride_w;
	int upscale_h, upscale_w;
	float filter_lr_mult;
	float filter_weight_decay_mult;
	float bias_lr_mult;
	float bias_weight_decay_mult;
	cudnnConvolutionMode_t cudnn_conv_mode;
	cudnnConvolutionFwdPreference_t cudnn_conv_fwd_preference; // CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
};

class ConvolutionWithGroupLayer_t
{
public:
	Blob_t *filtersBlob;
	Blob_t *biasBlob;

	int CUDNN_STREAMS_PER_GROUP;
	bool handles_setup_;
	cudnnHandle_t* handle_;
	cudaStream_t*  stream_;

	cudnnDataType_t dataType;
	cudnnTensorFormat_t tensorFormat;

	cudnnTensorDescriptor_t bottomTensorDesc;
	cudnnTensorDescriptor_t topTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnTensorDescriptor_t    biasTensorDesc;
	cudnnConvolutionDescriptor_t convDesc;
	int bottom_offset_, top_offset_, weight_offset_, bias_offset_;

	ConvolutionWithGroupParameter_t *convwithgroup_params;


	ConvolutionWithGroupLayer_t(const ConvolutionWithGroupParameter_t *convwithgroup_params_)
	{
		CUDNN_STREAMS_PER_GROUP = 3;
		convwithgroup_params = const_cast<ConvolutionWithGroupParameter_t*>(convwithgroup_params_);
		filtersBlob = new Blob_t(convwithgroup_params->filter_N, convwithgroup_params->filter_C, convwithgroup_params->filter_H, convwithgroup_params->filter_W);
		biasBlob = new Blob_t(1, convwithgroup_params->filter_C, 1, 1);

		filtersBlob->allocate_gpu_data();
		filtersBlob->allocate_gpu_diff();
		biasBlob->allocate_gpu_data();
		biasBlob->allocate_gpu_diff();

		dataType = CUDNN_DATA_FLOAT;
		tensorFormat = CUDNN_TENSOR_NCHW;

		// Initialize CUDA streams and cuDNN.
		stream_         = new cudaStream_t[convwithgroup_params->group * CUDNN_STREAMS_PER_GROUP];
		handle_         = new cudnnHandle_t[convwithgroup_params->group * CUDNN_STREAMS_PER_GROUP];

		for (int g = 0; g < convwithgroup_params->group * CUDNN_STREAMS_PER_GROUP; g++) {
			CUDA_CHECK(cudaStreamCreate(&stream_[g]));
			CUDNN_CHECK(cudnnCreate(&handle_[g]));
			CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
		}
		CUDNN_CHECK( cudnnCreateTensorDescriptor(&bottomTensorDesc) );
		CUDNN_CHECK( cudnnCreateTensorDescriptor(&topTensorDesc) );
		CUDNN_CHECK( cudnnCreateConvolutionDescriptor(&convDesc) );
		CUDNN_CHECK( cudnnCreateTensorDescriptor(&biasTensorDesc) );
		CUDNN_CHECK( cudnnCreateFilterDescriptor(&filterDesc) );

		handles_setup_ = true;
		bottom_offset_ = 0;
		top_offset_ = 0;
		weight_offset_ = 0;
		bias_offset_ = 0;
	};

	~ConvolutionWithGroupLayer_t()
	{
		delete filtersBlob; filtersBlob = NULL;
		delete biasBlob; biasBlob = NULL;

		// Check that handles have been setup before destroying.
		if (!handles_setup_) { return; }

		CUDNN_CHECK( cudnnDestroyTensorDescriptor(bottomTensorDesc) );
		CUDNN_CHECK( cudnnDestroyTensorDescriptor(topTensorDesc) );
		CUDNN_CHECK( cudnnDestroyConvolutionDescriptor(convDesc) );
		CUDNN_CHECK( cudnnDestroyTensorDescriptor(biasTensorDesc) );
		CUDNN_CHECK( cudnnDestroyFilterDescriptor(filterDesc) );

		for (int g = 0; g < convwithgroup_params->group * CUDNN_STREAMS_PER_GROUP; g++) {
			cudaStreamDestroy(stream_[g]);
			cudnnDestroy(handle_[g]);
		}

		delete [] stream_;
		delete [] handle_;

	}


	void Setup(const Blob_t *bottom, Blob_t *top)
	{

//		// Set the indexing parameters.
//		weight_offset_ = (this->num_output_ / this->group_)
//		    		  * (this->channels_ / this->group_) * this->kernel_h_ * this->kernel_w_;
//		bias_offset_ = (this->num_output_ / this->group_);
		// Set the indexing parameters.
		// printf("get weight_offset_ and bias_offst_\n");
		weight_offset_ = (convwithgroup_params->filter_C / convwithgroup_params->group)
		    		  * (convwithgroup_params->filter_N / convwithgroup_params->group)
		    		  * convwithgroup_params->filter_H * convwithgroup_params->filter_W;
		bias_offset_ = (convwithgroup_params->filter_C / convwithgroup_params->group);
		// printf("get weight_offset_ and bias_offst_(%d, %d)\n", weight_offset_, bias_offset_);

		// printf("create bottomTensorDesc\n");
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(bottomTensorDesc,
				tensorFormat,
				dataType,
				bottom->N,
				bottom->C / convwithgroup_params->group,
				bottom->H,
				bottom->W) );

		// printf("create filterDesc\n");
		CUDNN_CHECK( cudnnSetFilter4dDescriptor(filterDesc,
				dataType,
				filtersBlob->C / convwithgroup_params->group,
				bottom->C / convwithgroup_params->group,
				filtersBlob->H,
				filtersBlob->W) );


		// printf("create convDesc\n");
		CUDNN_CHECK( cudnnSetConvolution2dDescriptor(convDesc,
				convwithgroup_params->pad_h, // padding
				convwithgroup_params->pad_w,
				convwithgroup_params->stride_h, // stride
				convwithgroup_params->stride_w,
				convwithgroup_params->upscale_h, // upscale
				convwithgroup_params->upscale_w,
				convwithgroup_params->cudnn_conv_mode) );

		// printf("get top shape\n");
		// find dimension of convolution output
		CUDNN_CHECK( cudnnGetConvolution2dForwardOutputDim(convDesc,
				bottomTensorDesc,
				filterDesc,
				&(top->N),
				&(top->C),
				&(top->H),
				&(top->W)) );
		top->C = filtersBlob->C;
		// printf("get top shape (%d, %d, %d, %d)\n", top->N, top->C, top->H, top->W);

		// printf("create topTensorDesc\n");
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(topTensorDesc,
				tensorFormat,
				dataType,
				top->N,
				top->C,
				top->H,
				top->W) );

		// printf("create biasTensorDesc\n");
		// add bias
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(biasTensorDesc,
				tensorFormat,
				dataType,
				1,
				top->C,
				1,
				1) );

		top->allocate_gpu_data();
		top->allocate_gpu_diff();

//		bottom_offset_ = (this->channels_ / this->group_)
//		    		  * this->height_ * this->width_;
//		top_offset_ = (this->num_output_ / this->group_)
//		    		  * this->height_out_ * this->width_out_;
		// printf("get bottom_offset_ and top_offset_\n");
		bottom_offset_ = (bottom->C / convwithgroup_params->group)
		    		  * bottom->H * bottom->W;
		top_offset_ = (top->C / convwithgroup_params->group)
		    		  * top->H * top->W;
	}

	void Forward(const Blob_t *bottom, Blob_t *top)
	{
	    const float* bottom_data = bottom->data_gpu;
	    float* top_data = top->data_gpu;
	    const float* weight = filtersBlob->data_gpu;
	    const float* bias_data = biasBlob->data_gpu;
		for(int g = 0; g < convwithgroup_params->group; g++) {
			cudnnConvolutionFwdAlgo_t algo;
			CUDNN_CHECK( cudnnGetConvolutionForwardAlgorithm(handle_[g],
					bottomTensorDesc,
					filterDesc,
					convDesc,
					topTensorDesc,
					convwithgroup_params->cudnn_conv_fwd_preference,
					0,
					&algo ) );

			size_t sizeInBytes=0;
			void* workSpace=NULL;
			CUDNN_CHECK( cudnnGetConvolutionForwardWorkspaceSize(handle_[g],
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
			CUDNN_CHECK( cudnnConvolutionForward(handle_[g],
					&alpha,
					bottomTensorDesc,
					bottom_data + bottom_offset_ * g,
					filterDesc,
					weight + weight_offset_ * g,
					convDesc,
					algo,
					workSpace,
					sizeInBytes,
					&beta,
					topTensorDesc,
					top_data + top_offset_ * g) );

			// add bias
			alpha = float(1);
			beta  = float(1);
			CUDNN_CHECK( cudnnAddTensor(handle_[g], CUDNN_ADD_SAME_C,
					&alpha,
					biasTensorDesc,
					bias_data + bias_offset_ * g,
					&beta,
					topTensorDesc,
					top_data + top_offset_ * g) );

			// free buffer
			if (sizeInBytes!=0)
			{
				CUDA_CHECK( cudaFree(workSpace) );
			}
		}
	    // Synchronize the work across groups, each of which went into its own
	    // stream, by launching an empty kernel into the default (null) stream.
	    // NOLINT_NEXT_LINE(whitespace/operators)
	    sync_conv_groups<<<1, 1>>>();
	}

	void Backward(const Blob_t *top, Blob_t *bottom)
	{

		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		float *weight = filtersBlob->data_gpu;
		float *weight_diff = filtersBlob->diff_gpu;
		gpu_set(filtersBlob->count(), float(0), weight_diff);
		float *bias_diff = biasBlob->diff_gpu;
		gpu_set(biasBlob->count(), float(0), bias_diff);
		const float* top_diff = top->diff_gpu;
		const float* bottom_data = bottom->data_gpu;
		float* bottom_diff = bottom->diff_gpu;
		for(int g = 0; g < convwithgroup_params->group; g++) {
			CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0 * convwithgroup_params->group + g],
					&alpha,
					topTensorDesc,
					top_diff + top_offset_ * g,
					&beta,
					biasTensorDesc,
					bias_diff + bias_offset_ * g));

			CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_[1 * convwithgroup_params->group + g],
					&alpha,
					bottomTensorDesc,
					bottom_data + bottom_offset_ * g,
					topTensorDesc,
					top_diff + top_offset_ * g,
					convDesc,
					&beta,
					filterDesc,
					weight_diff + weight_offset_ * g));

			CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[2 * convwithgroup_params->group + g],
					&alpha,
					filterDesc,
					weight + weight_offset_ * g,
					topTensorDesc,
					top_diff + top_offset_ * g,
					convDesc,
					&beta,
					bottomTensorDesc,
					bottom_diff + bottom_offset_ * g));
		}
	    // Synchronize the work across groups, each of which went into its own
	    // stream, by launching an empty kernel into the default (null) stream.
	    // NOLINT_NEXT_LINE(whitespace/operators)
	    sync_conv_groups<<<1, 1>>>();
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
	float filter_lr_mult;
	float filter_weight_decay_mult;
	float bias_lr_mult;
	float bias_weight_decay_mult;
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

__global__ void SoftmaxLossForwardGPU(const int nthreads,
		const float* prob_data, const float* label, float* loss,
		const int num, const int dim, const int spatial_dim,
		const bool has_ignore_label_, const int ignore_label_,
		float* counts) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[n * spatial_dim + s]);
		if (has_ignore_label_ && label_value == ignore_label_) {
			loss[index] = 0;
			counts[index] = 0;
		} else {
			loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
					float(FLT_MIN)));
			counts[index] = 1;
		}
	}
}

__global__ void SoftmaxLossBackwardGPU(const int nthreads, const float* top,
		const float* label, float* bottom_diff, const int num, const int dim,
		const int spatial_dim, const bool has_ignore_label_,
		const int ignore_label_, float* counts) {
	const int channels = dim / spatial_dim;

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[n * spatial_dim + s]);

		if (has_ignore_label_ && label_value == ignore_label_) {
			for (int c = 0; c < channels; ++c) {
				bottom_diff[n * dim + c * spatial_dim + s] = 0;
			}
			counts[index] = 0;
		} else {
			bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
			counts[index] = 1;
		}
	}
}

class SoftmaxWithLossParameter_t
{
public:
	cudnnSoftmaxAlgorithm_t cudnn_softmax_algo;
	cudnnSoftmaxMode_t cudnn_softmax_mode;
	bool has_ignore_label;
	int ignore_label;
	bool normalize;
};

class SoftmaxWithLossLayer_t : public Layer_t
{
public:
	cublasHandle_t cublashandle;
	SoftmaxWithLossParameter_t *cudnn_softmaxwithloss_params;
	Blob_t *prob_;
	/// Whether to ignore instances with a certain label.
	bool has_ignore_label_;
	/// The label indicating that an instance should be ignored.
	int ignore_label_;
	/// Whether to normalize the loss by the total number of values present
	/// (otherwise just by the batch size).
	bool normalize_;

	SoftmaxWithLossLayer_t(const SoftmaxWithLossParameter_t *cudnn_softmaxwithloss_params_) {
		cudnn_softmaxwithloss_params = const_cast<SoftmaxWithLossParameter_t *>(cudnn_softmaxwithloss_params_);
		cublashandle = NULL;
		CUBLAS_CHECK( cublasCreate(&cublashandle) );
		prob_ = NULL;
		has_ignore_label_ = false;
		ignore_label_ = -1;
		normalize_ = false;
	}

	~SoftmaxWithLossLayer_t() {
		CUBLAS_CHECK( cublasDestroy(cublashandle) );
	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(bottomTensorDesc,
				tensorFormat,
				dataType,
				bottom->N,
				bottom->C,
				bottom->H,
				bottom->W) );

		prob_ = top;
		prob_->N = bottom->N;
		prob_->C = bottom->C;
		prob_->H = bottom->H;
		prob_->W = bottom->W;
		CUDNN_CHECK( cudnnSetTensor4dDescriptor(topTensorDesc,
				tensorFormat,
				dataType,
				prob_->N,
				prob_->C,
				prob_->H,
				prob_->W) );

		prob_->allocate_gpu_data();
		prob_->allocate_gpu_diff();

		if(cudnn_softmaxwithloss_params->has_ignore_label)
			has_ignore_label_ = cudnn_softmaxwithloss_params->has_ignore_label;
		if(has_ignore_label_)
			ignore_label_ = cudnn_softmaxwithloss_params->ignore_label;
		normalize_ = cudnn_softmaxwithloss_params->normalize;

	}

	void Forward(const Blob_t *bottom, const Blob_t *label, Blob_t *top, float *loss) {
		float alpha = (float)1.0f;
		float beta = (float)0.0f;
		CUDNN_CHECK( cudnnSoftmaxForward(cudnnHandle,
				cudnn_softmaxwithloss_params->cudnn_softmax_algo ,
				cudnn_softmaxwithloss_params->cudnn_softmax_mode,
				&alpha,
				bottomTensorDesc,
				bottom->data_gpu,
				&beta,
				topTensorDesc,
				top->data_gpu) );

		prob_ = top;

		const float* prob_data = prob_->data_gpu;
		const float* label_data = label->data_gpu;
		const int num = prob_->N;
		const int dim = prob_->count() / num;
		const int spatial_dim = prob_->H * prob_->W;
		const int nthreads = num * spatial_dim;
		// Since this memory is not used for anything until it is overwritten
		// on the backward pass, we use it here to avoid having to allocate new GPU
		// memory to accumulate intermediate results in the kernel.
		float* loss_data = bottom->diff_gpu;
		// Similarly, this memory is never used elsewhere, and thus we can use it
		// to avoid having to allocate additional GPU memory.
		float* counts = prob_->diff_gpu;
		// NOLINT_NEXT_LINE(whitespace/operators)
		SoftmaxLossForwardGPU<<<GPU_GET_BLOCKS(nthreads),
				GPU_CUDA_NUM_THREADS>>>(nthreads, prob_data, label_data, loss_data,
						num, dim, spatial_dim, has_ignore_label_, ignore_label_, counts);
		gpu_asum(cublashandle, nthreads, loss_data, loss);
		if (normalize_) {
			float count;
			gpu_asum(cublashandle, nthreads, counts, &count);
			*loss /= count;
		} else {
			*loss /= num;
		}
	}

	void Backward(const Blob_t *top, const Blob_t *label, Blob_t *bottom) {
		//		float alpha = (float)1.0f;
		//		float beta = (float)0.0f;
		//		CUDNN_CHECK( cudnnSoftmaxBackward( cudnnHandle,
		//				cudnn_softmax_params->cudnn_softmax_algo ,
		//				cudnn_softmax_params->cudnn_softmax_mode,
		//				&alpha,
		//				topTensorDesc,
		//				top->data_gpu,
		//				topTensorDesc,
		//				top->diff_gpu,
		//				&beta,
		//				bottomTensorDesc,
		//				bottom->diff_gpu) );

		float* bottom_diff = bottom->diff_gpu;
		const float* prob_data = prob_->data_gpu;
		const float* top_data = top->data_gpu;
		gpu_copy(prob_->count(), prob_data, bottom_diff);
		const float* label_data = label->data_gpu;
		const int num = prob_->N;
		const int dim = prob_->count() / num;
		const int spatial_dim = prob_->H * prob_->W;
		const int nthreads = num * spatial_dim;
		// Since this memory is never used for anything else,
		// we use to to avoid allocating new GPU memory.
		float* counts = prob_->diff_gpu;
		// NOLINT_NEXT_LINE(whitespace/operators)
		SoftmaxLossBackwardGPU<<<GPU_GET_BLOCKS(nthreads),
				GPU_CUDA_NUM_THREADS>>>(nthreads, top_data, label_data, bottom_diff,
						num, dim, spatial_dim, has_ignore_label_, ignore_label_, counts);
		const float loss_weight = float(1.0f);
		if (normalize_) {
			float count;
			gpu_asum(cublashandle, nthreads, counts, &count);
			gpu_scal(cublashandle, prob_->count(), loss_weight / count, bottom_diff);
		} else {
			gpu_scal(cublashandle, prob_->count(), loss_weight / num, bottom_diff);
		}
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

class ArgMaxParameter_t
{
public:
	bool out_max_val;
	int top_k;
};

class ArgMaxLayer_t
{
public:
	ArgMaxParameter_t *argmax_params;
	ArgMaxLayer_t(const ArgMaxParameter_t *argmax_params_) {
		argmax_params = const_cast<ArgMaxParameter_t *>(argmax_params_);
	}

	~ArgMaxLayer_t() {

	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		top->N = bottom->N;
		top->C = 2;
		top->H = argmax_params->top_k;
		top->W = 1;

		top->allocate_cpu_data();
	}

	void Forward_cpu(Blob_t *bottom, Blob_t *top) {

		bottom->data_to_cpu();

		const float* bottom_data = bottom->data_cpu;
		float* top_data = top->data_cpu;
		int num = bottom->N;
		int dim = bottom->count() / bottom->N;
		for (int i = 0; i < num; ++i) {
			std::vector<std::pair<float, int> > bottom_data_vector;
			for (int j = 0; j < dim; ++j) {
				bottom_data_vector.push_back(
						std::make_pair(bottom_data[i * dim + j], j));
			}
			std::partial_sort(
					bottom_data_vector.begin(), bottom_data_vector.begin() + argmax_params->top_k,
					bottom_data_vector.end(), std::greater<std::pair<float, int> >());
			for (int j = 0; j < argmax_params->top_k; ++j) {
				top_data[top->offset(i, 0, j)] = bottom_data_vector[j].second;
			}
			if (argmax_params->out_max_val) {
				for (int j = 0; j < argmax_params->top_k; ++j) {
					top_data[top->offset(i, 1, j)] = bottom_data_vector[j].first;
				}
			}
		}
	}
};

class AccuracyParameter_t
{
public:
	int top_k;
};

class AccuracyLayer_t
{
public:
	AccuracyParameter_t *accuracy_params;
	AccuracyLayer_t(const AccuracyParameter_t *accuracy_params_) {
		accuracy_params = const_cast<AccuracyParameter_t *>(accuracy_params_);
	}

	~AccuracyLayer_t() {

	}

	void Setup(const Blob_t *bottom, Blob_t *top) {
		top->N = 1;
		top->C = 1;
		top->H = 1;
		top->W = 1;

		top->allocate_cpu_data();
	}

	void Forward_cpu(Blob_t *bottom, Blob_t *label, Blob_t *top) {

		bottom->data_to_cpu();
		label->data_to_cpu();

		float accuracy = 0;
		const float* bottom_data = bottom->data_cpu;
		const float* bottom_label = label->data_cpu;
		int num = bottom->N;
		int dim = bottom->count() / bottom->N;
		vector<float> maxval(accuracy_params->top_k+1);
		vector<int> max_id(accuracy_params->top_k+1);
		for (int i = 0; i < num; ++i) {
			// Top-k accuracy
			std::vector<std::pair<float, int> > bottom_data_vector;
			for (int j = 0; j < dim; ++j) {
				bottom_data_vector.push_back(
						std::make_pair(bottom_data[i * dim + j], j));
			}
			std::partial_sort(
					bottom_data_vector.begin(), bottom_data_vector.begin() + accuracy_params->top_k,
					bottom_data_vector.end(), std::greater<std::pair<float, int> >());
			// check if true label is in top k predictions
			for (int k = 0; k < accuracy_params->top_k; k++) {
				if (bottom_data_vector[k].second == static_cast<int>(bottom_label[i])) {
					++accuracy;
					break;
				}
			}
		}

		// LOG(INFO) << "Accuracy: " << accuracy;
		top->data_cpu[0] = accuracy / num;
		// Accuracy layer should not be used as a loss function.
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

	ConvolutionParameter_t *conv2_params;
	ConvolutionLayer_t *conv2;
	Blob_t *conv2_top;

	ActivationParameter_t *relu2_params;
	ActivationLayer_t *relu2;
	Blob_t *relu2_top;

	PoolingParameter_t *mp2_params;
	PoolingLayer_t *mp2;
	Blob_t *mp2_top;

	ConvolutionParameter_t *conv3_params;
	ConvolutionLayer_t *conv3;
	Blob_t *conv3_top;

	ActivationParameter_t *relu3_params;
	ActivationLayer_t *relu3;
	Blob_t *relu3_top;

	PoolingParameter_t *mp3_params;
	PoolingLayer_t *mp3;
	Blob_t *mp3_top;

	FullyConnectedParameter_t *ip1_params;
	FullyConnectedLayer_t *ip1;
	Blob_t *ip1_top;

	// the following softmax layer and multinomial logistic loss layer have been replaced by the softmaxwithloss layer.
	//	SoftmaxParameter_t *sm1_params;
	//	SoftmaxLayer_t *sm1;
	//	Blob_t *sm1_top;
	//
	//	MultinomialLogisticLossParameter_t *mlr1_params;
	//	MultinomialLogisticLossLayer_t *mlr1;
	//	Blob_t *mlr1_top;

	SoftmaxWithLossParameter_t *sml1_params;
	SoftmaxWithLossLayer_t *sml1;
	Blob_t *sml1_top;

	//	ArgMaxParameter_t *argmax1_params;
	//	ArgMaxLayer_t *argmax1;
	//	Blob_t *argmax1_top;

	AccuracyParameter_t *accuracy1_params;
	AccuracyLayer_t *accuracy1;
	Blob_t *accuracy1_top;


	Blob_t *conv1_filtersBlob_old;
	Blob_t *conv1_biasBlob_old;
	Blob_t *conv2_filtersBlob_old;
	Blob_t *conv2_biasBlob_old;
	Blob_t *conv3_filtersBlob_old;
	Blob_t *conv3_biasBlob_old;
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
		conv2 = NULL;
		conv2_top = NULL;
		conv2_params = NULL;
		relu2 = NULL;
		relu2_top = NULL;
		relu2_params = NULL;
		mp2 = NULL;
		mp2_top = NULL;
		mp2_params = NULL;
		conv3 = NULL;
		conv3_top = NULL;
		conv3_params = NULL;
		relu3 = NULL;
		relu3_top = NULL;
		relu3_params = NULL;
		mp3 = NULL;
		mp3_top = NULL;
		mp3_params = NULL;
		ip1 = NULL;
		ip1_top = NULL;
		ip1_params = NULL;
		//		sm1 = NULL;
		//		sm1_top = NULL;
		//		sm1_params = NULL;
		//		mlr1 = NULL;
		//		mlr1_top = NULL;
		//		mlr1_params = NULL;
		sml1 = NULL;
		sml1_top = NULL;
		sml1_params = NULL;

		//		argmax1 = NULL;
		//		argmax1_top = NULL;
		//		argmax1_params = NULL;

		accuracy1 = NULL;
		accuracy1_top = NULL;
		accuracy1_params = NULL;

		conv1_filtersBlob_old = NULL;
		conv1_biasBlob_old = NULL;
		conv2_filtersBlob_old = NULL;
		conv2_biasBlob_old = NULL;
		conv3_filtersBlob_old = NULL;
		conv3_biasBlob_old = NULL;
		ip1_filtersBlob_old = NULL;
		ip1_biasBlob_old = NULL;

	}

	~Network_t() {
		DestroyNet();
	}

	void DestroyNet() {

		cudaSetDevice(gpu_id);

		delete batch_samples; batch_samples = NULL;
		delete batch_labels; batch_labels = NULL;

		delete conv1; conv1 = NULL;
		delete relu1; relu1 = NULL;
		delete mp1; mp1 = NULL;
		delete conv2; conv2 = NULL;
		delete relu2; relu2 = NULL;
		delete mp2; mp2 = NULL;
		delete conv3; conv3 = NULL;
		delete relu3; relu3 = NULL;
		delete mp3; mp3 = NULL;
		delete ip1; ip1 = NULL;
		//		delete sm1; sm1 = NULL;
		//		delete argmax1; argmax1 = NULL;
		//		delete mlr1; mlr1 = NULL;
		delete sml1; sml1 = NULL;
		delete accuracy1; accuracy1 = NULL;

		delete conv1_top; conv1_top = NULL;
		delete relu1_top; relu1_top = NULL;
		delete mp1_top; mp1_top = NULL;
		delete conv2_top; conv2_top = NULL;
		delete relu2_top; relu2_top = NULL;
		delete mp2_top; mp2_top = NULL;
		delete conv3_top; conv3_top = NULL;
		delete relu3_top; relu3_top = NULL;
		delete mp3_top; mp3_top = NULL;
		delete ip1_top; ip1_top = NULL;
		//		delete sm1_top; sm1_top = NULL;
		//		delete argmax1_top; argmax1_top = NULL;
		//		delete mlr1_top; mlr1_top = NULL;
		delete sml1_top; sml1_top = NULL;
		delete accuracy1_top; accuracy1_top = NULL;

		delete conv1_params; conv1_params = NULL;
		delete relu1_params; relu1_params = NULL;
		delete mp1_params; mp1_params = NULL;
		delete conv2_params; conv2_params = NULL;
		delete relu2_params; relu2_params = NULL;
		delete mp2_params; mp2_params = NULL;
		delete conv3_params; conv3_params = NULL;
		delete relu3_params; relu3_params = NULL;
		delete mp3_params; mp3_params = NULL;
		delete ip1_params; ip1_params = NULL;
		//		delete sm1_params; sm1_params = NULL;
		//		delete argmax1_params; argmax1_params = NULL;
		//		delete mlr1_params; mlr1_params = NULL;
		delete sml1_params; sml1_params = NULL;
		delete accuracy1_params; accuracy1_params = NULL;

		delete conv1_filtersBlob_old; conv1_filtersBlob_old = NULL;
		delete conv1_biasBlob_old; conv1_biasBlob_old = NULL;
		delete conv2_filtersBlob_old; conv2_filtersBlob_old = NULL;
		delete conv2_biasBlob_old; conv2_biasBlob_old = NULL;
		delete conv3_filtersBlob_old; conv3_filtersBlob_old = NULL;
		delete conv3_biasBlob_old; conv3_biasBlob_old = NULL;
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
		batch_labels->allocate_cpu_data();

		LOG(INFO) << "conv1 setup.\n";
		conv1_top = new Blob_t();
		conv1_params = new ConvolutionParameter_t();
		conv1_params->filter_N = 3;
		conv1_params->filter_C = 32;
		conv1_params->filter_H = 5;
		conv1_params->filter_W = 5;
		conv1_params->pad_h = 2;
		conv1_params->pad_w = 2;
		conv1_params->stride_h = 1;
		conv1_params->stride_w = 1;
		conv1_params->upscale_h = 1;
		conv1_params->upscale_w = 1;
		conv1_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv1 = new ConvolutionLayer_t(conv1_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->filtersBlob->data_gpu, conv1->filtersBlob->count(), (float)0.0f, (float)0.0001f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->biasBlob->data_gpu, conv1->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv1->biasBlob->count(), 0, conv1->biasBlob->data_gpu);
		conv1->Setup(batch_samples, conv1_top);


		LOG(INFO) << "relu1 setup.\n";
		relu1_top = new Blob_t();
		relu1_params = new ActivationParameter_t();
		relu1_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu1 = new ActivationLayer_t(relu1_params);
		relu1->Setup(conv1_top, relu1_top);

		LOG(INFO) << "mp1 setup.\n";
		mp1_top = new Blob_t();
		mp1_params = new PoolingParameter_t();
		mp1_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
		mp1_params->poolsize_h = 3;
		mp1_params->poolsize_w = 3;
		mp1_params->pad_h = 0;
		mp1_params->pad_w = 0;
		mp1_params->stride_h = 2;
		mp1_params->stride_w = 2;
		mp1 = new PoolingLayer_t(mp1_params);
		mp1->Setup(relu1_top, mp1_top);

		LOG(INFO) << "conv2 setup.\n";
		conv2_top = new Blob_t();
		conv2_params = new ConvolutionParameter_t();
		conv2_params->filter_N = 32;
		conv2_params->filter_C = 32;
		conv2_params->filter_H = 5;
		conv2_params->filter_W = 5;
		conv2_params->pad_h = 2;
		conv2_params->pad_w = 2;
		conv2_params->stride_h = 1;
		conv2_params->stride_w = 1;
		conv2_params->upscale_h = 1;
		conv2_params->upscale_w = 1;
		conv2_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv2 = new ConvolutionLayer_t(conv2_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->filtersBlob->data_gpu, conv2->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv2->biasBlob->count(), 0, conv2->biasBlob->data_gpu);
		conv2->Setup(mp1_top, conv2_top);


		LOG(INFO) << "relu2 setup.\n";
		relu2_top = new Blob_t();
		relu2_params = new ActivationParameter_t();
		relu2_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu2 = new ActivationLayer_t(relu2_params);
		relu2->Setup(conv2_top, relu2_top);

		LOG(INFO) << "mp2 setup.\n";
		mp2_top = new Blob_t();
		mp2_params = new PoolingParameter_t();
		mp2_params->cudnn_pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
		mp2_params->poolsize_h = 3;
		mp2_params->poolsize_w = 3;
		mp2_params->pad_h = 0;
		mp2_params->pad_w = 0;
		mp2_params->stride_h = 2;
		mp2_params->stride_w = 2;
		mp2 = new PoolingLayer_t(mp2_params);
		mp2->Setup(relu2_top, mp2_top);

		LOG(INFO) << "conv3 setup.\n";
		conv3_top = new Blob_t();
		conv3_params = new ConvolutionParameter_t();
		conv3_params->filter_N = 32;
		conv3_params->filter_C = 64;
		conv3_params->filter_H = 5;
		conv3_params->filter_W = 5;
		conv3_params->pad_h = 2;
		conv3_params->pad_w = 2;
		conv3_params->stride_h = 1;
		conv3_params->stride_w = 1;
		conv3_params->upscale_h = 1;
		conv3_params->upscale_w = 1;
		conv3_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv3 = new ConvolutionLayer_t(conv3_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->filtersBlob->data_gpu, conv3->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->biasBlob->data_gpu, conv3->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv3->biasBlob->count(), 0, conv3->biasBlob->data_gpu);
		conv3->Setup(mp2_top, conv3_top);


		LOG(INFO) << "relu3 setup.\n";
		relu3_top = new Blob_t();
		relu3_params = new ActivationParameter_t();
		relu3_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu3 = new ActivationLayer_t(relu3_params);
		relu3->Setup(conv3_top, relu3_top);

		LOG(INFO) << "mp3 setup.\n";
		mp3_top = new Blob_t();
		mp3_params = new PoolingParameter_t();
		mp3_params->cudnn_pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
		mp3_params->poolsize_h = 3;
		mp3_params->poolsize_w = 3;
		mp3_params->pad_h = 0;
		mp3_params->pad_w = 0;
		mp3_params->stride_h = 2;
		mp3_params->stride_w = 2;
		mp3 = new PoolingLayer_t(mp3_params);
		mp3->Setup(relu3_top, mp3_top);

		LOG(INFO) << "ip1 setup.\n";
		ip1_top = new Blob_t();
		ip1_params = new FullyConnectedParameter_t();
		ip1_params->hidden_size = 10;
		ip1 = new FullyConnectedLayer_t(ip1_params);
		ip1->Setup(mp3_top, ip1_top);
		CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->filtersBlob->data_gpu, ip1->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(ip1->biasBlob->count(), 0, ip1->biasBlob->data_gpu);

		//		printf("sm1 setup.\n");
		//		sm1_top = new Blob_t();
		//		sm1_params = new SoftmaxParameter_t();
		//		sm1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
		//		sm1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
		//		sm1 = new SoftmaxLayer_t(sm1_params);
		//		sm1->Setup(ip1_top, sm1_top);
		//
		//		printf("mlr1 setup (in cpu).\n");
		//		mlr1_top = new Blob_t();
		//		mlr1_params = new MultinomialLogisticLossParameter_t();
		//		mlr1 = new MultinomialLogisticLossLayer_t(mlr1_params);
		//		mlr1->Setup(sm1_top, mlr1_top);

		LOG(INFO) << "sml1 setup.\n";
		sml1_top = new Blob_t();
		sml1_params = new SoftmaxWithLossParameter_t();
		sml1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
		sml1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
		sml1_params->has_ignore_label = false;
		sml1_params->ignore_label = -1;
		sml1_params->normalize = false;
		sml1 = new SoftmaxWithLossLayer_t(sml1_params);
		sml1->Setup(ip1_top, sml1_top);

		//		printf("argmax1 setup.\n");
		//		argmax1_top = new Blob_t();
		//		argmax1_params = new ArgMaxParameter_t();
		//		argmax1_params->out_max_val = true;
		//		argmax1_params->top_k = 1;
		//		argmax1 = new ArgMaxLayer_t(argmax1_params);
		//		argmax1->Setup(sml1_top, argmax1_top);

		LOG(INFO) << "accuracy1 setup.\n";
		accuracy1_top = new Blob_t();
		accuracy1_params = new AccuracyParameter_t();
		accuracy1_params->top_k = 1;
		accuracy1 = new AccuracyLayer_t(accuracy1_params);
		accuracy1->Setup(ip1_top, accuracy1_top);

		LOG(INFO) << "initialize old net params.\n";
		conv1_filtersBlob_old = new Blob_t(conv1->filtersBlob->N, conv1->filtersBlob->C, conv1->filtersBlob->H, conv1->filtersBlob->W);
		conv1_biasBlob_old = new Blob_t(conv1->biasBlob->N, conv1->biasBlob->C, conv1->biasBlob->H, conv1->biasBlob->W);
		conv1_filtersBlob_old->allocate_gpu_data();
		conv1_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv1_filtersBlob_old->data_gpu, conv1_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv1_biasBlob_old->data_gpu, conv1_biasBlob_old->count(), 0.0f, 0.0f);

		conv2_filtersBlob_old = new Blob_t(conv2->filtersBlob->N, conv2->filtersBlob->C, conv2->filtersBlob->H, conv2->filtersBlob->W);
		conv2_biasBlob_old = new Blob_t(conv2->biasBlob->N, conv2->biasBlob->C, conv2->biasBlob->H, conv2->biasBlob->W);
		conv2_filtersBlob_old->allocate_gpu_data();
		conv2_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv2_filtersBlob_old->data_gpu, conv2_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv2_biasBlob_old->data_gpu, conv2_biasBlob_old->count(), 0.0f, 0.0f);

		conv3_filtersBlob_old = new Blob_t(conv3->filtersBlob->N, conv3->filtersBlob->C, conv3->filtersBlob->H, conv3->filtersBlob->W);
		conv3_biasBlob_old = new Blob_t(conv3->biasBlob->N, conv3->biasBlob->C, conv3->biasBlob->H, conv3->biasBlob->W);
		conv3_filtersBlob_old->allocate_gpu_data();
		conv3_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv3_filtersBlob_old->data_gpu, conv3_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv3_biasBlob_old->data_gpu, conv3_biasBlob_old->count(), 0.0f, 0.0f);

		ip1_filtersBlob_old = new Blob_t(ip1->filtersBlob->N, ip1->filtersBlob->C, ip1->filtersBlob->H, ip1->filtersBlob->W);
		ip1_biasBlob_old = new Blob_t(ip1->biasBlob->N, ip1->biasBlob->C, ip1->biasBlob->H, ip1->biasBlob->W);
		ip1_filtersBlob_old->allocate_gpu_data();
		ip1_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, ip1_filtersBlob_old->data_gpu, ip1_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, ip1_biasBlob_old->data_gpu, ip1_biasBlob_old->count(), 0.0f, 0.0f);

		LOG(INFO) << "build net (done).\n";
	}

	void Forward(float *loss, float *accuracy) {
		cudaSetDevice(gpu_id);

		// printf("conv1 forward.\n");
		conv1->Forward(batch_samples, conv1_top);

		// printf("relu1 forward.\n");
		relu1->Forward(conv1_top, relu1_top);

		// printf("mp1 forward.\n");
		mp1->Forward(relu1_top, mp1_top);

		// printf("conv2 forward.\n");
		conv2->Forward(mp1_top, conv2_top);

		// printf("relu2 forward.\n");
		relu2->Forward(conv2_top, relu2_top);

		// printf("mp2 forward.\n");
		mp2->Forward(relu2_top, mp2_top);

		// printf("conv3 forward.\n");
		conv3->Forward(mp2_top, conv3_top);

		// printf("relu3 forward.\n");
		relu3->Forward(conv3_top, relu3_top);

		// printf("mp2 forward.\n");
		mp3->Forward(relu3_top, mp3_top);

		// printf("ip1 forward.\n");
		ip1->Forward(mp3_top, ip1_top);

		// printf("sm1 forward.\n");
		// sm1->Forward(ip1_top, sm1_top);

		// printf("mlr1 forward.\n");
		// mlr1->Forward(sm1_top, batch_labels, mlr1_top);

		// loss = mlr1_top->data_cpu[0];

		sml1->Forward(ip1_top, batch_labels, sml1_top, loss);

		//		argmax1->Forward_cpu(sml1_top, argmax1_top);

		accuracy1->Forward_cpu(ip1_top, batch_labels, accuracy1_top);

		*accuracy = accuracy1_top->data_cpu[0];

	}

	void Backward() {
		cudaSetDevice(gpu_id);

		// printf("sml1 backward.\n");
		sml1->Backward(sml1_top, batch_labels, ip1_top);

		// printf("mlr1 backward.\n");
		// mlr1->Backward(mlr1_top, batch_labels, sm1_top);

		// printf("sm1 backward.\n");
		// sm1->Backward(sm1_top, ip1_top);

		// printf("ip1 backward.\n");
		ip1->Backward(ip1_top, mp3_top);

		// printf("mp3 backward.\n");
		mp3->Backward(mp3_top, relu3_top);

		// printf("relu3 backward.\n");
		relu3->Backward(relu3_top, conv3_top);

		// printf("conv3 backward.\n");
		conv3->Backward(conv3_top, mp2_top);

		// printf("mp2 backward.\n");
		mp2->Backward(mp2_top, relu2_top);

		// printf("relu2 backward.\n");
		relu2->Backward(relu2_top, conv2_top);

		// printf("conv2 backward.\n");
		conv2->Backward(conv2_top, mp1_top);

		// printf("mp1 backward.\n");
		mp1->Backward(mp1_top, relu1_top);

		// printf("relu1 backward.\n");
		relu1->Backward(relu1_top, conv1_top);

		// printf("conv1 backward.\n");
		conv1->Backward(conv1_top, batch_samples);
	}

	void ForwardBackward(float *loss, float *accuracy) {
		Forward(loss, accuracy);
		Backward();
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
		ComputeUpdateValueSingle(conv3->filtersBlob, conv3_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv3->biasBlob, 	 conv3_biasBlob_old, 	lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv2->filtersBlob, conv2_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv2->biasBlob, 	 conv2_biasBlob_old, 	lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv1->filtersBlob, conv1_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv1->biasBlob, 	 conv1_biasBlob_old, 	lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(ip1->filtersBlob,   ip1_filtersBlob_old,   lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(ip1->biasBlob,      ip1_biasBlob_old, 		lr_rate, momentum, weight_decay);
	}

	void UpdateNet(float scale = -1.0f) {
		cudaSetDevice(gpu_id);
		gpu_axpy(cublas_handle, conv3->filtersBlob->count(), float(scale), conv3->filtersBlob->diff_gpu, conv3->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv3->biasBlob->count(), 	 float(scale), conv3->biasBlob->diff_gpu, 	  conv3->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, conv2->filtersBlob->count(), float(scale), conv2->filtersBlob->diff_gpu, conv2->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv2->biasBlob->count(), 	 float(scale), conv2->biasBlob->diff_gpu, 	  conv2->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, conv1->filtersBlob->count(), float(scale), conv1->filtersBlob->diff_gpu, conv1->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv1->biasBlob->count(), 	 float(scale), conv1->biasBlob->diff_gpu,    conv1->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, ip1->filtersBlob->count(),   float(scale), ip1->filtersBlob->diff_gpu,   ip1->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, ip1->biasBlob->count(), 	 float(scale), ip1->biasBlob->diff_gpu,      ip1->biasBlob->data_gpu);
	}

	void SaveNetParams(int epoch) {
		cudaSetDevice(gpu_id);
		stringstream f1; f1 << net_name << "_c1_weight_e" << epoch << ".mat";
		conv1->filtersBlob->save_cpu_data_and_diff_to_mat(f1.str().c_str());

		stringstream f2; f2 << net_name << "_c1_bias_e" << epoch << ".mat";
		conv1->biasBlob->save_cpu_data_and_diff_to_mat(f2.str().c_str());

		stringstream f3; f3 << net_name << "_c2_weight_e" << epoch << ".mat";
		conv2->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
		stringstream f4; f4 << net_name << "_c2_bias_e" << epoch << ".mat";
		conv2->biasBlob->save_cpu_data_and_diff_to_mat(f4.str().c_str());

		stringstream f5; f5 << net_name << "_c3_weight_e" << epoch << ".mat";
		conv3->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
		stringstream f6; f6 << net_name << "_c3_bias_e" << epoch << ".mat";
		conv3->biasBlob->save_cpu_data_and_diff_to_mat(f6.str().c_str());

		stringstream f7; f7 << net_name << "_ip1_weight_e" << epoch << ".mat";
		ip1->filtersBlob->save_cpu_data_and_diff_to_mat(f7.str().c_str());
		stringstream f8; f8 << net_name << "_ip1_bias_e" << epoch << ".mat";
		ip1->biasBlob->save_cpu_data_and_diff_to_mat(f8.str().c_str());

	}

	void CopyNetParamsFrom(const Network_t *other) {
		CopyBlobData_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv3->biasBlob, 	other->gpu_id, conv3->biasBlob,	   gpu_id);
		CopyBlobData_gpu(other->conv2->filtersBlob, other->gpu_id, conv2->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv2->biasBlob, 	other->gpu_id, conv2->biasBlob,    gpu_id);
		CopyBlobData_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv1->biasBlob, 	other->gpu_id, conv1->biasBlob,    gpu_id);
		CopyBlobData_gpu(other->ip1->filtersBlob, 	other->gpu_id, ip1->filtersBlob,   gpu_id);
		CopyBlobData_gpu(other->ip1->biasBlob, 		other->gpu_id, ip1->biasBlob, 	   gpu_id);
	}

	void AddNetParamsDiffFrom(const Network_t *other) {
		AddBlobDiff_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv3->biasBlob,    other->gpu_id, conv3->biasBlob,    gpu_id);
		AddBlobDiff_gpu(other->conv2->filtersBlob, other->gpu_id, conv2->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv2->biasBlob,    other->gpu_id, conv2->biasBlob, 	  gpu_id);
		AddBlobDiff_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv1->biasBlob,    other->gpu_id, conv1->biasBlob, 	  gpu_id);
		AddBlobDiff_gpu(other->ip1->filtersBlob,   other->gpu_id, ip1->filtersBlob,   gpu_id);
		AddBlobDiff_gpu(other->ip1->biasBlob,      other->gpu_id, ip1->biasBlob, 	  gpu_id);
	}

	void ClearNetParamsDiff() {
		cudaSetDevice(gpu_id);
		gpu_set(conv3->filtersBlob->count(), 0, conv3->filtersBlob->diff_gpu);
		gpu_set(conv3->biasBlob->count(), 	 0, conv3->biasBlob->diff_gpu);
		gpu_set(conv2->filtersBlob->count(), 0, conv2->filtersBlob->diff_gpu);
		gpu_set(conv2->biasBlob->count(),    0, conv2->biasBlob->diff_gpu);
		gpu_set(conv1->filtersBlob->count(), 0, conv1->filtersBlob->diff_gpu);
		gpu_set(conv1->biasBlob->count(),    0, conv1->biasBlob->diff_gpu);
		gpu_set(ip1->filtersBlob->count(),   0, ip1->filtersBlob->diff_gpu);
		gpu_set(ip1->biasBlob->count(),      0, ip1->biasBlob->diff_gpu);
	}

};

class AlexNetwork_t
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

	PoolingParameter_t *pool1_params;
	PoolingLayer_t *pool1;
	Blob_t *pool1_top;

	ConvolutionWithGroupParameter_t *conv2g_params;
	ConvolutionWithGroupLayer_t *conv2g;
	Blob_t *conv2g_top;

	ActivationParameter_t *relu2_params;
	ActivationLayer_t *relu2;
	Blob_t *relu2_top;

	PoolingParameter_t *pool2_params;
	PoolingLayer_t *pool2;
	Blob_t *pool2_top;

	ConvolutionParameter_t *conv3_params;
	ConvolutionLayer_t *conv3;
	Blob_t *conv3_top;

	ActivationParameter_t *relu3_params;
	ActivationLayer_t *relu3;
	Blob_t *relu3_top;

//	PoolingParameter_t *mp3_params;
//	PoolingLayer_t *mp3;
//	Blob_t *mp3_top;

	ConvolutionWithGroupParameter_t *conv4g_params;
	ConvolutionWithGroupLayer_t *conv4g;
	Blob_t *conv4g_top;

	ActivationParameter_t *relu4_params;
	ActivationLayer_t *relu4;
	Blob_t *relu4_top;

	ConvolutionWithGroupParameter_t *conv5g_params;
	ConvolutionWithGroupLayer_t *conv5g;
	Blob_t *conv5g_top;

	ActivationParameter_t *relu5_params;
	ActivationLayer_t *relu5;
	Blob_t *relu5_top;

	PoolingParameter_t *pool5_params;
	PoolingLayer_t *pool5;
	Blob_t *pool5_top;

	FullyConnectedParameter_t *fc6_params;
	FullyConnectedLayer_t *fc6;
	Blob_t *fc6_top;

	ActivationParameter_t *relu6_params;
	ActivationLayer_t *relu6;
	Blob_t *relu6_top;

	FullyConnectedParameter_t *fc7_params;
	FullyConnectedLayer_t *fc7;
	Blob_t *fc7_top;

	ActivationParameter_t *relu7_params;
	ActivationLayer_t *relu7;
	Blob_t *relu7_top;

	FullyConnectedParameter_t *fc8_params;
	FullyConnectedLayer_t *fc8;
	Blob_t *fc8_top;


	// the following softmax layer and multinomial logistic loss layer have been replaced by the softmaxwithloss layer.
	//	SoftmaxParameter_t *sm1_params;
	//	SoftmaxLayer_t *sm1;
	//	Blob_t *sm1_top;
	//
	//	MultinomialLogisticLossParameter_t *mlr1_params;
	//	MultinomialLogisticLossLayer_t *mlr1;
	//	Blob_t *mlr1_top;

	SoftmaxWithLossParameter_t *sml1_params;
	SoftmaxWithLossLayer_t *sml1;
	Blob_t *sml1_top;

	//	ArgMaxParameter_t *argmax1_params;
	//	ArgMaxLayer_t *argmax1;
	//	Blob_t *argmax1_top;

	AccuracyParameter_t *accuracy1_params;
	AccuracyLayer_t *accuracy1;
	Blob_t *accuracy1_top;


	Blob_t *conv1_filtersBlob_old;
	Blob_t *conv1_biasBlob_old;
	Blob_t *conv2g_filtersBlob_old;
	Blob_t *conv2g_biasBlob_old;
	Blob_t *conv3_filtersBlob_old;
	Blob_t *conv3_biasBlob_old;
	Blob_t *conv4g_filtersBlob_old;
	Blob_t *conv4g_biasBlob_old;
	Blob_t *conv5g_filtersBlob_old;
	Blob_t *conv5g_biasBlob_old;
	Blob_t *fc6_filtersBlob_old;
	Blob_t *fc6_biasBlob_old;
	Blob_t *fc7_filtersBlob_old;
	Blob_t *fc7_biasBlob_old;
	Blob_t *fc8_filtersBlob_old;
	Blob_t *fc8_biasBlob_old;


	AlexNetwork_t(string net_name_, int gpu_id_ = 0) {
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
		pool1 = NULL;
		pool1_top = NULL;
		pool1_params = NULL;
		conv2g = NULL;
		conv2g_top = NULL;
		conv2g_params = NULL;
		relu2 = NULL;
		relu2_top = NULL;
		relu2_params = NULL;
		pool2 = NULL;
		pool2_top = NULL;
		pool2_params = NULL;
		conv3 = NULL;
		conv3_top = NULL;
		conv3_params = NULL;
		relu3 = NULL;
		relu3_top = NULL;
		relu3_params = NULL;
		conv4g = NULL;
		conv4g_top = NULL;
		conv4g_params = NULL;
		relu4 = NULL;
		relu4_top = NULL;
		relu4_params = NULL;
		conv5g = NULL;
		conv5g_top = NULL;
		conv5g_params = NULL;
		relu5 = NULL;
		relu5_top = NULL;
		relu5_params = NULL;
		pool5 = NULL;
		pool5_top = NULL;
		pool5_params = NULL;
		fc6 = NULL;
		fc6_top = NULL;
		fc6_params = NULL;
		relu6 = NULL;
		relu6_top = NULL;
		relu6_params = NULL;
		fc7 = NULL;
		fc7_top = NULL;
		fc7_params = NULL;
		relu7 = NULL;
		relu7_top = NULL;
		relu7_params = NULL;
		fc8 = NULL;
		fc8_top = NULL;
		fc8_params = NULL;
		//		sm1 = NULL;
		//		sm1_top = NULL;
		//		sm1_params = NULL;
		//		argmax1 = NULL;
		//		argmax1_top = NULL;
		//		argmax1_params = NULL;
		//		mlr1 = NULL;
		//		mlr1_top = NULL;
		//		mlr1_params = NULL;
		sml1 = NULL;
		sml1_top = NULL;
		sml1_params = NULL;

		accuracy1 = NULL;
		accuracy1_top = NULL;
		accuracy1_params = NULL;

		conv1_filtersBlob_old = NULL;
		conv1_biasBlob_old = NULL;
		conv2g_filtersBlob_old = NULL;
		conv2g_biasBlob_old = NULL;
		conv3_filtersBlob_old = NULL;
		conv3_biasBlob_old = NULL;
		conv4g_filtersBlob_old = NULL;
		conv4g_biasBlob_old = NULL;
		conv5g_filtersBlob_old = NULL;
		conv5g_biasBlob_old = NULL;
		fc6_filtersBlob_old = NULL;
		fc6_biasBlob_old = NULL;
		fc7_filtersBlob_old = NULL;
		fc7_biasBlob_old = NULL;
		fc8_filtersBlob_old = NULL;
		fc8_biasBlob_old = NULL;

	}

	~AlexNetwork_t() {
		DestroyNet();
	}

	void DestroyNet() {

		cudaSetDevice(gpu_id);

		delete batch_samples; batch_samples = NULL;
		delete batch_labels; batch_labels = NULL;

		delete conv1; conv1 = NULL;
		delete relu1; relu1 = NULL;
		delete pool1; pool1 = NULL;
		delete conv2g; conv2g = NULL;
		delete relu2; relu2 = NULL;
		delete pool2; pool2 = NULL;
		delete conv3; conv3 = NULL;
		delete relu3; relu3 = NULL;
		delete conv4g; conv4g = NULL;
		delete relu4; relu4 = NULL;
		delete conv5g; conv5g = NULL;
		delete relu5; relu5 = NULL;
		delete pool5; pool5 = NULL;
		delete fc6; fc6 = NULL;
		delete relu6; relu6 = NULL;
		delete fc7; fc7 = NULL;
		delete relu7; relu7 = NULL;
		delete fc8; fc8 = NULL;
		//		delete sm1; sm1 = NULL;
		//      delete argmax1; argmax1 = NULL;
		//		delete mlr1; mlr1 = NULL;
		delete sml1; sml1 = NULL;
		delete accuracy1; accuracy1 = NULL;

		delete conv1_top; conv1_top = NULL;
		delete relu1_top; relu1_top = NULL;
		delete pool1_top; pool1 = NULL;
		delete conv2g_top; conv2g_top = NULL;
		delete relu2_top; relu2_top = NULL;
		delete pool2_top; pool2_top = NULL;
		delete conv3_top; conv3_top = NULL;
		delete relu3_top; relu3_top = NULL;
		delete conv4g_top; conv4g_top = NULL;
		delete relu4_top; relu4_top = NULL;
		delete conv5g_top; conv5g_top = NULL;
		delete relu5_top; relu5_top = NULL;
		delete pool5_top; pool5_top = NULL;
		delete fc6_top; fc6_top = NULL;
		delete relu6_top; relu6_top = NULL;
		delete fc7_top; fc7_top = NULL;
		delete relu7_top; relu7_top = NULL;
		delete fc8_top; fc8_top = NULL;
		//		delete sm1_top; sm1_top = NULL;
		//      delete argmax1_top; argmax1_top = NULL;
		//		delete mlr1_top; mlr1_top = NULL;
		delete sml1_top; sml1_top = NULL;
		delete accuracy1_top; accuracy1_top = NULL;

		delete conv1_params; conv1_params = NULL;
		delete relu1_params; relu1_params = NULL;
		delete pool1_params; pool1_params = NULL;
		delete conv2g_params; conv2g_params = NULL;
		delete relu2_params; relu2_params = NULL;
		delete pool2_params; pool2_params = NULL;
		delete conv3_params; conv3_params = NULL;
		delete relu3_params; relu3_params = NULL;
		delete conv4g_params; conv4g_params = NULL;
		delete relu4_params; relu4_params = NULL;
		delete conv5g_params; conv5g_params = NULL;
		delete relu5_params; relu5_params = NULL;
		delete pool5_params; pool5_params = NULL;
		delete fc6_params; fc6_params = NULL;
		delete relu6_params; relu6_params = NULL;
		delete fc7_params; fc7_params = NULL;
		delete relu7_params; relu7_params = NULL;
		delete fc8_params; fc8_params = NULL;
		//		delete sm1_params; sm1_params = NULL;
		//      delete argmax1_params; argmax1_params = NULL;
		//		delete mlr1_params; mlr1_params = NULL;
		delete sml1_params; sml1_params = NULL;
		delete accuracy1_params; accuracy1_params = NULL;

		delete conv1_filtersBlob_old; conv1_filtersBlob_old = NULL;
		delete conv1_biasBlob_old; conv1_biasBlob_old = NULL;
		delete conv2g_filtersBlob_old; conv2g_filtersBlob_old = NULL;
		delete conv2g_biasBlob_old; conv2g_biasBlob_old = NULL;
		delete conv3_filtersBlob_old; conv3_filtersBlob_old = NULL;
		delete conv3_biasBlob_old; conv3_biasBlob_old = NULL;
		delete conv4g_filtersBlob_old; conv4g_filtersBlob_old = NULL;
		delete conv4g_biasBlob_old; conv4g_biasBlob_old = NULL;
		delete conv5g_filtersBlob_old; conv5g_filtersBlob_old = NULL;
		delete conv5g_biasBlob_old; conv5g_biasBlob_old = NULL;
		delete fc6_filtersBlob_old; fc6_filtersBlob_old = NULL;
		delete fc6_biasBlob_old; fc6_biasBlob_old = NULL;
		delete fc7_filtersBlob_old; fc7_filtersBlob_old = NULL;
		delete fc7_biasBlob_old; fc7_biasBlob_old = NULL;
		delete fc8_filtersBlob_old; fc8_filtersBlob_old = NULL;
		delete fc8_biasBlob_old; fc8_biasBlob_old = NULL;

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

		batch_samples = new Blob_t(batch_size_, 3, 227, 227);
		batch_labels = new Blob_t(batch_size_, 1, 1, 1);
		batch_samples->allocate_gpu_data();
		batch_samples->allocate_gpu_diff();
		batch_labels->allocate_gpu_data();
		batch_labels->allocate_cpu_data();

		LOG(INFO) << "conv1 setup.\n";
		conv1_top = new Blob_t();
		conv1_params = new ConvolutionParameter_t();
		conv1_params->filter_N = 3;
		conv1_params->filter_C = 96;
		conv1_params->filter_H = 11;
		conv1_params->filter_W = 11;
		conv1_params->pad_h = 0;
		conv1_params->pad_w = 0;
		conv1_params->stride_h = 4;
		conv1_params->stride_w = 4;
		conv1_params->upscale_h = 1;
		conv1_params->upscale_w = 1;
		conv1_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv1 = new ConvolutionLayer_t(conv1_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->filtersBlob->data_gpu, conv1->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->biasBlob->data_gpu, conv1->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv1->biasBlob->count(), 0, conv1->biasBlob->data_gpu);
		conv1->Setup(batch_samples, conv1_top);
		LOG(INFO) << "conv1 top: "
				<< conv1_top->N << ", "
				<< conv1_top->C << ", "
				<< conv1_top->H << ", "
				<< conv1_top->W;


		LOG(INFO) << "relu1 setup.\n";
		relu1_top = new Blob_t();
		relu1_params = new ActivationParameter_t();
		relu1_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu1 = new ActivationLayer_t(relu1_params);
		relu1->Setup(conv1_top, relu1_top);
		LOG(INFO) << "relu1 top: "
				<< relu1_top->N << ", "
				<< relu1_top->C << ", "
				<< relu1_top->H << ", "
				<< relu1_top->W;

		LOG(INFO) << "pool1 setup.\n";
		pool1_top = new Blob_t();
		pool1_params = new PoolingParameter_t();
		pool1_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
		pool1_params->poolsize_h = 3;
		pool1_params->poolsize_w = 3;
		pool1_params->pad_h = 0;
		pool1_params->pad_w = 0;
		pool1_params->stride_h = 2;
		pool1_params->stride_w = 2;
		pool1 = new PoolingLayer_t(pool1_params);
		pool1->Setup(relu1_top, pool1_top);
		LOG(INFO) << "pool1 top: "
				<< pool1_top->N << ", "
				<< pool1_top->C << ", "
				<< pool1_top->H << ", "
				<< pool1_top->W;

		LOG(INFO) << "conv2g setup.\n";
		conv2g_top = new Blob_t();
		conv2g_params = new ConvolutionWithGroupParameter_t();
		conv2g_params->group = 2;
		conv2g_params->filter_N = 32;
		conv2g_params->filter_C = 256;
		conv2g_params->filter_H = 5;
		conv2g_params->filter_W = 5;
		conv2g_params->pad_h = 2;
		conv2g_params->pad_w = 2;
		conv2g_params->stride_h = 1;
		conv2g_params->stride_w = 1;
		conv2g_params->upscale_h = 1;
		conv2g_params->upscale_w = 1;
		conv2g_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv2g_params->cudnn_conv_fwd_preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
		conv2g = new ConvolutionWithGroupLayer_t(conv2g_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv2g->filtersBlob->data_gpu, conv2g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv2g->biasBlob->count(), 0, conv2g->biasBlob->data_gpu);
		conv2g->Setup(pool1_top, conv2g_top);
		LOG(INFO) << "conv2g top: "
				<< conv2g_top->N << ", "
				<< conv2g_top->C << ", "
				<< conv2g_top->H << ", "
				<< conv2g_top->W;


		LOG(INFO) << "relu2 setup.\n";
		relu2_top = new Blob_t();
		relu2_params = new ActivationParameter_t();
		relu2_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu2 = new ActivationLayer_t(relu2_params);
		relu2->Setup(conv2g_top, relu2_top);
		LOG(INFO) << "relu2 top: "
				<< relu2_top->N << ", "
				<< relu2_top->C << ", "
				<< relu2_top->H << ", "
				<< relu2_top->W;

		LOG(INFO) << "pool2 setup.\n";
		pool2_top = new Blob_t();
		pool2_params = new PoolingParameter_t();
		pool2_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
		pool2_params->poolsize_h = 3;
		pool2_params->poolsize_w = 3;
		pool2_params->pad_h = 0;
		pool2_params->pad_w = 0;
		pool2_params->stride_h = 2;
		pool2_params->stride_w = 2;
		pool2 = new PoolingLayer_t(pool2_params);
		pool2->Setup(relu2_top, pool2_top);
		LOG(INFO) << "pool2 top: "
				<< pool2_top->N << ", "
				<< pool2_top->C << ", "
				<< pool2_top->H << ", "
				<< pool2_top->W;

		LOG(INFO) << "conv3 setup.\n";
		conv3_top = new Blob_t();
		conv3_params = new ConvolutionParameter_t();
		conv3_params->filter_N = 256;
		conv3_params->filter_C = 384;
		conv3_params->filter_H = 3;
		conv3_params->filter_W = 3;
		conv3_params->pad_h = 1;
		conv3_params->pad_w = 1;
		conv3_params->stride_h = 1;
		conv3_params->stride_w = 1;
		conv3_params->upscale_h = 1;
		conv3_params->upscale_w = 1;
		conv3_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv3 = new ConvolutionLayer_t(conv3_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->filtersBlob->data_gpu, conv3->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->biasBlob->data_gpu, conv3->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv3->biasBlob->count(), 0, conv3->biasBlob->data_gpu);
		conv3->Setup(pool2_top, conv3_top);
		LOG(INFO) << "conv3 top: "
				<< conv3_top->N << ", "
				<< conv3_top->C << ", "
				<< conv3_top->H << ", "
				<< conv3_top->W;


		LOG(INFO) << "relu3 setup.\n";
		relu3_top = new Blob_t();
		relu3_params = new ActivationParameter_t();
		relu3_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu3 = new ActivationLayer_t(relu3_params);
		relu3->Setup(conv3_top, relu3_top);
		LOG(INFO) << "relu3 top: "
				<< relu3_top->N << ", "
				<< relu3_top->C << ", "
				<< relu3_top->H << ", "
				<< relu3_top->W;

		LOG(INFO) << "conv4g setup.\n";
		conv4g_top = new Blob_t();
		conv4g_params = new ConvolutionWithGroupParameter_t();
		conv4g_params->group = 2;
		conv4g_params->filter_N = 384;
		conv4g_params->filter_C = 384;
		conv4g_params->filter_H = 3;
		conv4g_params->filter_W = 3;
		conv4g_params->pad_h = 1;
		conv4g_params->pad_w = 1;
		conv4g_params->stride_h = 1;
		conv4g_params->stride_w = 1;
		conv4g_params->upscale_h = 1;
		conv4g_params->upscale_w = 1;
		conv4g_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv4g_params->cudnn_conv_fwd_preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
		conv4g = new ConvolutionWithGroupLayer_t(conv4g_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv4g->filtersBlob->data_gpu, conv4g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv4g->biasBlob->count(), 0, conv4g->biasBlob->data_gpu);
		conv4g->Setup(relu3_top, conv4g_top);
		LOG(INFO) << "conv4g top: "
				<< conv4g_top->N << ", "
				<< conv4g_top->C << ", "
				<< conv4g_top->H << ", "
				<< conv4g_top->W;

		LOG(INFO) << "relu4 setup.\n";
		relu4_top = new Blob_t();
		relu4_params = new ActivationParameter_t();
		relu4_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu4 = new ActivationLayer_t(relu4_params);
		relu4->Setup(conv4g_top, relu4_top);
		LOG(INFO) << "relu4 top: "
				<< relu4_top->N << ", "
				<< relu4_top->C << ", "
				<< relu4_top->H << ", "
				<< relu4_top->W;

		LOG(INFO) << ("conv5g setup.\n");
		conv5g_top = new Blob_t();
		conv5g_params = new ConvolutionWithGroupParameter_t();
		conv5g_params->group = 2;
		conv5g_params->filter_N = 384;
		conv5g_params->filter_C = 256;
		conv5g_params->filter_H = 3;
		conv5g_params->filter_W = 3;
		conv5g_params->pad_h = 1;
		conv5g_params->pad_w = 1;
		conv5g_params->stride_h = 1;
		conv5g_params->stride_w = 1;
		conv5g_params->upscale_h = 1;
		conv5g_params->upscale_w = 1;
		conv5g_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
		conv5g_params->cudnn_conv_fwd_preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
		conv5g = new ConvolutionWithGroupLayer_t(conv5g_params);
		CURAND_CHECK( curandGenerateNormal(curand_generator, conv5g->filtersBlob->data_gpu, conv5g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(conv5g->biasBlob->count(), float(0.1f), conv5g->biasBlob->data_gpu);
		conv5g->Setup(relu4_top, conv5g_top);
		LOG(INFO) << "conv5g top: "
				<< conv5g_top->N << ", "
				<< conv5g_top->C << ", "
				<< conv5g_top->H << ", "
				<< conv5g_top->W;

		LOG(INFO) << ("relu5 setup.\n");
		relu5_top = new Blob_t();
		relu5_params = new ActivationParameter_t();
		relu5_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu5 = new ActivationLayer_t(relu5_params);
		relu5->Setup(conv5g_top, relu5_top);
		LOG(INFO) << "relu5 top: "
				<< relu5_top->N << ", "
				<< relu5_top->C << ", "
				<< relu5_top->H << ", "
				<< relu5_top->W;

		LOG(INFO) << ("pool5 setup.\n");
		pool5_top = new Blob_t();
		pool5_params = new PoolingParameter_t();
		pool5_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
		pool5_params->poolsize_h = 3;
		pool5_params->poolsize_w = 3;
		pool5_params->pad_h = 0;
		pool5_params->pad_w = 0;
		pool5_params->stride_h = 2;
		pool5_params->stride_w = 2;
		pool5 = new PoolingLayer_t(pool5_params);
		pool5->Setup(relu5_top, pool5_top);
		LOG(INFO) << "pool5 top: "
				<< pool5_top->N << ", "
				<< pool5_top->C << ", "
				<< pool5_top->H << ", "
				<< pool5_top->W;

		LOG(INFO) << ("fc6 setup.\n");
		fc6_top = new Blob_t();
		fc6_params = new FullyConnectedParameter_t();
		fc6_params->hidden_size = 4096;
		fc6 = new FullyConnectedLayer_t(fc6_params);
		fc6->Setup(pool5_top, fc6_top);
		CURAND_CHECK( curandGenerateNormal(curand_generator, fc6->filtersBlob->data_gpu, fc6->filtersBlob->count(), (float)0.0f, (float)0.005f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(fc6->biasBlob->count(), float(0.1f), fc6->biasBlob->data_gpu);
		LOG(INFO) << "fc6 top: "
				<< fc6_top->N << ", "
				<< fc6_top->C << ", "
				<< fc6_top->H << ", "
				<< fc6_top->W;

		LOG(INFO) << ("relu6 setup.\n");
		relu6_top = new Blob_t();
		relu6_params = new ActivationParameter_t();
		relu6_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu6 = new ActivationLayer_t(relu6_params);
		relu6->Setup(fc6_top, relu6_top);
		LOG(INFO) << "relu6 top: "
				<< relu6_top->N << ", "
				<< relu6_top->C << ", "
				<< relu6_top->H << ", "
				<< relu6_top->W;

		LOG(INFO) << ("fc7 setup.\n");
		fc7_top = new Blob_t();
		fc7_params = new FullyConnectedParameter_t();
		fc7_params->hidden_size = 4096;
		fc7 = new FullyConnectedLayer_t(fc7_params);
		fc7->Setup(relu6_top, fc7_top);
		CURAND_CHECK( curandGenerateNormal(curand_generator, fc7->filtersBlob->data_gpu, fc7->filtersBlob->count(), (float)0.0f, (float)0.005f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(fc7->biasBlob->count(), float(0.1f), fc7->biasBlob->data_gpu);
		LOG(INFO) << "fc7 top: "
				<< fc7_top->N << ", "
				<< fc7_top->C << ", "
				<< fc7_top->H << ", "
				<< fc7_top->W;

		LOG(INFO) << ("relu7 setup.\n");
		relu7_top = new Blob_t();
		relu7_params = new ActivationParameter_t();
		relu7_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
		relu7 = new ActivationLayer_t(relu7_params);
		relu7->Setup(fc7_top, relu7_top);
		LOG(INFO) << "relu7 top: "
				<< relu7_top->N << ", "
				<< relu7_top->C << ", "
				<< relu7_top->H << ", "
				<< relu7_top->W;

		LOG(INFO) << ("fc8 setup.\n");
		fc8_top = new Blob_t();
		fc8_params = new FullyConnectedParameter_t();
		fc8_params->hidden_size = 1000;
		fc8 = new FullyConnectedLayer_t(fc8_params);
		fc8->Setup(relu7_top, fc8_top);
		CURAND_CHECK( curandGenerateNormal(curand_generator, fc8->filtersBlob->data_gpu, fc8->filtersBlob->count(), (float)0.0f, (float)0.01f) );
		// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
		gpu_set(fc8->biasBlob->count(), float(0.0f), fc8->biasBlob->data_gpu);
		LOG(INFO) << "fc8 top: "
				<< fc8_top->N << ", "
				<< fc8_top->C << ", "
				<< fc8_top->H << ", "
				<< fc8_top->W;

		//		printf("sm1 setup.\n");
		//		sm1_top = new Blob_t();
		//		sm1_params = new SoftmaxParameter_t();
		//		sm1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
		//		sm1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
		//		sm1 = new SoftmaxLayer_t(sm1_params);
		//		sm1->Setup(ip1_top, sm1_top);
		//
		//		printf("mlr1 setup (in cpu).\n");
		//		mlr1_top = new Blob_t();
		//		mlr1_params = new MultinomialLogisticLossParameter_t();
		//		mlr1 = new MultinomialLogisticLossLayer_t(mlr1_params);
		//		mlr1->Setup(sm1_top, mlr1_top);

		LOG(INFO) << ("sml1 setup.\n");
		sml1_top = new Blob_t();
		sml1_params = new SoftmaxWithLossParameter_t();
		sml1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
		sml1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
		sml1_params->has_ignore_label = false;
		sml1_params->ignore_label = -1;
		sml1_params->normalize = false;
		sml1 = new SoftmaxWithLossLayer_t(sml1_params);
		sml1->Setup(fc8_top, sml1_top);
		LOG(INFO) << "sml1 top: "
				<< sml1_top->N << ", "
				<< sml1_top->C << ", "
				<< sml1_top->H << ", "
				<< sml1_top->W;

		//		printf("argmax1 setup.\n");
		//		argmax1_top = new Blob_t();
		//		argmax1_params = new ArgMaxParameter_t();
		//		argmax1_params->out_max_val = true;
		//		argmax1_params->top_k = 1;
		//		argmax1 = new ArgMaxLayer_t(argmax1_params);
		//		argmax1->Setup(sml1_top, argmax1_top);

		LOG(INFO) << ("accuracy1 setup.\n");
		accuracy1_top = new Blob_t();
		accuracy1_params = new AccuracyParameter_t();
		accuracy1_params->top_k = 1;
		accuracy1 = new AccuracyLayer_t(accuracy1_params);
		accuracy1->Setup(fc8_top, accuracy1_top);
		LOG(INFO) << "accuracy1 top: "
				<< accuracy1_top->N << ", "
				<< accuracy1_top->C << ", "
				<< accuracy1_top->H << ", "
				<< accuracy1_top->W;

		LOG(INFO) << ("initialize old net params.\n");
		conv1_filtersBlob_old = new Blob_t(conv1->filtersBlob->N, conv1->filtersBlob->C, conv1->filtersBlob->H, conv1->filtersBlob->W);
		conv1_biasBlob_old = new Blob_t(conv1->biasBlob->N, conv1->biasBlob->C, conv1->biasBlob->H, conv1->biasBlob->W);
		conv1_filtersBlob_old->allocate_gpu_data();
		conv1_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv1_filtersBlob_old->data_gpu, conv1_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv1_biasBlob_old->data_gpu, conv1_biasBlob_old->count(), 0.0f, 0.0f);

		conv2g_filtersBlob_old = new Blob_t(conv2g->filtersBlob->N, conv2g->filtersBlob->C, conv2g->filtersBlob->H, conv2g->filtersBlob->W);
		conv2g_biasBlob_old = new Blob_t(conv2g->biasBlob->N, conv2g->biasBlob->C, conv2g->biasBlob->H, conv2g->biasBlob->W);
		conv2g_filtersBlob_old->allocate_gpu_data();
		conv2g_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv2g_filtersBlob_old->data_gpu, conv2g_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv2g_biasBlob_old->data_gpu, conv2g_biasBlob_old->count(), 0.0f, 0.0f);

		conv3_filtersBlob_old = new Blob_t(conv3->filtersBlob->N, conv3->filtersBlob->C, conv3->filtersBlob->H, conv3->filtersBlob->W);
		conv3_biasBlob_old = new Blob_t(conv3->biasBlob->N, conv3->biasBlob->C, conv3->biasBlob->H, conv3->biasBlob->W);
		conv3_filtersBlob_old->allocate_gpu_data();
		conv3_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv3_filtersBlob_old->data_gpu, conv3_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv3_biasBlob_old->data_gpu, conv3_biasBlob_old->count(), 0.0f, 0.0f);

		conv4g_filtersBlob_old = new Blob_t(conv4g->filtersBlob->N, conv4g->filtersBlob->C, conv4g->filtersBlob->H, conv4g->filtersBlob->W);
		conv4g_biasBlob_old = new Blob_t(conv4g->biasBlob->N, conv4g->biasBlob->C, conv4g->biasBlob->H, conv4g->biasBlob->W);
		conv4g_filtersBlob_old->allocate_gpu_data();
		conv4g_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv4g_filtersBlob_old->data_gpu, conv4g_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv4g_biasBlob_old->data_gpu, conv4g_biasBlob_old->count(), 0.0f, 0.0f);

		conv5g_filtersBlob_old = new Blob_t(conv5g->filtersBlob->N, conv5g->filtersBlob->C, conv5g->filtersBlob->H, conv5g->filtersBlob->W);
		conv5g_biasBlob_old = new Blob_t(conv5g->biasBlob->N, conv5g->biasBlob->C, conv5g->biasBlob->H, conv5g->biasBlob->W);
		conv5g_filtersBlob_old->allocate_gpu_data();
		conv5g_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, conv5g_filtersBlob_old->data_gpu, conv5g_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, conv5g_biasBlob_old->data_gpu, conv5g_biasBlob_old->count(), 0.0f, 0.0f);

		fc6_filtersBlob_old = new Blob_t(fc6->filtersBlob->N, fc6->filtersBlob->C, fc6->filtersBlob->H, fc6->filtersBlob->W);
		fc6_biasBlob_old = new Blob_t(fc6->biasBlob->N, fc6->biasBlob->C, fc6->biasBlob->H, fc6->biasBlob->W);
		fc6_filtersBlob_old->allocate_gpu_data();
		fc6_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, fc6_filtersBlob_old->data_gpu, fc6_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, fc6_biasBlob_old->data_gpu, fc6_biasBlob_old->count(), 0.0f, 0.0f);

		fc7_filtersBlob_old = new Blob_t(fc7->filtersBlob->N, fc7->filtersBlob->C, fc7->filtersBlob->H, fc7->filtersBlob->W);
		fc7_biasBlob_old = new Blob_t(fc7->biasBlob->N, fc7->biasBlob->C, fc7->biasBlob->H, fc7->biasBlob->W);
		fc7_filtersBlob_old->allocate_gpu_data();
		fc7_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, fc7_filtersBlob_old->data_gpu, fc7_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, fc7_biasBlob_old->data_gpu, fc7_biasBlob_old->count(), 0.0f, 0.0f);

		fc8_filtersBlob_old = new Blob_t(fc8->filtersBlob->N, fc8->filtersBlob->C, fc8->filtersBlob->H, fc8->filtersBlob->W);
		fc8_biasBlob_old = new Blob_t(fc8->biasBlob->N, fc8->biasBlob->C, fc8->biasBlob->H, fc8->biasBlob->W);
		fc8_filtersBlob_old->allocate_gpu_data();
		fc8_biasBlob_old->allocate_gpu_data();
		gpu_fill(NULL, fc8_filtersBlob_old->data_gpu, fc8_filtersBlob_old->count(), 0.0f, 0.0f);
		gpu_fill(NULL, fc8_biasBlob_old->data_gpu, fc8_biasBlob_old->count(), 0.0f, 0.0f);

		LOG(INFO) << ("build net (done).\n");
	}

	void Forward(float *loss, float *accuracy) {
		cudaSetDevice(gpu_id);

		conv1->Forward(batch_samples, conv1_top);
		relu1->Forward(conv1_top, relu1_top);
		pool1->Forward(relu1_top, pool1_top);

		conv2g->Forward(pool1_top, conv2g_top);
		relu2->Forward(conv2g_top, relu2_top);
		pool2->Forward(relu2_top, pool2_top);

		conv3->Forward(pool2_top, conv3_top);
		relu3->Forward(conv3_top, relu3_top);

		conv4g->Forward(relu3_top, conv4g_top);
		relu4->Forward(conv4g_top, relu4_top);

		conv5g->Forward(relu4_top, conv5g_top);
		relu5->Forward(conv5g_top, relu5_top);
		pool5->Forward(relu5_top, pool5_top);

		fc6->Forward(pool5_top, fc6_top);
		relu6->Forward(fc6_top, relu6_top);

		fc7->Forward(relu6_top, fc7_top);
		relu7->Forward(fc7_top, relu7_top);

		fc8->Forward(relu7_top, fc8_top);

		// sm1->Forward(fc8_top, sm1_top);

		// mlr1->Forward(sm1_top, batch_labels, mlr1_top);

		// loss = mlr1_top->data_cpu[0];

		sml1->Forward(fc8_top, batch_labels, sml1_top, loss);

		//		argmax1->Forward_cpu(sml1_top, argmax1_top);

		accuracy1->Forward_cpu(fc8_top, batch_labels, accuracy1_top);

		*accuracy = accuracy1_top->data_cpu[0];

	}

	void Backward() {
		cudaSetDevice(gpu_id);

		sml1->Backward(sml1_top, batch_labels, fc8_top);

		// mlr1->Backward(mlr1_top, batch_labels, sm1_top);

		// sm1->Backward(sm1_top, ip1_top);

		fc8->Backward(fc8_top, relu7_top);

		relu7->Backward(relu7_top, fc7_top);
		fc7->Backward(fc7_top, relu6_top);

		relu6->Backward(relu6_top, fc6_top);
		fc6->Backward(fc6_top, pool5_top);

		pool5->Backward(pool5_top, relu5_top);
		relu5->Backward(relu5_top, conv5g_top);
		conv5g->Backward(conv5g_top, relu4_top);

		relu4->Backward(relu4_top, conv4g_top);
		conv4g->Backward(conv4g_top, relu3_top);

		relu3->Backward(relu3_top, conv3_top);
		conv3->Backward(conv3_top, pool2_top);

		pool2->Backward(pool2_top, relu2_top);
		relu2->Backward(relu2_top, conv2g_top);
		conv2g->Backward(conv2g_top, pool1_top);

		pool1->Backward(pool1_top, relu1_top);
		relu1->Backward(relu1_top, conv1_top);
		conv1->Backward(conv1_top, batch_samples);
	}

	void ForwardBackward(float *loss, float *accuracy) {
		Forward(loss, accuracy);
		Backward();
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
		ComputeUpdateValueSingle(fc8->filtersBlob,   fc8_filtersBlob_old,   lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(fc8->biasBlob,      fc8_biasBlob_old, 		lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(fc7->filtersBlob,   fc7_filtersBlob_old,   lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(fc7->biasBlob,      fc7_biasBlob_old, 		lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(fc6->filtersBlob,   fc6_filtersBlob_old,   lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(fc6->biasBlob,      fc6_biasBlob_old, 		lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv5g->filtersBlob, conv5g_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv5g->biasBlob, 	 conv5g_biasBlob_old, 	lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv4g->filtersBlob, conv4g_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv4g->biasBlob, 	 conv4g_biasBlob_old, 	lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv3->filtersBlob, conv3_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv3->biasBlob, 	 conv3_biasBlob_old, 	lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv2g->filtersBlob, conv2g_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv2g->biasBlob, 	 conv2g_biasBlob_old, 	lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv1->filtersBlob, conv1_filtersBlob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(conv1->biasBlob, 	 conv1_biasBlob_old, 	lr_rate, momentum, weight_decay);
	}

	void UpdateNet(float scale = -1.0f) {
		cudaSetDevice(gpu_id);
		gpu_axpy(cublas_handle, fc8->filtersBlob->count(),   float(scale), fc8->filtersBlob->diff_gpu,   fc8->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, fc8->biasBlob->count(), 	 float(scale), fc8->biasBlob->diff_gpu,      fc8->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, fc7->filtersBlob->count(),   float(scale), fc7->filtersBlob->diff_gpu,   fc7->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, fc7->biasBlob->count(), 	 float(scale), fc7->biasBlob->diff_gpu,      fc7->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, fc6->filtersBlob->count(),   float(scale), fc6->filtersBlob->diff_gpu,   fc6->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, fc6->biasBlob->count(), 	 float(scale), fc6->biasBlob->diff_gpu,      fc6->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, conv5g->filtersBlob->count(), float(scale), conv5g->filtersBlob->diff_gpu, conv5g->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv5g->biasBlob->count(), 	 float(scale), conv5g->biasBlob->diff_gpu, 	  conv5g->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, conv4g->filtersBlob->count(), float(scale), conv4g->filtersBlob->diff_gpu, conv4g->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv4g->biasBlob->count(), 	 float(scale), conv4g->biasBlob->diff_gpu, 	  conv4g->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, conv3->filtersBlob->count(), float(scale), conv3->filtersBlob->diff_gpu, conv3->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv3->biasBlob->count(), 	 float(scale), conv3->biasBlob->diff_gpu, 	  conv3->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, conv2g->filtersBlob->count(), float(scale), conv2g->filtersBlob->diff_gpu, conv2g->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv2g->biasBlob->count(), 	 float(scale), conv2g->biasBlob->diff_gpu, 	  conv2g->biasBlob->data_gpu);
		gpu_axpy(cublas_handle, conv1->filtersBlob->count(), float(scale), conv1->filtersBlob->diff_gpu, conv1->filtersBlob->data_gpu);
		gpu_axpy(cublas_handle, conv1->biasBlob->count(), 	 float(scale), conv1->biasBlob->diff_gpu,    conv1->biasBlob->data_gpu);
	}

	void SaveNetParams(int epoch) {
		cudaSetDevice(gpu_id);
		stringstream f1; f1 << net_name << "_c1_weight_e" << epoch << ".mat";
		conv1->filtersBlob->save_cpu_data_and_diff_to_mat(f1.str().c_str());
		stringstream f2; f2 << net_name << "_c1_bias_e" << epoch << ".mat";
		conv1->biasBlob->save_cpu_data_and_diff_to_mat(f2.str().c_str());

		stringstream f3; f3 << net_name << "_c2g_weight_e" << epoch << ".mat";
		conv2g->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
		stringstream f4; f4 << net_name << "_c2g_bias_e" << epoch << ".mat";
		conv2g->biasBlob->save_cpu_data_and_diff_to_mat(f4.str().c_str());

		stringstream f5; f5 << net_name << "_c3_weight_e" << epoch << ".mat";
		conv3->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
		stringstream f6; f6 << net_name << "_c3_bias_e" << epoch << ".mat";
		conv3->biasBlob->save_cpu_data_and_diff_to_mat(f6.str().c_str());

		stringstream f31; f31 << net_name << "_c4g_weight_e" << epoch << ".mat";
		conv4g->filtersBlob->save_cpu_data_and_diff_to_mat(f31.str().c_str());
		stringstream f41; f41 << net_name << "_c4g_bias_e" << epoch << ".mat";
		conv4g->biasBlob->save_cpu_data_and_diff_to_mat(f41.str().c_str());

		stringstream f311; f311 << net_name << "_c5g_weight_e" << epoch << ".mat";
		conv5g->filtersBlob->save_cpu_data_and_diff_to_mat(f311.str().c_str());
		stringstream f411; f411 << net_name << "_c5g_bias_e" << epoch << ".mat";
		conv5g->biasBlob->save_cpu_data_and_diff_to_mat(f411.str().c_str());

		stringstream f7; f7 << net_name << "_fc6_weight_e" << epoch << ".mat";
		fc6->filtersBlob->save_cpu_data_and_diff_to_mat(f7.str().c_str());
		stringstream f8; f8 << net_name << "_fc6_bias_e" << epoch << ".mat";
		fc6->biasBlob->save_cpu_data_and_diff_to_mat(f8.str().c_str());

		stringstream f71; f71<< net_name << "_fc7_weight_e" << epoch << ".mat";
		fc7->filtersBlob->save_cpu_data_and_diff_to_mat(f71.str().c_str());
		stringstream f81; f81 << net_name << "_fc7_bias_e" << epoch << ".mat";
		fc7->biasBlob->save_cpu_data_and_diff_to_mat(f81.str().c_str());

		stringstream f711; f711<< net_name << "_fc8_weight_e" << epoch << ".mat";
		fc8->filtersBlob->save_cpu_data_and_diff_to_mat(f711.str().c_str());
		stringstream f811; f811 << net_name << "_fc8_bias_e" << epoch << ".mat";
		fc8->biasBlob->save_cpu_data_and_diff_to_mat(f811.str().c_str());

	}

	void CopyNetParamsFrom(const AlexNetwork_t *other) {
		CopyBlobData_gpu(other->fc8->filtersBlob, 	other->gpu_id, fc8->filtersBlob,   gpu_id);
		CopyBlobData_gpu(other->fc8->biasBlob, 		other->gpu_id, fc8->biasBlob, 	   gpu_id);
		CopyBlobData_gpu(other->fc7->filtersBlob, 	other->gpu_id, fc7->filtersBlob,   gpu_id);
		CopyBlobData_gpu(other->fc7->biasBlob, 		other->gpu_id, fc7->biasBlob, 	   gpu_id);
		CopyBlobData_gpu(other->fc6->filtersBlob, 	other->gpu_id, fc6->filtersBlob,   gpu_id);
		CopyBlobData_gpu(other->fc6->biasBlob, 		other->gpu_id, fc6->biasBlob, 	   gpu_id);
		CopyBlobData_gpu(other->conv5g->filtersBlob, other->gpu_id, conv5g->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv5g->biasBlob, 	other->gpu_id, conv5g->biasBlob,    gpu_id);
		CopyBlobData_gpu(other->conv4g->filtersBlob, other->gpu_id, conv4g->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv4g->biasBlob, 	other->gpu_id, conv4g->biasBlob,    gpu_id);
		CopyBlobData_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv3->biasBlob, 	other->gpu_id, conv3->biasBlob,	   gpu_id);
		CopyBlobData_gpu(other->conv2g->filtersBlob, other->gpu_id, conv2g->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv2g->biasBlob, 	other->gpu_id, conv2g->biasBlob,    gpu_id);
		CopyBlobData_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
		CopyBlobData_gpu(other->conv1->biasBlob, 	other->gpu_id, conv1->biasBlob,    gpu_id);
	}

	void AddNetParamsDiffFrom(const AlexNetwork_t *other) {
		AddBlobDiff_gpu(other->fc8->filtersBlob, 	other->gpu_id, fc8->filtersBlob,   gpu_id);
		AddBlobDiff_gpu(other->fc8->biasBlob, 		other->gpu_id, fc8->biasBlob, 	   gpu_id);
		AddBlobDiff_gpu(other->fc7->filtersBlob, 	other->gpu_id, fc7->filtersBlob,   gpu_id);
		AddBlobDiff_gpu(other->fc7->biasBlob, 		other->gpu_id, fc7->biasBlob, 	   gpu_id);
		AddBlobDiff_gpu(other->fc6->filtersBlob, 	other->gpu_id, fc6->filtersBlob,   gpu_id);
		AddBlobDiff_gpu(other->fc6->biasBlob, 		other->gpu_id, fc6->biasBlob, 	   gpu_id);
		AddBlobDiff_gpu(other->conv5g->filtersBlob, other->gpu_id, conv5g->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv5g->biasBlob, 	other->gpu_id, conv5g->biasBlob,    gpu_id);
		AddBlobDiff_gpu(other->conv4g->filtersBlob, other->gpu_id, conv4g->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv4g->biasBlob, 	other->gpu_id, conv4g->biasBlob,    gpu_id);
		AddBlobDiff_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv3->biasBlob,    other->gpu_id, conv3->biasBlob,    gpu_id);
		AddBlobDiff_gpu(other->conv2g->filtersBlob, other->gpu_id, conv2g->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv2g->biasBlob,    other->gpu_id, conv2g->biasBlob, 	  gpu_id);
		AddBlobDiff_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
		AddBlobDiff_gpu(other->conv1->biasBlob,    other->gpu_id, conv1->biasBlob, 	  gpu_id);
	}

	void ClearNetParamsDiff() {
		cudaSetDevice(gpu_id);
		gpu_set(fc8->filtersBlob->count(),   0, fc8->filtersBlob->diff_gpu);
		gpu_set(fc8->biasBlob->count(),      0, fc8->biasBlob->diff_gpu);
		gpu_set(fc7->filtersBlob->count(),   0, fc7->filtersBlob->diff_gpu);
		gpu_set(fc7->biasBlob->count(),      0, fc7->biasBlob->diff_gpu);
		gpu_set(fc6->filtersBlob->count(),   0, fc6->filtersBlob->diff_gpu);
		gpu_set(fc6->biasBlob->count(),      0, fc6->biasBlob->diff_gpu);
		gpu_set(conv5g->filtersBlob->count(), 0, conv5g->filtersBlob->diff_gpu);
		gpu_set(conv5g->biasBlob->count(),    0, conv5g->biasBlob->diff_gpu);
		gpu_set(conv4g->filtersBlob->count(), 0, conv4g->filtersBlob->diff_gpu);
		gpu_set(conv4g->biasBlob->count(),    0, conv4g->biasBlob->diff_gpu);
		gpu_set(conv3->filtersBlob->count(), 0, conv3->filtersBlob->diff_gpu);
		gpu_set(conv3->biasBlob->count(), 	 0, conv3->biasBlob->diff_gpu);
		gpu_set(conv2g->filtersBlob->count(), 0, conv2g->filtersBlob->diff_gpu);
		gpu_set(conv2g->biasBlob->count(),    0, conv2g->biasBlob->diff_gpu);
		gpu_set(conv1->filtersBlob->count(), 0, conv1->filtersBlob->diff_gpu);
		gpu_set(conv1->biasBlob->count(),    0, conv1->biasBlob->diff_gpu);
	}

};

pthread_barrier_t barr;
struct thread_data_t
{
public:
	Blob_t *batch_samples;
	Blob_t *batch_labels;
	Network_t *net;
	int main_gpu_id;
	int net_gpu_id;
	float lr_rate;
	float momentum;
	float weight_decay;
};

void do_slave(void *data_)
{
	thread_data_t *data = (thread_data_t *)data_;
	cudaSetDevice(data->net_gpu_id);
	// CUDA_CHECK( cudaMemcpy(data->net->batch_samples->data_gpu, data->batch_samples->data_cpu, data->batch_samples->count() * sizeof(float), cudaMemcpyHostToDevice) );
	// CUDA_CHECK( cudaMemcpy(data->net->batch_labels->data_gpu, data->batch_labels->data_cpu, data->batch_labels->count() * sizeof(float), cudaMemcpyHostToDevice) );
	float trn_loss, trn_acc;
	data->net->ForwardBackward(&trn_loss, &trn_acc);
	data->net->ComputeUpdateValue(data->lr_rate, data->momentum, data->weight_decay);
	// printf("gpuid[%d]: trn_loss=%.6f, trn_acc=%.6f\n", data->net_gpu_id, trn_loss, trn_acc);

	pthread_barrier_wait(&barr);
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


	int main_gpu_id;
	cudaGetDevice(&main_gpu_id);
	printf("current gpu id: %d\n", main_gpu_id);

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
		gpus.push_back(main_gpu_id);
	}

	cudaSetDevice(main_gpu_id);

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

	cudaSetDevice(main_gpu_id);

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
		thread_data[i].main_gpu_id = main_gpu_id;
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

	cudaSetDevice(main_gpu_id);
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

int main_mgpu_ok_loss_is_decreasing(int argc, char *argv[]) {
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


	int main_gpu_id;
	cudaGetDevice(&main_gpu_id);
	printf("current gpu id: %d\n", main_gpu_id);

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
		gpus.push_back(main_gpu_id);
	}

	cudaSetDevice(main_gpu_id);

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

	cudaSetDevice(main_gpu_id);

	Blob_t *trn_batch_samples = new Blob_t();
	Blob_t *trn_batch_labels = new Blob_t();
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	Blob_t *tst_batch_samples = new Blob_t();
	Blob_t *tst_batch_labels = new Blob_t();
	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	Network_t *trn_net = new Network_t("trn_net", main_gpu_id);
	trn_net->BuildNet(trn_batch_size, "");
	trn_net->batch_labels->allocate_cpu_data();
	Network_t *tst_net = new Network_t("tst_net", main_gpu_id);
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
		thread_data[i].main_gpu_id = main_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples_slices[i];
		thread_data[i].batch_labels = batch_labels_slices[i];
	}

	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_acc = 0.0f;
		tst_net->CopyNetParamsFrom(trn_net);
		for(int iter = 0; iter < floor(10000 / tst_batch_size); iter++) {
			tst_data_layer->Forward_cpu(tst_batch_samples, tst_batch_labels);
			tst_net->Forward(&tst_loss, &tst_acc);
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
			cudaSetDevice(main_gpu_id);
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

	cudaSetDevice(main_gpu_id);
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

int main_test_memcpy(int argc, char **argv) {

	int N = 64;
	float *data_h = NULL;
	MallocHost((void **)&data_h, N * sizeof(float));
	for(int i = 0; i < N; i++) {
		data_h[i] = (float)rand() / (float)RAND_MAX;
	}

	cudaSetDevice(1);
	float *data_d = NULL;
	CUDA_CHECK( cudaMalloc((void **)&data_d, N * sizeof(float)) );
	cudaSetDevice(2);
	CUDA_CHECK( cudaMemset(data_d, 0, N * sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(data_d, data_h, N * sizeof(float), cudaMemcpyHostToDevice) );
	float *data_d_copy = NULL;
	CUDA_CHECK( cudaMalloc((void **)&data_d_copy, N * sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(data_d_copy, data_d, N * sizeof(float), cudaMemcpyDeviceToDevice) );

	float *data_h2 = new float[N];
	CUDA_CHECK( cudaMemcpy(data_h2, data_d_copy, N * sizeof(float), cudaMemcpyDeviceToHost) );
	bool isyes = true;
	for(int i = 0; i < N; i++) {
		if(abs(data_h[i] - data_h2[i]) > 1e-6) {
			isyes = false;
			break;
		}
	}
	printf("data_h %s data_h2\n", isyes ? "==" : "!=");

	cudaSetDevice(2);
	float *gpu2_data_d = NULL;
	CUDA_CHECK( cudaMalloc((void**)&gpu2_data_d, N *sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(gpu2_data_d, data_d, N * sizeof(float), cudaMemcpyDefault) );
	CUDA_CHECK( cudaMemcpy(data_h2, data_d_copy, N * sizeof(float), cudaMemcpyDeviceToHost) );
	isyes = true;
	for(int i = 0; i < N; i++) {
		if(abs(data_h[i] - data_h2[i]) > 1e-6) {
			isyes = false;
			break;
		}
	}
	printf("data_h %s data_h2\n", isyes ? "==" : "!=");

	cudaSetDevice(2);
	CUDA_CHECK( cudaFree(gpu2_data_d) );

	cudaSetDevice(1);
	CUDA_CHECK( cudaFree(data_d) );
	CUDA_CHECK( cudaFree(data_d_copy) );
	FreeHost(data_h);
	delete[] data_h2;
	cudaDeviceReset();
	return 0;
}

int main_single_gpu_ok(int argc, char **argv) {
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

	int main_gpu_id = 0;
	cudaSetDevice(main_gpu_id);
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	Network_t *trn_net = new Network_t("trn_net", main_gpu_id);
	trn_net->BuildNet(trn_batch_size, "");

	Network_t *tst_net = new Network_t("tst_net", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");

	int num_tst_iters = floor(10000 / tst_batch_size);
	int num_trn_iters = floor(50000 / trn_batch_size);
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		tst_net->CopyNetParamsFrom(trn_net);
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(tst_net->batch_samples, tst_net->batch_labels);
			tst_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;

		// training net
		float trn_loss = 0.0f, trn_loss_batch = 0.0f;
		float trn_acc  = 0.0f, trn_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network(trn_net->batch_samples, trn_net->batch_labels);
			trn_net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
			trn_loss += trn_loss_batch;
			trn_acc  += trn_acc_batch;
			trn_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);
			trn_net->UpdateNet();
		}
		trn_loss /= num_trn_iters;
		trn_acc  /= num_trn_iters;

		// update learning rate
		if((epoch != 0) && (epoch % lr_stepsize == 0))
		{
			lr_rate /= 10;
			trn_net->SaveNetParams(epoch);
		}
		printf("epoch[%d]: trn_loss=%.6f, trn_acc=%.6f, tst_loss=%.6f, tst_acc=%.6f\n",
				epoch, trn_loss, trn_acc, tst_loss, tst_acc);
	}

	delete trn_net;
	delete tst_net;

	delete trn_data_layer;
	delete tst_data_layer;
	delete trn_data_param;
	delete tst_data_param;

	cudaDeviceReset();
	return 0;
}

int main(int argc, char *argv[]) {
	if(argc != 13) {
		LOG(FATAL) << ("Usage: <filename> main_gpu_id trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n");
		return -1;
	}
	int main_gpu_id = atoi(argv[1]);
	string trn_db_filename = string(argv[2]);
	string tst_db_filename = string(argv[3]);
	string mean_file = string(argv[4]);
	float lr_rate = atof(argv[5]);
	int lr_stepsize = atoi(argv[6]);
	float momentum = atof(argv[7]);
	float weight_decay = atof(argv[8]);
	int trn_batch_size = atoi(argv[9]);
	int tst_batch_size = atoi(argv[10]);
	int max_epoch_num = atoi(argv[11]);
	string gpu_ids_str = string(argv[12]);

	cudaSetDevice(main_gpu_id);
	LOG(INFO) << "current gpu id: " << main_gpu_id;

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	LOG(INFO) << "number of manually-set gpus: " << gpus.size() <<
			"total " << num_gpus << " gpus.";

	if(num_gpus >= gpus.size()) {
		LOG(INFO) << ("enable P2P: ");
		EnableP2P(gpus);
	} else {
		gpus.clear();
		gpus.push_back(main_gpu_id);
	}

	if(trn_batch_size % gpus.size() != 0) {
		LOG(FATAL) << "trn_batch_size: " << trn_batch_size
				<< ", number of given gpus: " << gpus.size()
				<< ", trn_batch_size must be times of the number of given gpus.";
		return -1;
	}

	cudaSetDevice(main_gpu_id);
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	cudaSetDevice(main_gpu_id);
	Network_t *tst_net = new Network_t("tst_net", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");
	tst_net->SaveNetParams(0);

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
	LOG(INFO) << ("initialize nets for each gpu ...\n");
	for(int i = 0; i < gpus.size(); i++)
	{
		LOG(INFO) << "gpu[" <<  gpus[i] << "]:\n";
		cudaSetDevice(main_gpu_id);
		batch_sizes[i] = trn_batch_size / gpus.size();
		trn_nets[i] = new Network_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(batch_sizes[i], "");

		batch_samples_slices[i] = trn_nets[i]->batch_samples;
		batch_labels_slices[i] = trn_nets[i]->batch_labels;
	}
	LOG(INFO) << ("initialize nets for each gpu (done) ...\n");

	pthread_t *threads;
	pthread_attr_t pta;
	threads = (pthread_t *) malloc(sizeof(pthread_t) * gpus.size());
	int ret_count = pthread_attr_init(&pta);
	pthread_attr_setdetachstate(&pta, PTHREAD_CREATE_JOINABLE);
	pthread_barrier_init(&barr, NULL, gpus.size());

	thread_data_t thread_data[gpus.size()];
	for(int i = 0; i < gpus.size(); i++) {
		thread_data[i].lr_rate = lr_rate;
		thread_data[i].momentum = momentum;
		thread_data[i].weight_decay = weight_decay;
		thread_data[i].main_gpu_id = main_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples_slices[i];
		thread_data[i].batch_labels = batch_labels_slices[i];
	}

	int num_tst_iters = floor(10000 / tst_batch_size);
	int num_trn_iters = floor(50000 / trn_batch_size);
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(tst_net->batch_samples, tst_net->batch_labels);
			tst_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc  += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;
		LOG(INFO) << "epoch[" << epoch << "]: tst_loss=" << tst_loss << ", tst_acc=" << tst_acc << "\n";

		// training net
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network_multi(gpus, batch_sizes, batch_samples_slices, batch_labels_slices);

			// copy trn_net params into trn_nets_i
			for(int i = 0; i < gpus.size(); i++) {
				trn_nets[i]->CopyNetParamsFrom(tst_net);
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_create(&(threads[i]), &pta, (void*(*)(void*))do_slave, (void*)(&(thread_data[i])));
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_join(threads[i], NULL);
			}

			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			tst_net->ClearNetParamsDiff();
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			for(int i = 0; i < gpus.size(); i++) {
				tst_net->AddNetParamsDiffFrom(trn_nets[i]);
			}
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			tst_net->UpdateNet(-1.0f / (gpus.size()));
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			// update learning rate
			if((epoch != 0) && (epoch % lr_stepsize == 0))
			{
				lr_rate /= 10;
				tst_net->SaveNetParams(epoch);
			}
		}
	}

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(main_gpu_id);
	batch_samples_slices.clear();
	batch_labels_slices.clear();

	delete tst_net;

	delete trn_data_param; trn_data_param = NULL;
	delete trn_data_layer; trn_data_layer = NULL;
	delete tst_data_param; tst_data_param = NULL;
	delete tst_data_layer; tst_data_layer = NULL;

	if(num_gpus >= gpus.size()) {
		LOG(INFO) << ("disable P2P: ");
		DisableP2P(gpus);
	}

	pthread_barrier_destroy(&barr);
	ret_count = pthread_attr_destroy(&pta);
	free(threads); threads = NULL;
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}

int main_test_conv_wigh_group_seems_ok(int argc, char **argv) {

	cudaStream_t curand_stream;
	curandRngType_t curand_rngtype;
	curandGenerator_t curand_generator;
	cublasHandle_t cublas_handle;
	CUDA_CHECK( cudaStreamCreate(&curand_stream) );
	curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
	CURAND_CHECK( curandCreateGenerator(&curand_generator, curand_rngtype) );
	CURAND_CHECK( curandSetStream(curand_generator, curand_stream) );
	CUBLAS_CHECK( cublasCreate(&cublas_handle) );

	Blob_t *batch_samples = new Blob_t(16, 3, 227, 227);
	Blob_t *batch_labels = new Blob_t(16, 1, 1, 1);
	batch_samples->allocate_gpu_data();
	batch_samples->allocate_gpu_diff();
	batch_labels->allocate_gpu_data();
	batch_labels->allocate_cpu_data();

	CURAND_CHECK( curandGenerateNormal(curand_generator, batch_samples->data_gpu, batch_samples->count(), (float)0.0f, (float)0.1f) );
	for(int i=0; i<batch_labels->count(); i++) {
		batch_labels->data_cpu[i] = i%10;
	}
	batch_labels->data_to_gpu();

	printf("conv1 setup.\n");
	Blob_t *conv1_top = new Blob_t();
	ConvolutionParameter_t *conv1_params = new ConvolutionParameter_t();
	conv1_params->filter_N = 3;
	conv1_params->filter_C = 96;
	conv1_params->filter_H = 11;
	conv1_params->filter_W = 11;
	conv1_params->pad_h = 0;
	conv1_params->pad_w = 0;
	conv1_params->stride_h = 4;
	conv1_params->stride_w = 4;
	conv1_params->upscale_h = 1;
	conv1_params->upscale_w = 1;
	conv1_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	ConvolutionLayer_t *conv1 = new ConvolutionLayer_t(conv1_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->filtersBlob->data_gpu, conv1->filtersBlob->count(), (float)0.0f, (float)0.0001f) );
	gpu_set(conv1->biasBlob->count(), 0, conv1->biasBlob->data_gpu);
	conv1->Setup(batch_samples, conv1_top);

	printf("mp1 setup.\n");
	Blob_t *mp1_top = new Blob_t();
	PoolingParameter_t *mp1_params = new PoolingParameter_t();
	mp1_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	mp1_params->poolsize_h = 3;
	mp1_params->poolsize_w = 3;
	mp1_params->pad_h = 0;
	mp1_params->pad_w = 0;
	mp1_params->stride_h = 2;
	mp1_params->stride_w = 2;
	PoolingLayer_t *mp1 = new PoolingLayer_t(mp1_params);
	mp1->Setup(conv1_top, mp1_top);

	printf("conv2g setup.\n");
	Blob_t *conv2g_top = new Blob_t();
	ConvolutionWithGroupParameter_t *conv2g_params = new ConvolutionWithGroupParameter_t();
	conv2g_params->group = 2;
	conv2g_params->filter_N = 96;
	conv2g_params->filter_C = 256;
	conv2g_params->filter_H = 5;
	conv2g_params->filter_W = 5;
	conv2g_params->pad_h = 2;
	conv2g_params->pad_w = 2;
	conv2g_params->stride_h = 1;
	conv2g_params->stride_w = 1;
	conv2g_params->upscale_h = 1;
	conv2g_params->upscale_w = 1;
	conv2g_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv2g_params->cudnn_conv_fwd_preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	ConvolutionWithGroupLayer_t *conv2g = new ConvolutionWithGroupLayer_t(conv2g_params);
	printf("init\n");
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv2g->filtersBlob->data_gpu, conv2g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv2g->biasBlob->count(), 0, conv2g->biasBlob->data_gpu);
	conv2g->Setup(mp1_top, conv2g_top);

	printf("mp2 setup.\n");
	Blob_t *mp2_top = new Blob_t();
	PoolingParameter_t *mp2_params = new PoolingParameter_t();
	mp2_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	mp2_params->poolsize_h = 3;
	mp2_params->poolsize_w = 3;
	mp2_params->pad_h = 0;
	mp2_params->pad_w = 0;
	mp2_params->stride_h = 2;
	mp2_params->stride_w = 2;
	PoolingLayer_t *mp2 = new PoolingLayer_t(mp1_params);
	mp2->Setup(conv2g_top, mp2_top);

	conv1->Forward(batch_samples, conv1_top);
	printf("conv1: (%d, %d, %d, %d)\n", conv1_top->N, conv1_top->C, conv1_top->H, conv1_top->W);

	mp1->Forward(conv1_top, mp1_top);
	printf("mp1: (%d, %d, %d, %d)\n", mp1_top->N, mp1_top->C, mp1_top->H, mp1_top->W);

	conv2g->Forward(mp1_top, conv2g_top);
	printf("conv2g_top: (%d, %d, %d, %d)\n", conv2g_top->N, conv2g_top->C, conv2g_top->H, conv2g_top->W);

	mp2->Forward(conv2g_top, mp2_top);
	printf("mp2: (%d, %d, %d, %d)\n", mp2_top->N, mp2_top->C, mp2_top->H, mp2_top->W);

	mp2->Backward(mp2_top, conv2g_top);

	conv2g->Backward(conv2g_top, mp1_top);

	mp1->Backward(mp1_top, conv1_top);

	conv1->Backward(conv1_top, batch_samples);


	delete conv1;
	delete conv1_top;
	delete conv1_params;
	delete mp1;
	delete mp1_top;
	delete mp1_params;
	delete conv2g;
	delete conv2g_top;
	delete conv2g_params;
	delete mp2;
	delete mp2_top;
	delete mp2_params;

	CURAND_CHECK( curandDestroyGenerator(curand_generator) );
	CUDA_CHECK( cudaStreamDestroy(curand_stream) );
	CUBLAS_CHECK( cublasDestroy(cublas_handle) );
	return 0;
}


// test alex net
int main_alexnet_to_be_tested(int argc, char **argv) {
	if(argc != 13) {
		LOG(FATAL) << "Usage: <filename> main_gpu_id trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n";
		return -1;
	}
	int main_gpu_id = atoi(argv[1]);
	string trn_db_filename = string(argv[2]);
	string tst_db_filename = string(argv[3]);
	string mean_file = string(argv[4]);
	float lr_rate = atof(argv[5]);
	int lr_stepsize = atoi(argv[6]);
	float momentum = atof(argv[7]);
	float weight_decay = atof(argv[8]);
	int trn_batch_size = atoi(argv[9]);
	int tst_batch_size = atoi(argv[10]);
	int max_epoch_num = atoi(argv[11]);
	string gpu_ids_str = string(argv[12]);

	cudaSetDevice(main_gpu_id);
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 227;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 227;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	AlexNetwork_t *trn_net = new AlexNetwork_t("alexnet_trn", main_gpu_id);
	trn_net->BuildNet(trn_batch_size, "");

	AlexNetwork_t *tst_net = new AlexNetwork_t("alexnet_tst", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");

	int num_tst_iters = 80000;//floor(10000 / tst_batch_size);
	int num_trn_iters = 12500;//floor(50000 / trn_batch_size);
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		tst_net->CopyNetParamsFrom(trn_net);
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(tst_net->batch_samples, tst_net->batch_labels);
			tst_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;
		LOG(INFO) << "epoch[" << epoch << "]: tst_loss=" << tst_loss << ", tst_acc= " << tst_acc;

		// training net
		float trn_loss = 0.0f, trn_loss_batch = 0.0f;
		float trn_acc  = 0.0f, trn_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network(trn_net->batch_samples, trn_net->batch_labels);
			trn_net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
			trn_loss += trn_loss_batch;
			trn_acc  += trn_acc_batch;
			trn_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);
			trn_net->UpdateNet();
		}
		trn_loss /= num_trn_iters;
		trn_acc  /= num_trn_iters;

		// update learning rate
		if((epoch != 0) && (epoch % lr_stepsize == 0))
		{
			lr_rate /= 10;
			trn_net->SaveNetParams(epoch);
		}
		LOG(INFO) << "epoch[" << epoch << "]: trn_loss=" << trn_loss << ", trn_acc= " << trn_acc;
	}

	delete trn_net;
	delete tst_net;

	delete trn_data_layer;
	delete tst_data_layer;
	delete trn_data_param;
	delete tst_data_param;

	cudaDeviceReset();
	return 0;
}
