
#include "common.hpp"

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

void cpu_add(const int N, const float *a, const float *b, float *y) {
#pragma omp parallel
	for(int i = 0; i < N; i++) {
		y[i] = a[i] + b[i];
	}
}

__global__ void add_kernel(const int n, const float* a,
		const float* b, float* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = a[index] + b[index];
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

void cpu_fill(float *ptr, int count, float value) {
	memset(ptr, value, count * sizeof(float));
}

void gpu_fill(curandGenerator_t curand_generator, float *ptr, int count, float mu, float std) {
	if(std == 0.0f) {
		CUDA_CHECK( cudaMemset(ptr, mu, count * sizeof(float)) );
	} else {
		CURAND_CHECK( curandGenerateNormal(curand_generator, ptr, count, mu, std) );
	}
}
