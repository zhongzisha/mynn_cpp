/*
 * common.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <glog/logging.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

extern "C" {
#include <cblas.h>
}

#include "matio.h"

#include <vector>
using namespace std;

const char* curandGetErrorString(curandStatus_t error);

const char* cublasGetErrorString(cublasStatus_t error);

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

void EnableP2P(vector<int> gpus);

void DisableP2P(vector<int> gpus);

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


void cpu_add(const int N, const float *a, const float *b, float *y);
void gpu_add(const int N, const float* a, const float* b, float* y);
void cpu_set(const int N, const float alpha, float *Y);
void gpu_set(const int N, const float alpha, float* Y);
void gpu_copy(const int N, const float *X, float *Y);
void gpu_asum(cublasHandle_t cublashandle, const int n, const float* x, float* y);
void gpu_scal(cublasHandle_t cublashandle, const int N, const float alpha, float *X);
void gpu_axpy(cublasHandle_t cublashandle, const int N, const float alpha,
		const float* X, float* Y);
void gpu_axpby(cublasHandle_t cublashandle, const int N, const float alpha,
		const float* X, const float beta, float* Y);
void gpu_gemv(cublasHandle_t cublashandle,
		const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const float alpha, const float* A, const float* x,
		const float beta, float* y);
void gpu_gemm(cublasHandle_t cublashandle,
		const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta,
		float* C);
void cpu_fill(float *ptr, int count, float value);
void gpu_fill(curandGenerator_t curand_generator, float *ptr, int count, float mu, float std = 0.0f);



#endif /* COMMON_HPP_ */
