/*
 * blob.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef BLOB_HPP_
#define BLOB_HPP_

#include "common.hpp"

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

	~Blob_t() {
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

	void print_gpu_data();

	void print_gpu_data(int howmany);

	void print_cpu_data(int howmany) ;

	void save_cpu_data_and_diff_to_mat(const char *fname, bool is_save_diff = false);

	/*
	 *  data allocate
	 */
	void allocate_gpu_data();

	void allocate_gpu_diff();

	void allocate_cpu_data();

	void allocate_cpu_diff();

	/*
	 * data copy
	 */
	void data_to_gpu();

	void diff_to_gpu();

	void data_to_cpu();

	void diff_to_cpu();
};

void CopyBlobData_gpu(const Blob_t *src, int src_gpu_id, Blob_t *dst, int dst_gpu_id);

void AddBlobDiff_gpu(const Blob_t *src, int src_gpu_id, Blob_t *dst, int dst_gpu_id);



#endif /* BLOB_HPP_ */
