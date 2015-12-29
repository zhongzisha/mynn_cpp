
#include "blob.hpp"

void Blob_t::print_gpu_data() {
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

void Blob_t::print_gpu_data(int howmany) {
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

void Blob_t::print_cpu_data(int howmany) {
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

void Blob_t::save_cpu_data_and_diff_to_mat(const char *fname, bool is_save_diff)
{
	data_to_cpu();

	// mat_t *matfp = Mat_Create(fname, 0);
	mat_t *matfp = Mat_CreateVer(fname, 0, MAT_FT_MAT73);
	size_t dims[4];
	dims[0] = W;
	dims[1] = H;
	dims[2] = C;
	dims[3] = N;
	matvar_t *matvar;
	// save data
	matvar = Mat_VarCreate("data", MAT_C_SINGLE, MAT_T_SINGLE, 4, dims, data_cpu, 0);
	if(matvar == NULL)
		LOG(FATAL) << "Error creating 'data' variable";
	if(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0)
		LOG(FATAL) << "Error saving array 'data' into MAT file " << fname;
	Mat_VarFree(matvar);

	// save diff
	if(is_save_diff) {
		diff_to_cpu();

		matvar_t *matvar2;
		matvar2 = Mat_VarCreate("diff", MAT_C_SINGLE, MAT_T_SINGLE, 4, dims, diff_cpu, 0);
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
void Blob_t::allocate_gpu_data()
{
	int count = N * C * H * W;
	if(data_gpu != NULL)
		CUDA_CHECK( cudaFree(data_gpu) );
	CUDA_CHECK( cudaMalloc((void**)&data_gpu, count * sizeof(float)) );
	CUDA_CHECK( cudaMemset(data_gpu, 0, count * sizeof(float)) );
}

void Blob_t::allocate_gpu_diff()
{
	int count = N * C * H * W;
	if(diff_gpu != NULL)
		CUDA_CHECK( cudaFree(diff_gpu) );
	CUDA_CHECK( cudaMalloc((void**)&diff_gpu, count * sizeof(float)) );
	CUDA_CHECK( cudaMemset(diff_gpu, 0, count * sizeof(float)) );
}

void Blob_t::allocate_cpu_data()
{
	int count = N * C * H * W;
	if(data_cpu != NULL)
		FreeHost(data_cpu);
	MallocHost((void**)&data_cpu, count * sizeof(float));
	cpu_set(count, 0, data_cpu);
}

void Blob_t::allocate_cpu_diff()
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
void Blob_t::data_to_gpu()
{
	int count = N * C * H * W;
	if(data_gpu == NULL)
		CUDA_CHECK( cudaMalloc((void**)&data_gpu, count * sizeof(float)) );
	if(data_cpu != NULL)
		CUDA_CHECK( cudaMemcpy(data_gpu, data_cpu, count * sizeof(float), cudaMemcpyHostToDevice) );
}

void Blob_t::diff_to_gpu()
{
	int count = N * C * H * W;
	if(diff_gpu == NULL)
		CUDA_CHECK( cudaMalloc((void**)&diff_gpu, count * sizeof(float)) );
	if(diff_cpu != NULL)
		CUDA_CHECK( cudaMemcpy(diff_gpu, diff_cpu, count * sizeof(float), cudaMemcpyHostToDevice) );
}

void Blob_t::data_to_cpu()
{
	int count = N * C * H * W;
	if(data_cpu == NULL)
		MallocHost((void**)&data_cpu, count * sizeof(float));
	if(data_gpu != NULL)
		CUDA_CHECK( cudaMemcpy(data_cpu, data_gpu, count * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Blob_t::diff_to_cpu()
{
	int count = N * C * H * W;
	if(diff_cpu == NULL)
		MallocHost((void**)&diff_cpu, count * sizeof(float));
	if(diff_gpu != NULL)
		CUDA_CHECK( cudaMemcpy(diff_cpu, diff_gpu, count * sizeof(float), cudaMemcpyDeviceToHost) );
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
