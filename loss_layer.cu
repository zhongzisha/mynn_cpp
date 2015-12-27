
#include "loss_layer.hpp"

#include <cfloat> // for FLT_MIN

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

void SoftmaxWithLossLayer_t::Setup(const Blob_t *bottom, Blob_t *top) {
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

void SoftmaxWithLossLayer_t::Forward(const Blob_t *bottom, const Blob_t *label, Blob_t *top, float *loss) {
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

void SoftmaxWithLossLayer_t::Backward(const Blob_t *top, const Blob_t *label, Blob_t *bottom) {
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


void MultinomialLogisticLossLayer_t::Setup(const Blob_t *bottom, Blob_t *top) {
	top->N = 1;
	top->C = 1;
	top->H = 1;
	top->W = 1;
	top->allocate_cpu_data();
	top->allocate_cpu_diff();
	top->data_cpu[0] = 1.0f;
}

void MultinomialLogisticLossLayer_t::Forward(const Blob_t *bottom, const Blob_t *label, Blob_t *top) {
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

void MultinomialLogisticLossLayer_t::Backward(const Blob_t *top, const Blob_t *label, Blob_t *bottom) {
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
