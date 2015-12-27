#include "common_layer.hpp"

#include <algorithm>
using namespace std;

void ActivationLayer_t::Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem) {
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

	if(is_allocate_top_mem) {
		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}
}

void ActivationLayer_t::Forward(const Blob_t *bottom, Blob_t *top) {
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

void ActivationLayer_t::Backward(const Blob_t *top, Blob_t *bottom) {
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


void PoolingLayer_t::Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem) {
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

	if(is_allocate_top_mem) {
		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}
}

void PoolingLayer_t::Forward(const Blob_t *bottom, Blob_t *top) {
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

void PoolingLayer_t::Backward(const Blob_t *top, Blob_t *bottom) {
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

FullyConnectedLayer_t::FullyConnectedLayer_t(const FullyConnectedParameter_t *fc_params_) {
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

FullyConnectedLayer_t::~FullyConnectedLayer_t() {
	delete filtersBlob; filtersBlob = NULL;
	delete biasBlob; biasBlob = NULL;
	delete bias_multiplier; bias_multiplier = NULL;

	CUBLAS_CHECK( cublasDestroy(cublashandle) );
}

void FullyConnectedLayer_t::Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem) {
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

	if(is_allocate_top_mem) {
		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}

}

void FullyConnectedLayer_t::Forward(const Blob_t *bottom, Blob_t *top) {
	gpu_gemm(cublashandle, CblasNoTrans, CblasTrans, M_, N_, K_, (float)1.,
			bottom->data_gpu, filtersBlob->data_gpu, (float)0., top->data_gpu);
	gpu_gemm(cublashandle, CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
			bias_multiplier->data_gpu, biasBlob->data_gpu, (float)1., top->data_gpu);

}

void FullyConnectedLayer_t::Backward(const Blob_t *top, Blob_t *bottom) {
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



void SoftmaxLayer_t::Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem) {
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

	if (is_allocate_top_mem) {
		top->allocate_gpu_data();
		top->allocate_gpu_diff();
	}
}

void SoftmaxLayer_t::Forward(const Blob_t *bottom, Blob_t *top) {
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

void SoftmaxLayer_t::Backward(const Blob_t *top, Blob_t *bottom) {
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

void ArgMaxLayer_t::Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem) {
	top->N = bottom->N;
	top->C = 2;
	top->H = argmax_params->top_k;
	top->W = 1;

	if(is_allocate_top_mem) {
		top->allocate_cpu_data();
	}
}

void ArgMaxLayer_t::Forward_cpu(Blob_t *bottom, Blob_t *top) {

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




void AccuracyLayer_t::Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem) {
	top->N = 1;
	top->C = 1;
	top->H = 1;
	top->W = 1;

	if(is_allocate_top_mem) {
		top->allocate_cpu_data();
	}
}

void AccuracyLayer_t::Forward_cpu(Blob_t *bottom, Blob_t *label, Blob_t *top) {

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
