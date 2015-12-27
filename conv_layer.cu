

#include "conv_layer.hpp"

ConvolutionLayer_t::ConvolutionLayer_t(const ConvolutionParameter_t *conv_params_)
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

ConvolutionLayer_t::~ConvolutionLayer_t()
{
	delete filtersBlob; filtersBlob = NULL;
	delete biasBlob; biasBlob = NULL;

	CUDNN_CHECK( cudnnDestroyConvolutionDescriptor(convDesc) );
	CUDNN_CHECK( cudnnDestroyFilterDescriptor(filterDesc) );
	CUDNN_CHECK( cudnnDestroyTensorDescriptor(biasTensorDesc) );
}


void ConvolutionLayer_t::Setup(const Blob_t *bottom, Blob_t *top)
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

void ConvolutionLayer_t::Forward(const Blob_t *bottom, Blob_t *top)
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

void ConvolutionLayer_t::Backward(const Blob_t *top, Blob_t *bottom)
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


__global__ void sync_conv_groups() { }

ConvolutionWithGroupLayer_t::ConvolutionWithGroupLayer_t(const ConvolutionWithGroupParameter_t *convwithgroup_params_)
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

ConvolutionWithGroupLayer_t::~ConvolutionWithGroupLayer_t()
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


void ConvolutionWithGroupLayer_t::Setup(const Blob_t *bottom, Blob_t *top)
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

void ConvolutionWithGroupLayer_t::Forward(const Blob_t *bottom, Blob_t *top)
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

void ConvolutionWithGroupLayer_t::Backward(const Blob_t *top, Blob_t *bottom)
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

