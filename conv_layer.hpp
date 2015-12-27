/*
 * conv_layer.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef CONV_LAYER_HPP_
#define CONV_LAYER_HPP_

#include "common.hpp"
#include "blob.hpp"
#include "common_layer.hpp"


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

class ConvolutionLayer_t : public Layer_t
{
public:
	Blob_t *filtersBlob;
	Blob_t *biasBlob;

	cudnnFilterDescriptor_t filterDesc;
	cudnnTensorDescriptor_t biasTensorDesc;
	cudnnConvolutionDescriptor_t convDesc;
	ConvolutionParameter_t *conv_params;


	ConvolutionLayer_t(const ConvolutionParameter_t *conv_params_);

	~ConvolutionLayer_t();


	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward(const Blob_t *bottom, Blob_t *top);

	void Backward(const Blob_t *top, Blob_t *bottom);
};


__global__ void sync_conv_groups();

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


	ConvolutionWithGroupLayer_t(const ConvolutionWithGroupParameter_t *convwithgroup_params_);

	~ConvolutionWithGroupLayer_t();


	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward(const Blob_t *bottom, Blob_t *top);

	void Backward(const Blob_t *top, Blob_t *bottom);
};




#endif /* CONV_LAYER_HPP_ */
