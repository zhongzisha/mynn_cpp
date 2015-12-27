/*
 * common_layer.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef COMMON_LAYER_HPP_
#define COMMON_LAYER_HPP_

#include "common.hpp"
#include "blob.hpp"

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

	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward(const Blob_t *bottom, Blob_t *top);

	void Backward(const Blob_t *top, Blob_t *bottom);
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

	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward(const Blob_t *bottom, Blob_t *top);

	void Backward(const Blob_t *top, Blob_t *bottom);
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
	FullyConnectedLayer_t(const FullyConnectedParameter_t *fc_params_);

	~FullyConnectedLayer_t();

	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward(const Blob_t *bottom, Blob_t *top);

	void Backward(const Blob_t *top, Blob_t *bottom);
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

	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward(const Blob_t *bottom, Blob_t *top);

	void Backward(const Blob_t *top, Blob_t *bottom);
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

	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward_cpu(Blob_t *bottom, Blob_t *top);
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

	void Setup(const Blob_t *bottom, Blob_t *top);

	void Forward_cpu(Blob_t *bottom, Blob_t *label, Blob_t *top);
};



#endif /* COMMON_LAYER_HPP_ */
