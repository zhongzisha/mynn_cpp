/*
 * cifar10_network.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef NETWORK_CIFAR10_HPP_
#define NETWORK_CIFAR10_HPP_

#include "common.hpp"
#include "blob.hpp"
#include "common_layer.hpp"
#include "data_layer.hpp"
#include "conv_layer.hpp"
#include "loss_layer.hpp"


class Cifar10Network_t
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


	Cifar10Network_t(string net_name_, int gpu_id_ = 0);

	~Cifar10Network_t();

	void DestroyNet();

	void BuildNet(int batch_size_, const string &net_params_file = "");

	void Forward(float *loss, float *accuracy);

	void Backward();

	void ForwardBackward(float *loss, float *accuracy);

	void ComputeUpdateValueSingle(Blob_t *param_gradient_blob, Blob_t *param_blob_old, float lr_rate, float momentum, float weight_decay);

	void ComputeUpdateValue(float lr_rate, float momentum, float weight_decay);

	void UpdateNet(float scale = -1.0f);

	void SaveNetParams(int epoch);

	void CopyNetParamsFrom(const Cifar10Network_t *other);

	void AddNetParamsDiffFrom(const Cifar10Network_t *other);

	void ClearNetParamsDiff() ;

};


#endif /* CIFAR10_NETWORK_HPP_ */
