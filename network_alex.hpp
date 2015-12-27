/*
 * network_alex.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef NETWORK_ALEX_HPP_
#define NETWORK_ALEX_HPP_

#include "common.hpp"
#include "blob.hpp"
#include "common_layer.hpp"
#include "data_layer.hpp"
#include "conv_layer.hpp"
#include "loss_layer.hpp"


class AlexNetwork_t
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

	PoolingParameter_t *pool1_params;
	PoolingLayer_t *pool1;
	Blob_t *pool1_top;

	ConvolutionWithGroupParameter_t *conv2g_params;
	ConvolutionWithGroupLayer_t *conv2g;
	Blob_t *conv2g_top;

	ActivationParameter_t *relu2_params;
	ActivationLayer_t *relu2;
	Blob_t *relu2_top;

	PoolingParameter_t *pool2_params;
	PoolingLayer_t *pool2;
	Blob_t *pool2_top;

	ConvolutionParameter_t *conv3_params;
	ConvolutionLayer_t *conv3;
	Blob_t *conv3_top;

	ActivationParameter_t *relu3_params;
	ActivationLayer_t *relu3;
	Blob_t *relu3_top;

//	PoolingParameter_t *mp3_params;
//	PoolingLayer_t *mp3;
//	Blob_t *mp3_top;

	ConvolutionWithGroupParameter_t *conv4g_params;
	ConvolutionWithGroupLayer_t *conv4g;
	Blob_t *conv4g_top;

	ActivationParameter_t *relu4_params;
	ActivationLayer_t *relu4;
	Blob_t *relu4_top;

	ConvolutionWithGroupParameter_t *conv5g_params;
	ConvolutionWithGroupLayer_t *conv5g;
	Blob_t *conv5g_top;

	ActivationParameter_t *relu5_params;
	ActivationLayer_t *relu5;
	Blob_t *relu5_top;

	PoolingParameter_t *pool5_params;
	PoolingLayer_t *pool5;
	Blob_t *pool5_top;

	FullyConnectedParameter_t *fc6_params;
	FullyConnectedLayer_t *fc6;
	Blob_t *fc6_top;

	ActivationParameter_t *relu6_params;
	ActivationLayer_t *relu6;
	Blob_t *relu6_top;

	FullyConnectedParameter_t *fc7_params;
	FullyConnectedLayer_t *fc7;
	Blob_t *fc7_top;

	ActivationParameter_t *relu7_params;
	ActivationLayer_t *relu7;
	Blob_t *relu7_top;

	FullyConnectedParameter_t *fc8_params;
	FullyConnectedLayer_t *fc8;
	Blob_t *fc8_top;


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
	Blob_t *conv2g_filtersBlob_old;
	Blob_t *conv2g_biasBlob_old;
	Blob_t *conv3_filtersBlob_old;
	Blob_t *conv3_biasBlob_old;
	Blob_t *conv4g_filtersBlob_old;
	Blob_t *conv4g_biasBlob_old;
	Blob_t *conv5g_filtersBlob_old;
	Blob_t *conv5g_biasBlob_old;
	Blob_t *fc6_filtersBlob_old;
	Blob_t *fc6_biasBlob_old;
	Blob_t *fc7_filtersBlob_old;
	Blob_t *fc7_biasBlob_old;
	Blob_t *fc8_filtersBlob_old;
	Blob_t *fc8_biasBlob_old;


	AlexNetwork_t(string net_name_, int gpu_id_ = 0);

	~AlexNetwork_t();

	void DestroyNet();

	void BuildNet(int batch_size_, const string &net_params_file = "");

	void Forward(float *loss, float *accuracy);

	void Backward();

	void ForwardBackward(float *loss, float *accuracy);

	void ComputeUpdateValueSingle(Blob_t *param_gradient_blob, Blob_t *param_blob_old,
			float lr_rate, float momentum, float weight_decay) ;

	void ComputeUpdateValue(float lr_rate, float momentum, float weight_decay);

	void UpdateNet(float scale = -1.0f);

	void SaveNetParams(int epoch);

	void CopyNetParamsFrom(const AlexNetwork_t *other);

	void AddNetParamsDiffFrom(const AlexNetwork_t *other);

	void ClearNetParamsDiff();

};



#endif /* NETWORK_ALEX_HPP_ */
