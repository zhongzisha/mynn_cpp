/*
 * loss_layer.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef LOSS_LAYER_HPP_
#define LOSS_LAYER_HPP_

#include "common.hpp"
#include "blob.hpp"
#include "common_layer.hpp"

__global__ void SoftmaxLossForwardGPU(const int nthreads,
		const float* prob_data, const float* label, float* loss,
		const int num, const int dim, const int spatial_dim,
		const bool has_ignore_label_, const int ignore_label_,
		float* counts);

__global__ void SoftmaxLossBackwardGPU(const int nthreads, const float* top,
		const float* label, float* bottom_diff, const int num, const int dim,
		const int spatial_dim, const bool has_ignore_label_,
		const int ignore_label_, float* counts);

class SoftmaxWithLossParameter_t
{
public:
	cudnnSoftmaxAlgorithm_t cudnn_softmax_algo;
	cudnnSoftmaxMode_t cudnn_softmax_mode;
	bool has_ignore_label;
	int ignore_label;
	bool normalize;
};

class SoftmaxWithLossLayer_t : public Layer_t
{
public:
	cublasHandle_t cublashandle;
	SoftmaxWithLossParameter_t *cudnn_softmaxwithloss_params;
	Blob_t *prob_;
	/// Whether to ignore instances with a certain label.
	bool has_ignore_label_;
	/// The label indicating that an instance should be ignored.
	int ignore_label_;
	/// Whether to normalize the loss by the total number of values present
	/// (otherwise just by the batch size).
	bool normalize_;

	SoftmaxWithLossLayer_t(const SoftmaxWithLossParameter_t *cudnn_softmaxwithloss_params_) {
		cudnn_softmaxwithloss_params = const_cast<SoftmaxWithLossParameter_t *>(cudnn_softmaxwithloss_params_);
		cublashandle = NULL;
		CUBLAS_CHECK( cublasCreate(&cublashandle) );
		prob_ = NULL;
		has_ignore_label_ = false;
		ignore_label_ = -1;
		normalize_ = false;
	}

	~SoftmaxWithLossLayer_t() {
		CUBLAS_CHECK( cublasDestroy(cublashandle) );
	}

	void Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem = true);

	void Forward(const Blob_t *bottom, const Blob_t *label, Blob_t *top, float *loss);

	void Backward(const Blob_t *top, const Blob_t *label, Blob_t *bottom);
};

class MultinomialLogisticLossParameter_t
{
public:
	std::vector<int> ignore_labels;
	bool normalize;
};

class MultinomialLogisticLossLayer_t
{
public:
	MultinomialLogisticLossParameter_t *mlr_params;


	MultinomialLogisticLossLayer_t(const MultinomialLogisticLossParameter_t *mlr_params_) {
		mlr_params = const_cast<MultinomialLogisticLossParameter_t *>(mlr_params_);
	}

	~MultinomialLogisticLossLayer_t() {

	}

	void Setup(const Blob_t *bottom, Blob_t *top, bool is_allocate_top_mem = true);

	void Forward(const Blob_t *bottom, const Blob_t *label, Blob_t *top);

	void Backward(const Blob_t *top, const Blob_t *label, Blob_t *bottom);
};



#endif /* LOSS_LAYER_HPP_ */
