#include <glog/logging.h>

// #include <mpi.h>

#include <omp.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <map>
#include <list>
#include <stack>
#include <deque>
using namespace std;

#include <cuda_runtime.h>
#include <driver_types.h>  // cuda driver types

#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <boost/filesystem.hpp>
using namespace boost;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"
#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
using namespace caffe;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include "matio.h"


template <typename Dtype> enum matio_types matio_type_map();
template <> enum matio_types matio_type_map<float>() { return MAT_T_SINGLE; }
template <> enum matio_types matio_type_map<double>() { return MAT_T_DOUBLE; }
template <> enum matio_types matio_type_map<int>() { return MAT_T_INT32; }
template <> enum matio_types matio_type_map<unsigned int>() { return MAT_T_UINT32; }

template <typename Dtype> enum matio_classes matio_class_map();
template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

void save_blob_to_mat(const char *fname, shared_ptr<Blob<float> > blob)
{
	// save results into matlab format
	mat_t *matfp = Mat_Create(fname, 0);
	//matfp = Mat_CreateVer(fname, 0, MAT_FT_MAT73);
	size_t dims[4];
	dims[0] = blob->width();
	dims[1] = blob->height();
	dims[2] = blob->channels();
	dims[3] = blob->num();
	matvar_t *matvar, *matvar2;
	// save data
	{
		matvar = Mat_VarCreate("data", matio_class_map<float>(), matio_type_map<float>(), 4, dims, blob->mutable_cpu_data(), 0);
		if(matvar == NULL)
			cout << "Error creating 'data' variable";
		matvar2 = Mat_VarCreate("diff", matio_class_map<float>(), matio_type_map<float>(), 4, dims, blob->mutable_cpu_diff(), 0);
		if(matvar2 == NULL)
			cout << "Error creating 'diff' variable";
		if(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0)
			cout << "Error saving array 'data' into MAT file " << fname;

		if(Mat_VarWrite(matfp, matvar2, MAT_COMPRESSION_NONE) != 0)
			cout << "Error saving array 'diff' into MAT file " << fname;

		Mat_VarFree(matvar);
		Mat_VarFree(matvar2);
	}
	Mat_Close(matfp);
}


class SimpleNet
{
public:
	SimpleNet(string name_)
{
		name = name_;
		gpu_id = 0;

		layer_c1 = NULL;
		layer_relu1 = NULL;
		layer_mp1 = NULL;

		layer_c2 = NULL;
		layer_relu2 = NULL;
		layer_mp2 = NULL;

		layer_c3 = NULL;
		layer_relu3 = NULL;
		layer_mp3 = NULL;

		layer_ip1 = NULL;
		layer_sml1 = NULL;

		label_blob = NULL;
		c1_bottom_blob = NULL;

		c1_top_blob = NULL;
		relu1_top_blob = NULL;
		mp1_top_blob = NULL;
		c2_top_blob = NULL;
		relu2_top_blob = NULL;
		mp2_top_blob = NULL;
		c3_top_blob = NULL;
		relu3_top_blob = NULL;
		mp3_top_blob = NULL;
		ip1_top_blob = NULL;
		sml1_top_blob = NULL;

		c1_weight_blob_old = NULL;
		c1_bias_blob_old = NULL;
		c2_weight_blob_old = NULL;
		c2_bias_blob_old = NULL;
		c3_weight_blob_old = NULL;
		c3_bias_blob_old = NULL;
		ip1_weight_blob_old = NULL;
		ip1_bias_blob_old = NULL;
}

	~SimpleNet()
	{
		delete layer_sml1;
		delete layer_ip1;

		delete layer_mp3;
		delete layer_relu3;
		delete layer_c3;

		delete layer_mp2;
		delete layer_relu2;
		delete layer_c2;

		delete layer_mp1;
		delete layer_relu1;
		delete layer_c1;

		delete label_blob;
		delete c1_bottom_blob;

		delete c1_top_blob;
		delete relu1_top_blob;
		delete mp1_top_blob;

		delete c2_top_blob;
		delete relu2_top_blob;
		delete mp2_top_blob;

		delete c3_top_blob;
		delete relu3_top_blob;
		delete mp3_top_blob;

		delete ip1_top_blob;
		delete sml1_top_blob;

		delete c1_weight_blob_old;
		delete c1_bias_blob_old;
		delete c2_weight_blob_old;
		delete c2_bias_blob_old;
		delete c3_weight_blob_old;
		delete c3_bias_blob_old;
		delete ip1_weight_blob_old;
		delete ip1_bias_blob_old;
	}

	string name;
	int gpu_id;

	ConvolutionLayer<float> *layer_c1;
	ReLULayer<float> *layer_relu1;
	PoolingLayer<float> *layer_mp1;

	ConvolutionLayer<float> *layer_c2;
	ReLULayer<float> *layer_relu2;
	PoolingLayer<float> *layer_mp2;

	ConvolutionLayer<float> *layer_c3;
	ReLULayer<float> *layer_relu3;
	PoolingLayer<float> *layer_mp3;

	InnerProductLayer<float> *layer_ip1;

	SoftmaxWithLossLayer<float> *layer_sml1;

	Blob<float> *label_blob;     // this is the label of the batch data
	Blob<float> *c1_bottom_blob; // this is the batch data

	Blob<float> *c1_top_blob;
	Blob<float> *relu1_top_blob;
	Blob<float> *mp1_top_blob;

	Blob<float> *c2_top_blob;
	Blob<float> *relu2_top_blob;
	Blob<float> *mp2_top_blob;

	Blob<float> *c3_top_blob;
	Blob<float> *relu3_top_blob;
	Blob<float> *mp3_top_blob;

	Blob<float> *ip1_top_blob;
	Blob<float> *sml1_top_blob;

	vector<Blob<float> *> c1_bottom_;
	vector<Blob<float> *> c1_top_;
	vector<Blob<float> *> relu1_top_;
	vector<Blob<float> *> mp1_top_;

	vector<Blob<float> *> c2_top_;
	vector<Blob<float> *> relu2_top_;
	vector<Blob<float> *> mp2_top_;

	vector<Blob<float> *> c3_top_;
	vector<Blob<float> *> relu3_top_;
	vector<Blob<float> *> mp3_top_;


	vector<Blob<float> *> ip1_top_;
	vector<Blob<float> *> sml1_bottom_;
	vector<Blob<float> *> sml1_top_;

	Blob<float> *c1_weight_blob_old;
	Blob<float> *c1_bias_blob_old;
	Blob<float> *c2_weight_blob_old;
	Blob<float> *c2_bias_blob_old;
	Blob<float> *c3_weight_blob_old;
	Blob<float> *c3_bias_blob_old;
	Blob<float> *ip1_weight_blob_old;
	Blob<float> *ip1_bias_blob_old;

	void BuildNet(int batch_size, int gpu_id_ = 0)
	{
		gpu_id = gpu_id_;

		cudaSetDevice(gpu_id);
		// initialize the blobs used by each layer
		label_blob = new Blob<float>(batch_size,1,1,1);
		c1_bottom_blob = new Blob<float>(batch_size, 3, 32, 32);
		c1_top_blob = new Blob<float>();
		relu1_top_blob = new Blob<float>();
		mp1_top_blob = new Blob<float>();

		c2_top_blob = new Blob<float>();
		relu2_top_blob = new Blob<float>();
		mp2_top_blob = new Blob<float>();

		c3_top_blob = new Blob<float>();
		relu3_top_blob = new Blob<float>();
		mp3_top_blob = new Blob<float>();

		ip1_top_blob = new Blob<float>();
		sml1_top_blob = new Blob<float>();

		c1_bottom_.push_back(c1_bottom_blob);
		c1_top_.push_back(c1_top_blob);
		relu1_top_.push_back(relu1_top_blob);
		mp1_top_.push_back(mp1_top_blob);

		c2_top_.push_back(c2_top_blob);
		relu2_top_.push_back(relu2_top_blob);
		mp2_top_.push_back(mp2_top_blob);

		c3_top_.push_back(c3_top_blob);
		relu3_top_.push_back(relu3_top_blob);
		mp3_top_.push_back(mp3_top_blob);

		ip1_top_.push_back(ip1_top_blob);

		sml1_bottom_.push_back(ip1_top_blob);
		sml1_bottom_.push_back(label_blob);
		sml1_top_.push_back(sml1_top_blob);

		// initialize the layers
		LayerParameter layer_param1;
		ConvolutionParameter *layer_c1_param = layer_param1.mutable_convolution_param();
		layer_c1_param->set_kernel_size(5);
		layer_c1_param->set_stride(1);
		layer_c1_param->set_num_output(32);
		layer_c1_param->set_pad(2);
		layer_c1_param->mutable_weight_filler()->set_type("gaussian");
		layer_c1_param->mutable_weight_filler()->set_mean(0.0f);
		layer_c1_param->mutable_weight_filler()->set_std(0.0001f);
		layer_c1_param->mutable_bias_filler()->set_type("constant");
		layer_c1_param->mutable_bias_filler()->set_value(0.f);
		layer_c1 = new ConvolutionLayer<float>(layer_param1);
		layer_c1->SetUp(c1_bottom_, c1_top_);

		LayerParameter layer_param2;
		ReLUParameter *layer_relu1_param = layer_param2.mutable_relu_param();
		layer_relu1 = new ReLULayer<float>(layer_param2);
		layer_relu1->SetUp(c1_top_, relu1_top_);

		LayerParameter layer_param3;
		PoolingParameter *layer_mp1_param = layer_param3.mutable_pooling_param();
		layer_mp1_param->set_kernel_h(2);
		layer_mp1_param->set_kernel_w(2);
		layer_mp1_param->set_stride_h(2);
		layer_mp1_param->set_stride_w(2);
		layer_mp1_param->set_pool(PoolingParameter_PoolMethod_MAX);
		layer_mp1 = new PoolingLayer<float>(layer_param3);
		layer_mp1->SetUp(relu1_top_, mp1_top_);

		LayerParameter layer_param4;
		ConvolutionParameter *layer_c2_param = layer_param4.mutable_convolution_param();
		layer_c2_param->set_kernel_size(5);
		layer_c2_param->set_stride(1);
		layer_c2_param->set_num_output(32);
		layer_c2_param->set_pad(2);
		layer_c2_param->mutable_weight_filler()->set_type("gaussian");
		layer_c2_param->mutable_weight_filler()->set_mean(0.0f);
		layer_c2_param->mutable_weight_filler()->set_std(0.01f);
		layer_c2_param->mutable_bias_filler()->set_type("constant");
		layer_c2_param->mutable_bias_filler()->set_value(0.f);
		layer_c2 = new ConvolutionLayer<float>(layer_param4);
		layer_c2->SetUp(mp1_top_, c2_top_);

		LayerParameter layer_param5;
		ReLUParameter *layer_relu2_param = layer_param5.mutable_relu_param();
		layer_relu2 = new ReLULayer<float>(layer_param5);
		layer_relu2->SetUp(c2_top_, relu2_top_);

		LayerParameter layer_param6;
		PoolingParameter *layer_mp2_param = layer_param6.mutable_pooling_param();
		layer_mp2_param->set_kernel_size(3);
		layer_mp2_param->set_stride(2);
		layer_mp2_param->set_pool(PoolingParameter_PoolMethod_AVE);
		layer_mp2 = new PoolingLayer<float>(layer_param6);
		layer_mp2->SetUp(relu2_top_, mp2_top_);

		LayerParameter layer_param41;
		ConvolutionParameter *layer_c3_param = layer_param41.mutable_convolution_param();
		layer_c3_param->set_kernel_size(5);
		layer_c3_param->set_stride(1);
		layer_c3_param->set_num_output(64);
		layer_c3_param->set_pad(2);
		layer_c3_param->mutable_weight_filler()->set_type("gaussian");
		layer_c3_param->mutable_weight_filler()->set_mean(0.0f);
		layer_c3_param->mutable_weight_filler()->set_std(0.01f);
		layer_c3_param->mutable_bias_filler()->set_type("constant");
		layer_c3_param->mutable_bias_filler()->set_value(0.f);
		layer_c3 = new ConvolutionLayer<float>(layer_param41);
		layer_c3->SetUp(mp2_top_, c3_top_);

		LayerParameter layer_param51;
		ReLUParameter *layer_relu3_param = layer_param51.mutable_relu_param();
		layer_relu3 = new ReLULayer<float>(layer_param51);
		layer_relu3->SetUp(c3_top_, relu3_top_);

		LayerParameter layer_param61;
		PoolingParameter *layer_mp3_param = layer_param61.mutable_pooling_param();
		layer_mp3_param->set_kernel_size(3);
		layer_mp3_param->set_stride(2);
		layer_mp3_param->set_pool(PoolingParameter_PoolMethod_AVE);
		layer_mp3 = new PoolingLayer<float>(layer_param61);
		layer_mp3->SetUp(relu3_top_, mp3_top_);

		LayerParameter layer_param7;
		InnerProductParameter *layer_ip1_param = layer_param7.mutable_inner_product_param();
		layer_ip1_param->set_num_output(10);
		layer_ip1_param->mutable_weight_filler()->set_type("gaussian");
		layer_ip1_param->mutable_weight_filler()->set_mean(0.0f);
		layer_ip1_param->mutable_weight_filler()->set_std(0.01f);
		layer_ip1_param->mutable_bias_filler()->set_type("constant");
		layer_ip1_param->mutable_bias_filler()->set_value(0.f);
		layer_ip1 = new InnerProductLayer<float>(layer_param7);
		layer_ip1->SetUp(mp3_top_, ip1_top_);

		LayerParameter layer_param8;
		LossParameter *layer_sml1_param = layer_param8.mutable_loss_param();
		layer_sml1_param->set_normalize(true);
		layer_sml1 = new SoftmaxWithLossLayer<float>(layer_param8);
		layer_sml1->SetUp(sml1_bottom_, sml1_top_);

		// initialize the old params
		FillerParameter filler_param;
		filler_param.set_value(0.0f);
		ConstantFiller<float> filler(filler_param);

		c1_weight_blob_old = new Blob<float>();
		c1_weight_blob_old->ReshapeLike(*(layer_c1->blobs()[0]));
		filler.Fill(c1_weight_blob_old);

		c1_bias_blob_old = new Blob<float>();
		c1_bias_blob_old->ReshapeLike(*(layer_c1->blobs()[1]));
		filler.Fill(c1_bias_blob_old);

		c2_weight_blob_old = new Blob<float>();
		c2_weight_blob_old->ReshapeLike(*(layer_c2->blobs()[0]));
		filler.Fill(c2_weight_blob_old);

		c2_bias_blob_old = new Blob<float>();
		c2_bias_blob_old->ReshapeLike(*(layer_c2->blobs()[1]));
		filler.Fill(c2_bias_blob_old);

		c3_weight_blob_old = new Blob<float>();
		c3_weight_blob_old->ReshapeLike(*(layer_c3->blobs()[0]));
		filler.Fill(c3_weight_blob_old);

		c3_bias_blob_old = new Blob<float>();
		c3_bias_blob_old->ReshapeLike(*(layer_c3->blobs()[1]));
		filler.Fill(c3_bias_blob_old);

		ip1_weight_blob_old = new Blob<float>();
		ip1_weight_blob_old->ReshapeLike(*(layer_ip1->blobs()[0]));
		filler.Fill(ip1_weight_blob_old);

		ip1_bias_blob_old = new Blob<float>();
		ip1_bias_blob_old->ReshapeLike(*(layer_ip1->blobs()[1]));
		filler.Fill(ip1_bias_blob_old);

	}

	float Forward()
	{
		// before run forward, you should fill batch data into: c1_bottom_blob and label_blob
		cudaSetDevice(gpu_id);

		layer_c1->Forward(c1_bottom_, c1_top_);
		layer_relu1->Forward(c1_top_, relu1_top_);
		layer_mp1->Forward(relu1_top_, mp1_top_);

		layer_c2->Forward(mp1_top_, c2_top_);
		layer_relu2->Forward(c2_top_, relu2_top_);
		layer_mp2->Forward(relu2_top_, mp2_top_);

		layer_c3->Forward(mp2_top_, c3_top_);
		layer_relu3->Forward(c3_top_, relu3_top_);
		layer_mp3->Forward(relu3_top_, mp3_top_);

		layer_ip1->Forward(mp3_top_, ip1_top_);


		layer_sml1->Forward(sml1_bottom_, sml1_top_);


		float loss = sml1_top_blob->cpu_data()[0];

		return loss;
	}

	void Backward()
	{
		cudaSetDevice(gpu_id);

		vector<bool> propagate_down1(sml1_bottom_.size());
		propagate_down1[0] = true;
		propagate_down1[1] = false;
		layer_sml1->Backward(sml1_top_, propagate_down1, sml1_bottom_);

		vector<bool> propagate_down2(mp3_top_.size());
		propagate_down2[0] = true;
		layer_ip1->Backward(ip1_top_, propagate_down2, mp3_top_);

		vector<bool> propagate_down31(relu3_top_.size());
		propagate_down31[0] = true;
		layer_mp3->Backward(mp3_top_, propagate_down31, relu3_top_);

		vector<bool> propagate_down41(c3_top_.size());
		propagate_down41[0] = true;
		layer_relu3->Backward(relu3_top_, propagate_down41, c3_top_);

		vector<bool> propagate_down51(mp2_top_.size());
		propagate_down51[0] = true;
		layer_c3->Backward(c3_top_, propagate_down51, mp2_top_);

		vector<bool> propagate_down3(relu2_top_.size());
		propagate_down3[0] = true;
		layer_mp2->Backward(mp2_top_, propagate_down3, relu2_top_);

		vector<bool> propagate_down4(c2_top_.size());
		propagate_down4[0] = true;
		layer_relu2->Backward(relu2_top_, propagate_down4, c2_top_);

		vector<bool> propagate_down5(mp1_top_.size());
		propagate_down5[0] = true;
		layer_c2->Backward(c2_top_, propagate_down5, mp1_top_);

		vector<bool> propagate_down6(relu1_top_.size());
		propagate_down6[0] = true;
		layer_mp1->Backward(mp1_top_, propagate_down6, relu1_top_);

		vector<bool> propagate_down7(c1_top_.size());
		propagate_down7[0] = true;
		layer_relu1->Backward(relu1_top_, propagate_down7, c1_top_);

		vector<bool> propagate_down8(c1_bottom_.size());
		propagate_down8[0] = true;
		layer_c1->Backward(c1_top_, propagate_down8, c1_bottom_);
	}

	// float ForwardBackward(const vector<Blob<float>*>& bottom)
	float ForwardBackward()
	{
		float loss = Forward();
		Backward();
		return loss;
	}

	void CopyNetParamsFrom(const SimpleNet *net1)
	{
		for(int i = 0; i <= 1; i++)
		{
			caffe_copy(net1->layer_c1->blobs()[i]->count(),
					net1->layer_c1->blobs()[i]->gpu_data(),
					this->layer_c1->blobs()[i]->mutable_gpu_data());

			caffe_copy(net1->layer_c2->blobs()[i]->count(),
					net1->layer_c2->blobs()[i]->gpu_data(),
					this->layer_c2->blobs()[i]->mutable_gpu_data());

			caffe_copy(net1->layer_c3->blobs()[i]->count(),
					net1->layer_c3->blobs()[i]->gpu_data(),
					this->layer_c3->blobs()[i]->mutable_gpu_data());

			caffe_copy(net1->layer_ip1->blobs()[i]->count(),
					net1->layer_ip1->blobs()[i]->gpu_data(),
					this->layer_ip1->blobs()[i]->mutable_gpu_data());
		}
	}

	void CopyNetParamsAddDiffFrom(const SimpleNet *net1)
	{
		for(int i = 0; i <= 1; i++)
		{
			caffe_add(net1->layer_c1->blobs()[i]->count(),
					net1->layer_c1->blobs()[i]->gpu_diff(),
					this->layer_c1->blobs()[i]->gpu_diff(),
					this->layer_c1->blobs()[i]->mutable_gpu_diff());

			caffe_add(net1->layer_c2->blobs()[i]->count(),
					net1->layer_c2->blobs()[i]->gpu_diff(),
					this->layer_c2->blobs()[i]->gpu_diff(),
					this->layer_c2->blobs()[i]->mutable_gpu_diff());

			caffe_add(net1->layer_c3->blobs()[i]->count(),
					net1->layer_c3->blobs()[i]->gpu_diff(),
					this->layer_c3->blobs()[i]->gpu_diff(),
					this->layer_c3->blobs()[i]->mutable_gpu_diff());

			caffe_add(net1->layer_ip1->blobs()[i]->count(),
					net1->layer_ip1->blobs()[i]->gpu_diff(),
					this->layer_ip1->blobs()[i]->gpu_diff(),
					this->layer_ip1->blobs()[i]->mutable_gpu_diff());
		}
	}

	void ComputeUpdateValueSingle(shared_ptr<Blob<float> > param_gradient_blob, Blob<float> *param_blob_old, float lr_rate, float momentum, float weight_decay)
	{
		caffe_gpu_axpy(param_gradient_blob->count(),
				weight_decay,
				param_gradient_blob->gpu_data(),
				param_gradient_blob->mutable_gpu_diff());

		caffe_gpu_axpby(param_gradient_blob->count(), lr_rate,
				param_gradient_blob->gpu_diff(), momentum,
				param_blob_old->mutable_gpu_data());
		// copy
		caffe_copy(param_gradient_blob->count(),
				param_blob_old->gpu_data(),
				param_gradient_blob->mutable_gpu_diff());
	}
	void ComputeUpdateValue(float lr_rate, float momentum, float weight_decay)
	{
		cudaSetDevice(gpu_id);

		ComputeUpdateValueSingle(layer_c1->blobs()[0], c1_weight_blob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(layer_c1->blobs()[1], c1_bias_blob_old,   lr_rate, momentum, weight_decay);

		ComputeUpdateValueSingle(layer_c2->blobs()[0], c2_weight_blob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(layer_c2->blobs()[1], c2_bias_blob_old,   lr_rate, momentum, weight_decay);

		ComputeUpdateValueSingle(layer_c3->blobs()[0], c3_weight_blob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(layer_c3->blobs()[1], c3_bias_blob_old,   lr_rate, momentum, weight_decay);

		ComputeUpdateValueSingle(layer_ip1->blobs()[0], ip1_weight_blob_old, lr_rate, momentum, weight_decay);
		ComputeUpdateValueSingle(layer_ip1->blobs()[1], ip1_bias_blob_old, lr_rate, momentum, weight_decay);
	}

	void SetNetDiffToZero()
	{
		cudaSetDevice(gpu_id);

		caffe_gpu_set<float>(layer_c1->blobs()[0]->count(), 0, layer_c1->blobs()[0]->mutable_gpu_diff());
		caffe_gpu_set<float>(layer_c1->blobs()[1]->count(), 0, layer_c1->blobs()[1]->mutable_gpu_diff());

		caffe_gpu_set<float>(layer_c2->blobs()[0]->count(), 0, layer_c2->blobs()[0]->mutable_gpu_diff());
		caffe_gpu_set<float>(layer_c2->blobs()[1]->count(), 0, layer_c2->blobs()[1]->mutable_gpu_diff());

		caffe_gpu_set<float>(layer_c3->blobs()[0]->count(), 0, layer_c3->blobs()[0]->mutable_gpu_diff());
		caffe_gpu_set<float>(layer_c3->blobs()[1]->count(), 0, layer_c3->blobs()[1]->mutable_gpu_diff());

		caffe_gpu_set<float>(layer_ip1->blobs()[0]->count(), 0, layer_ip1->blobs()[0]->mutable_gpu_diff());
		caffe_gpu_set<float>(layer_ip1->blobs()[1]->count(), 0, layer_ip1->blobs()[1]->mutable_gpu_diff());
	}

	void UpdateNet()
	{
		cudaSetDevice(gpu_id);

		layer_c1->blobs()[0]->Update();
		layer_c1->blobs()[1]->Update();

		layer_c2->blobs()[0]->Update();
		layer_c2->blobs()[1]->Update();

		layer_c3->blobs()[0]->Update();
		layer_c3->blobs()[1]->Update();

		layer_ip1->blobs()[0]->Update();
		layer_ip1->blobs()[1]->Update();
	}

	void PrintNet()
	{
		printf("Net Input and Output: \n");
		printf("label_blob: 	(%d, %d, %d, %d)\n", label_blob->num(), label_blob->channels(), label_blob->height(), label_blob->width());
		printf("c1_bottom_blob: (%d, %d, %d, %d)\n", c1_bottom_blob->num(), c1_bottom_blob->channels(), c1_bottom_blob->height(), c1_bottom_blob->width());
		printf("c1_top_blob: 	(%d, %d, %d, %d)\n", c1_top_blob->num(), c1_top_blob->channels(), c1_top_blob->height(), c1_top_blob->width());
		printf("relu1_top_blob: (%d, %d, %d, %d)\n", relu1_top_blob->num(), relu1_top_blob->channels(), relu1_top_blob->height(), relu1_top_blob->width());
		printf("mp1_top_blob: 	(%d, %d, %d, %d)\n", mp1_top_blob->num(), mp1_top_blob->channels(), mp1_top_blob->height(), mp1_top_blob->width());
		printf("c2_top_blob: 	(%d, %d, %d, %d)\n", c2_top_blob->num(), c2_top_blob->channels(), c2_top_blob->height(), c2_top_blob->width());
		printf("relu2_top_blob: (%d, %d, %d, %d)\n", relu2_top_blob->num(), relu2_top_blob->channels(), relu2_top_blob->height(), relu2_top_blob->width());
		printf("mp2_top_blob: 	(%d, %d, %d, %d)\n", mp2_top_blob->num(), mp2_top_blob->channels(), mp2_top_blob->height(), mp2_top_blob->width());
		printf("c3_top_blob: 	(%d, %d, %d, %d)\n", c3_top_blob->num(), c3_top_blob->channels(), c3_top_blob->height(), c3_top_blob->width());
		printf("relu3_top_blob: (%d, %d, %d, %d)\n", relu3_top_blob->num(), relu3_top_blob->channels(), relu3_top_blob->height(), relu3_top_blob->width());
		printf("mp3_top_blob: 	(%d, %d, %d, %d)\n", mp3_top_blob->num(), mp3_top_blob->channels(), mp3_top_blob->height(), mp3_top_blob->width());
		printf("ip1_top_blob: 	(%d, %d, %d, %d)\n", ip1_top_blob->num(), ip1_top_blob->channels(), ip1_top_blob->height(), ip1_top_blob->width());
		printf("sml1_top_blob: 	(%d, %d, %d, %d)\n", sml1_top_blob->num(), sml1_top_blob->channels(), sml1_top_blob->height(), sml1_top_blob->width());
	}

	void PrintNetParamsSingle(const char *name, vector<shared_ptr<Blob<float> > > &params)
	{
		printf("%s: ", name);
		for(int i = 0; i < params.size(); i++)
		{
			printf("blob_%d (%d, %d, %d, %d), ", i, params[i]->num(), params[i]->channels(), params[i]->height(), params[i]->width());
		}
		printf("\n");
	}

	void PrintNetParams()
	{
		printf("Net Params: \n");
		PrintNetParamsSingle("c1", layer_c1->blobs());
		PrintNetParamsSingle("relu1", layer_relu1->blobs());
		PrintNetParamsSingle("mp1", layer_mp1->blobs());
		PrintNetParamsSingle("c2", layer_c2->blobs());
		PrintNetParamsSingle("relu2", layer_relu2->blobs());
		PrintNetParamsSingle("mp2", layer_mp2->blobs());
		PrintNetParamsSingle("c3", layer_c3->blobs());
		PrintNetParamsSingle("relu3", layer_relu3->blobs());
		PrintNetParamsSingle("mp3", layer_mp3->blobs());
		PrintNetParamsSingle("ip1", layer_ip1->blobs());
		PrintNetParamsSingle("sml1", layer_sml1->blobs());
	}


	void SaveNetParams(int epoch)
	{
		cudaSetDevice(gpu_id);

		stringstream f1; f1 << name << "_c1_weight_e" << epoch << ".mat";
		save_blob_to_mat(f1.str().c_str(), layer_c1->blobs()[0]);
		stringstream f2; f2 << name << "_c1_bias_e" << epoch << ".mat";
		save_blob_to_mat(f2.str().c_str(), layer_c1->blobs()[1]);


		stringstream f3; f3 << name << "_c2_weight_e" << epoch << ".mat";
		save_blob_to_mat(f3.str().c_str(), layer_c2->blobs()[0]);
		stringstream f4; f4 << name << "_c2_bias_e" << epoch << ".mat";
		save_blob_to_mat(f4.str().c_str(), layer_c2->blobs()[1]);


		stringstream f31; f31 << name << "_c3_weight_e" << epoch << ".mat";
		save_blob_to_mat(f31.str().c_str(), layer_c3->blobs()[0]);
		stringstream f41; f41 << name << "_c3_bias_e" << epoch << ".mat";
		save_blob_to_mat(f41.str().c_str(), layer_c3->blobs()[1]);

		stringstream f5; f5 << name << "_ip1_weight_e" << epoch << ".mat";
		save_blob_to_mat(f5.str().c_str(), layer_ip1->blobs()[0]);
		stringstream f6; f6 << name << "_ip1_bias_e" << epoch << ".mat";
		save_blob_to_mat(f6.str().c_str(), layer_ip1->blobs()[1]);
	}

};

void CopyNetParams_gpu(const SimpleNet *src, SimpleNet *dst)
{
	// CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
	int src_gpu_id = src->gpu_id;
	int dst_gpu_id = dst->gpu_id;

	if(src_gpu_id == dst_gpu_id) // they are in the same device
	{
		// printf("CopyNetParams_gpu: yes! src_gpu_id == dst_gpu_id.\n");
		for(int i = 0; i <= 1; i++)
		{
			caffe_copy(src->layer_c1->blobs()[i]->count(),
					src->layer_c1->blobs()[i]->gpu_data(),
					dst->layer_c1->blobs()[i]->mutable_gpu_data());

			caffe_copy(src->layer_c2->blobs()[i]->count(),
					src->layer_c2->blobs()[i]->gpu_data(),
					dst->layer_c2->blobs()[i]->mutable_gpu_data());

			caffe_copy(src->layer_c3->blobs()[i]->count(),
					src->layer_c3->blobs()[i]->gpu_data(),
					dst->layer_c3->blobs()[i]->mutable_gpu_data());

			caffe_copy(src->layer_ip1->blobs()[i]->count(),
					src->layer_ip1->blobs()[i]->gpu_data(),
					dst->layer_ip1->blobs()[i]->mutable_gpu_data());
		}
	}
	else // they are in different device
	{
		// printf("CopyNetParams_gpu: no! src_gpu_id != dst_gpu_id.\n");
		// printf("get device properties.\n");
		// check p2p access
		cudaDeviceProp prop[2];
		cudaGetDeviceProperties(&prop[0], src_gpu_id);
		cudaGetDeviceProperties(&prop[1], dst_gpu_id);
		// printf("get device properties (done).\n");

		// printf("check p2p access.\n");
		int can_access_peer;
		cudaDeviceCanAccessPeer(&can_access_peer, src_gpu_id, dst_gpu_id);
		// printf("check p2p access (done).\n");
		if(can_access_peer)
		{
			// printf("yes! both gpus have p2p access to each other.\n");
			const bool has_uva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
			if(has_uva)
			{
				// printf("yes! they have UVA access.\n");
				for(int i = 0; i <= 1; i++)
				{
					caffe_copy(src->layer_c1->blobs()[i]->count(),
							src->layer_c1->blobs()[i]->gpu_data(),
							dst->layer_c1->blobs()[i]->mutable_gpu_data());

					caffe_copy(src->layer_c2->blobs()[i]->count(),
							src->layer_c2->blobs()[i]->gpu_data(),
							dst->layer_c2->blobs()[i]->mutable_gpu_data());

					caffe_copy(src->layer_c3->blobs()[i]->count(),
							src->layer_c3->blobs()[i]->gpu_data(),
							dst->layer_c3->blobs()[i]->mutable_gpu_data());

					caffe_copy(src->layer_ip1->blobs()[i]->count(),
							src->layer_ip1->blobs()[i]->gpu_data(),
							dst->layer_ip1->blobs()[i]->mutable_gpu_data());
				}
				return;
			}
		}

		// printf("Both gpus have p2p access, but no UVA access. Thus we should first copy data into host, then copy to device.\n");
		// no p2p or UVA
		float *temp_data = NULL;
		int count = 0;
		for(int i = 0; i <= 1; i++)
		{
			cudaSetDevice(src_gpu_id);
			count = src->layer_c1->blobs()[i]->count();
			cudaMallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->layer_c1->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMemcpy(dst->layer_c1->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			cudaFreeHost(temp_data);

			cudaSetDevice(src_gpu_id);
			count = src->layer_c2->blobs()[i]->count();
			cudaMallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->layer_c2->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMemcpy(dst->layer_c2->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			cudaFreeHost(temp_data);

			cudaSetDevice(src_gpu_id);
			count = src->layer_c3->blobs()[i]->count();
			cudaMallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->layer_c3->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMemcpy(dst->layer_c3->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			cudaFreeHost(temp_data);

			cudaSetDevice(src_gpu_id);
			count = src->layer_ip1->blobs()[i]->count();
			cudaMallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->layer_ip1->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMemcpy(dst->layer_ip1->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			cudaFreeHost(temp_data);
		}
	}
}

void CopyBlobData_gpu(const Blob<float> *src, int src_gpu_id, Blob<float> *dst, int dst_gpu_id)
{
	int count = src->count();
	if(src_gpu_id == dst_gpu_id)
	{
		cudaSetDevice(src_gpu_id);
		cudaMemcpy(dst->mutable_gpu_data(), src->gpu_data(), count * sizeof(float), cudaMemcpyDefault);
	}
	else
	{
		// printf("CopyBlob_gpu: no! src_gpu_id != dst_gpu_id.\n");
		// printf("get device properties.\n");
		// check p2p access
		cudaDeviceProp prop[2];
		cudaGetDeviceProperties(&prop[0], src_gpu_id);
		cudaGetDeviceProperties(&prop[1], dst_gpu_id);
		// printf("get device properties (done).\n");

		// printf("check p2p access.\n");
		int can_access_peer;
		cudaDeviceCanAccessPeer(&can_access_peer, src_gpu_id, dst_gpu_id);
		// printf("check p2p access (done).\n");
		if(can_access_peer)
		{
			// printf("yes! both gpus have p2p access to each other.\n");
			const bool has_uva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
			if(has_uva)
			{
				// printf("yes! they have UVA access.\n");
				cudaMemcpy(dst->mutable_gpu_data(), src->gpu_data(), count * sizeof(float), cudaMemcpyDefault);
				return;
			}
		}

		// printf("Both gpus have p2p access, but no UVA access. Thus we should first copy data into host, then copy to device.\n");
		// no p2p or UVA
		float *temp_data = NULL;
		for(int i = 0; i <= 1; i++)
		{
			cudaSetDevice(src_gpu_id);
			cudaMallocHost((void **)&temp_data, count * sizeof(float));
			cudaMemcpy(temp_data, src->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(dst_gpu_id);
			cudaMemcpy(dst->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
			cudaFreeHost(temp_data);
		}
	}
}

void AddBlobDiff_gpu(const SimpleNet *src, SimpleNet *dst)
{
	// CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
	int src_gpu_id = src->gpu_id;
	int dst_gpu_id = dst->gpu_id;

	if(src_gpu_id == dst_gpu_id) // they are in the same device
	{
		// printf("CopyNetParams_gpu: yes! src_gpu_id == dst_gpu_id.\n");
		for(int i = 0; i <= 1; i++)
		{
			caffe_add(dst->layer_c1->blobs()[i]->count(),
					dst->layer_c1->blobs()[i]->gpu_diff(),
					src->layer_c1->blobs()[i]->gpu_diff(),
					src->layer_c1->blobs()[i]->mutable_gpu_diff());

			caffe_add(dst->layer_c2->blobs()[i]->count(),
					dst->layer_c2->blobs()[i]->gpu_diff(),
					src->layer_c2->blobs()[i]->gpu_diff(),
					src->layer_c2->blobs()[i]->mutable_gpu_diff());

			caffe_add(dst->layer_c3->blobs()[i]->count(),
					dst->layer_c3->blobs()[i]->gpu_diff(),
					src->layer_c3->blobs()[i]->gpu_diff(),
					src->layer_c3->blobs()[i]->mutable_gpu_diff());

			caffe_add(dst->layer_ip1->blobs()[i]->count(),
					dst->layer_ip1->blobs()[i]->gpu_diff(),
					src->layer_ip1->blobs()[i]->gpu_diff(),
					src->layer_ip1->blobs()[i]->mutable_gpu_diff());
		}
	}
	else // they are in different device
	{
		// printf("CopyNetParams_gpu: no! src_gpu_id != dst_gpu_id.\n");
		// printf("get device properties.\n");
		// check p2p access
		cudaDeviceProp prop[2];
		cudaGetDeviceProperties(&prop[0], src_gpu_id);
		cudaGetDeviceProperties(&prop[1], dst_gpu_id);
		// printf("get device properties (done).\n");

		// printf("check p2p access.\n");
		int can_access_peer;
		cudaDeviceCanAccessPeer(&can_access_peer, src_gpu_id, dst_gpu_id);
		// printf("check p2p access (done).\n");
		if(can_access_peer)
		{
			// printf("yes! both gpus have p2p access to each other.\n");
			const bool has_uva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
			if(has_uva)
			{
				// printf("yes! they have UVA access.\n");
				for(int i = 0; i <= 1; i++)
				{
					caffe_add(dst->layer_c1->blobs()[i]->count(),
							dst->layer_c1->blobs()[i]->gpu_diff(),
							src->layer_c1->blobs()[i]->gpu_diff(),
							src->layer_c1->blobs()[i]->mutable_gpu_diff());

					caffe_add(dst->layer_c2->blobs()[i]->count(),
							dst->layer_c2->blobs()[i]->gpu_diff(),
							src->layer_c2->blobs()[i]->gpu_diff(),
							src->layer_c2->blobs()[i]->mutable_gpu_diff());

					caffe_add(dst->layer_c3->blobs()[i]->count(),
							dst->layer_c3->blobs()[i]->gpu_diff(),
							src->layer_c3->blobs()[i]->gpu_diff(),
							src->layer_c3->blobs()[i]->mutable_gpu_diff());

					caffe_add(dst->layer_ip1->blobs()[i]->count(),
							dst->layer_ip1->blobs()[i]->gpu_diff(),
							src->layer_ip1->blobs()[i]->gpu_diff(),
							src->layer_ip1->blobs()[i]->mutable_gpu_diff());
				}
				return;
			}
		}

		// printf("Both gpus have p2p access, but no UVA access. Thus we should first copy data into host, then copy to device.\n");
		// no p2p or UVA
		float *temp_data = NULL;
		int count = 0;
		for(int i = 0; i <= 1; i++)
		{
			caffe_add(dst->layer_c1->blobs()[i]->count(),
					dst->layer_c1->blobs()[i]->gpu_diff(),
					src->layer_c1->blobs()[i]->gpu_diff(),
					src->layer_c1->blobs()[i]->mutable_gpu_diff());

			caffe_add(dst->layer_c2->blobs()[i]->count(),
					dst->layer_c2->blobs()[i]->gpu_diff(),
					src->layer_c2->blobs()[i]->gpu_diff(),
					src->layer_c2->blobs()[i]->mutable_gpu_diff());

			caffe_add(dst->layer_c3->blobs()[i]->count(),
					dst->layer_c3->blobs()[i]->gpu_diff(),
					src->layer_c3->blobs()[i]->gpu_diff(),
					src->layer_c3->blobs()[i]->mutable_gpu_diff());

			caffe_add(dst->layer_ip1->blobs()[i]->count(),
					dst->layer_ip1->blobs()[i]->gpu_diff(),
					src->layer_ip1->blobs()[i]->gpu_diff(),
					src->layer_ip1->blobs()[i]->mutable_gpu_diff());

//			cudaSetDevice(src_gpu_id);
//			count = src->layer_c1->blobs()[i]->count();
//			cudaMallocHost((void **)&temp_data, count * sizeof(float));
//			cudaMemcpy(temp_data, src->layer_c1->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
//			cudaSetDevice(dst_gpu_id);
//			cudaMemcpy(dst->layer_c1->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
//			cudaFreeHost(temp_data);
//
//			cudaSetDevice(src_gpu_id);
//			count = src->layer_c2->blobs()[i]->count();
//			cudaMallocHost((void **)&temp_data, count * sizeof(float));
//			cudaMemcpy(temp_data, src->layer_c2->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
//			cudaSetDevice(dst_gpu_id);
//			cudaMemcpy(dst->layer_c2->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
//			cudaFreeHost(temp_data);
//
//			cudaSetDevice(src_gpu_id);
//			count = src->layer_c3->blobs()[i]->count();
//			cudaMallocHost((void **)&temp_data, count * sizeof(float));
//			cudaMemcpy(temp_data, src->layer_c3->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
//			cudaSetDevice(dst_gpu_id);
//			cudaMemcpy(dst->layer_c3->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
//			cudaFreeHost(temp_data);
//
//			cudaSetDevice(src_gpu_id);
//			count = src->layer_ip1->blobs()[i]->count();
//			cudaMallocHost((void **)&temp_data, count * sizeof(float));
//			cudaMemcpy(temp_data, src->layer_ip1->blobs()[i]->gpu_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
//			cudaSetDevice(dst_gpu_id);
//			cudaMemcpy(dst->layer_ip1->blobs()[i]->mutable_gpu_data(), temp_data, count * sizeof(float), cudaMemcpyHostToDevice);
//			cudaFreeHost(temp_data);
		}
	}
}

void EnableP2P(vector<int> gpus)
{
	// check p2p access
	cudaDeviceProp prop[gpus.size()];
	for(int i = 0; i < gpus.size(); i++)
	{
		cudaGetDeviceProperties(&prop[i], gpus[i]);
	}

	for(int i = 0; i < gpus.size(); i++)
	{
		for(int j = 0; j < gpus.size(); j++)
		{
			if(i==j)
				continue;
			int can_access_peer;
			cudaDeviceCanAccessPeer(&can_access_peer, gpus[i], gpus[j]);
			if(can_access_peer)
			{
				cudaSetDevice(gpus[i]);
				cudaDeviceEnablePeerAccess(gpus[j], 0);
				cudaSetDevice(gpus[j]);
				cudaDeviceEnablePeerAccess(gpus[i], 0);
				const bool has_uva = (prop[gpus[i]].unifiedAddressing && prop[gpus[j]].unifiedAddressing);
				if(has_uva)
				{
					printf("yes! %d and %d have UVA access.\n", gpus[i], gpus[j]);
				}
			}
		}
	}
}

void DisableP2P(vector<int> gpus)
{
	for(int i = 0; i < gpus.size(); i++)
	{
		cudaSetDevice(gpus[i]);
		cudaDeviceDisablePeerAccess(gpus[i]);
	}
}

int main(int argc, char **argv)
{
	if(argc != 12)
	{
		printf("Usage: <filename> trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n");
		return -1;
	}
	string trn_db_filename = string(argv[1]);
	string tst_db_filename = string(argv[2]);
	string mean_file = string(argv[3]);
	float lr_rate = atof(argv[4]);
	int lr_stepsize = atoi(argv[5]);
	float momentum = atof(argv[6]);
	float weight_decay = atof(argv[7]);
	int trn_batch_size = atoi(argv[8]);
	int tst_batch_size = atoi(argv[9]);
	int max_epoch_num = atoi(argv[10]);
	string gpu_ids_str = string(argv[11]);

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	printf("number of gpus: %ld\n", gpus.size());

	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);

	if(num_gpus >= gpus.size())
	{
		printf("enable P2P: ");
		EnableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	}

	int current_gpu_id;
	cudaGetDevice(&current_gpu_id);
	printf("current gpu id: %d\n", current_gpu_id);

	Caffe::set_mode(Caffe::GPU);

	LayerParameter trn_param;
	trn_param.set_phase(TRAIN);
	DataParameter* trn_data_param = trn_param.mutable_data_param();
	trn_data_param->set_batch_size(trn_batch_size);
	trn_data_param->set_source(trn_db_filename.c_str());
	trn_data_param->set_mean_file(mean_file);
	trn_data_param->set_backend(DataParameter_DB_LMDB);

	Blob<float> *trn_blob_top_data_ = new Blob<float>();
	Blob<float> *trn_blob_top_label_ = new Blob<float>();
	vector<Blob<float>*> trn_blob_bottom_vec_;
	vector<Blob<float>*> trn_blob_top_vec_;
	trn_blob_top_vec_.push_back(trn_blob_top_data_);
	trn_blob_top_vec_.push_back(trn_blob_top_label_);

	DataLayer<float> trn_data_layer(trn_param);
	trn_data_layer.SetUp(trn_blob_bottom_vec_, trn_blob_top_vec_);

	LayerParameter layer_param1;
	LayerParameter layer_param2;
	SliceParameter *slice_data_layer_param = layer_param1.mutable_slice_param();
	SliceParameter *slice_label_layer_param = layer_param2.mutable_slice_param();
	slice_data_layer_param->set_slice_dim(0);
	slice_label_layer_param->set_slice_dim(0);
	SliceLayer<float> slice_data_layer(layer_param1);
	SliceLayer<float> slice_label_layer(layer_param2);
	Blob<float> *slice_data_bottom_blob = new Blob<float>(trn_batch_size, 3, 32, 32);
	Blob<float> *slice_label_bottom_blob = new Blob<float>(trn_batch_size, 1, 1, 1);
	vector<Blob<float>*> slice_data_bottom_vec_;
	slice_data_bottom_vec_.push_back(slice_data_bottom_blob);
	vector<Blob<float>*> slice_label_bottom_vec_;
	slice_label_bottom_vec_.push_back(slice_label_bottom_blob);
	vector<Blob<float>*> slice_data_top_vec_(gpus.size());
	vector<Blob<float>*> slice_label_top_vec_(gpus.size());
	for(int i = 0; i < gpus.size(); i++)
	{
		slice_data_top_vec_[i] = new Blob<float>();
		slice_label_top_vec_[i] = new Blob<float>();
	}
	slice_data_layer.SetUp(slice_data_bottom_vec_, slice_data_top_vec_);
	slice_label_layer.SetUp(slice_label_bottom_vec_, slice_label_top_vec_);
	vector<int> slice_batch_sizes(gpus.size());
	for(int i = 0; i < gpus.size(); i++)
	{
		slice_batch_sizes[i] = slice_data_top_vec_[i]->num();
	}

	LayerParameter tst_param;
	tst_param.set_phase(TEST);
	DataParameter* tst_data_param = tst_param.mutable_data_param();
	tst_data_param->set_batch_size(tst_batch_size);
	tst_data_param->set_source(tst_db_filename.c_str());
	tst_data_param->set_mean_file(mean_file);
	tst_data_param->set_backend(DataParameter_DB_LMDB);

	Blob<float> *tst_blob_top_data_ = new Blob<float>();
	Blob<float> *tst_blob_top_label_ = new Blob<float>();
	vector<Blob<float>*> tst_blob_bottom_vec_;
	vector<Blob<float>*> tst_blob_top_vec_;
	tst_blob_top_vec_.push_back(tst_blob_top_data_);
	tst_blob_top_vec_.push_back(tst_blob_top_label_);

	DataLayer<float> tst_data_layer(tst_param);
	tst_data_layer.SetUp(tst_blob_bottom_vec_, tst_blob_top_vec_);

	cudaSetDevice(current_gpu_id);
	SimpleNet *trn_net = new SimpleNet("trn_net");
	trn_net->BuildNet(trn_batch_size, current_gpu_id);
	trn_net->PrintNet();
	trn_net->PrintNetParams();
	trn_net->SaveNetParams(0);

	vector<SimpleNet *> trn_nets(gpus.size());
	for(int i = 0; i < gpus.size(); i++)
	{
		trn_nets[i] = NULL;
	}
	if(num_gpus > 1)
	{
		printf("initialize nets for each gpu ...\n");
		for(int i = 0; i < gpus.size(); i++)
		{
			printf("=========== gpu [%d] ==============\n", gpus[i]);
			cudaSetDevice(gpus[i]);
			trn_nets[i] = new SimpleNet(string("trn_nets_"+i));
			trn_nets[i]->BuildNet(slice_batch_sizes[i], gpus[i]);
			trn_nets[i]->PrintNet();
			trn_nets[i]->PrintNetParams();
		}
		printf("cudaLastError: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("initialize nets for each gpu (done) ...\n");
	}

	cudaSetDevice(current_gpu_id);
	SimpleNet *tst_net = new SimpleNet("tst_net");
	tst_net->BuildNet(tst_batch_size, current_gpu_id);
	tst_net->PrintNet();
	tst_net->PrintNetParams();

	for (int epoch = 0; epoch < max_epoch_num; ++epoch)
	{
		float trn_loss = 0.0f;
		float tst_loss = 0.0f;

		printf("epoch [%d] begin ...\n", epoch);
		printf("testing net ... \n ");
		// tst_net->CopyNetParamsFrom(trn_net);
		CopyNetParams_gpu(trn_net, tst_net);
		cudaSetDevice(current_gpu_id);
		printf("tst_net has copied from trn_net.\n");
		for (int iter = 0; iter < (int)(10000/tst_batch_size); ++iter)
		{
			tst_data_layer.Forward(tst_blob_bottom_vec_, tst_blob_top_vec_);

			tst_net->c1_bottom_blob->CopyFrom(*(tst_blob_top_vec_[0]), false, false);
			tst_net->label_blob->CopyFrom(*(tst_blob_top_vec_[1]), false, false);

			tst_loss += tst_net->Forward();
		}
		printf("testing net (done) ... \n ");

		printf("training net ... \n ");
		if(num_gpus == 1)
		{
			for (int iter = 0; iter < (int)(50000/trn_batch_size); ++iter)
			{
				trn_net->SetNetDiffToZero();
				// get batch_data
				trn_data_layer.Forward(trn_blob_bottom_vec_, trn_blob_top_vec_);

				trn_net->c1_bottom_blob->CopyFrom(*(trn_blob_top_vec_[0]), false, false);
				trn_net->label_blob->CopyFrom(*(trn_blob_top_vec_[1]), false, false);

				trn_loss += trn_net->ForwardBackward();
				trn_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);
				trn_net->UpdateNet();
			}
		}
		else
		{
			for (int iter = 0; iter < (int)(50000/trn_batch_size); ++iter)
			{
				trn_net->SetNetDiffToZero();

				// get batch_data
				trn_data_layer.Forward(trn_blob_bottom_vec_, trn_blob_top_vec_);

				// slice into the
				slice_data_bottom_vec_[0] = trn_blob_top_vec_[0];
				slice_data_layer.Forward(slice_data_bottom_vec_, slice_data_top_vec_);
				slice_label_bottom_vec_[0] = trn_blob_top_vec_[1];
				slice_label_layer.Forward(slice_label_bottom_vec_, slice_label_top_vec_);


				printf("begin multi-gpu ...\n");
				printf("copy net params into each gpu ...\n");
				for(int i = 0; i < gpus.size(); i++)
				{
					CopyNetParams_gpu(trn_net, trn_nets[i]);
				}
				printf("copy net params into each gpu (done) ...\n");
				omp_set_num_threads(gpus.size());
#pragma omp parallel
				{
					unsigned int cpu_thread_id = omp_get_thread_num();
					unsigned int num_cpu_threads = omp_get_num_threads();
					printf("cpu thread: %d / %d\n", cpu_thread_id, num_cpu_threads);

					CopyBlobData_gpu(slice_data_top_vec_[cpu_thread_id],  current_gpu_id, trn_nets[cpu_thread_id]->c1_bottom_blob, gpus[cpu_thread_id]);
					CopyBlobData_gpu(slice_label_top_vec_[cpu_thread_id],  current_gpu_id, trn_nets[cpu_thread_id]->label_blob, gpus[cpu_thread_id]);

					cudaSetDevice(gpus[cpu_thread_id]);
					printf("begin forward and backwrad ...\n");
					float trn_loss1 = trn_nets[cpu_thread_id]->ForwardBackward();
					printf("begin forward and backwrad (done)...\n");

					printf("compute update values ...\n");
					trn_nets[cpu_thread_id]->ComputeUpdateValue(lr_rate, momentum, weight_decay);
					printf("compute update values (done) ...\n");

				}
				cudaDeviceSynchronize();
				printf("seems done.\n");

				cudaSetDevice(current_gpu_id);
				printf("copy the gradient ...\n");
				for(int i = 0; i < gpus.size(); i++)
				{
					AddBlobDiff_gpu(trn_net, trn_nets[i]);
				}
				printf("copy the gradient (done) ...\n");

				cudaSetDevice(current_gpu_id);
				printf("update nets ...\n");
				trn_net->UpdateNet();
				printf("update nets (done) ...\n");
			}
		}

		if((epoch != 0) && (epoch % lr_stepsize == 0))
		{
			lr_rate /= 10;
			trn_net->SaveNetParams(epoch);
		}
		printf("training net (done) ... \n ");

		printf("epoch [%d] end: trn_loss=%.6f, tst_loss=%.6f\n", epoch, trn_loss, tst_loss);
	}

	if(num_gpus > 1)
	{
		printf("release resources in other gpus.\n");
		for(int i = 0; i < gpus.size(); i++)
		{
			if(trn_nets[i] != NULL)
			{
				cudaSetDevice(gpus[i]);
				delete trn_nets[i];
				cudaDeviceReset();
			}
		}
		printf("release resources in other gpus (done).\n");
	}

	printf("release resources for main gpus.\n");
	cudaSetDevice(current_gpu_id);
	for(int i = 0; i < gpus.size(); i++)
	{
		delete slice_data_top_vec_[i];
		delete slice_label_top_vec_[i];
	}
	delete slice_data_bottom_blob;
	delete slice_label_bottom_blob;

	delete trn_blob_top_data_;
	delete trn_blob_top_label_;
	delete tst_blob_top_data_;
	delete tst_blob_top_label_;

	delete trn_net;
	delete tst_net;
	printf("release resources for main gpus (done).\n");

	if(num_gpus >= gpus.size())
	{
		printf("disable P2P: ");
		DisableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	}

	return 0;
}























