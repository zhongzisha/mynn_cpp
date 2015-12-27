#include "network_alex.hpp"



AlexNetwork_t::AlexNetwork_t(string net_name_, int gpu_id_) {
	net_name = net_name_;
	gpu_id = gpu_id_;
	curand_stream = NULL;
	curand_generator = NULL;
	curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
	cublas_handle = NULL;

	is_allocate_top_mem = true;
	batch_samples = NULL;
	batch_labels = NULL;

	conv1 = NULL;
	conv1_top = NULL;
	conv1_params = NULL;
	relu1 = NULL;
	relu1_top = NULL;
	relu1_params = NULL;
	pool1 = NULL;
	pool1_top = NULL;
	pool1_params = NULL;
	conv2g = NULL;
	conv2g_top = NULL;
	conv2g_params = NULL;
	relu2 = NULL;
	relu2_top = NULL;
	relu2_params = NULL;
	pool2 = NULL;
	pool2_top = NULL;
	pool2_params = NULL;
	conv3 = NULL;
	conv3_top = NULL;
	conv3_params = NULL;
	relu3 = NULL;
	relu3_top = NULL;
	relu3_params = NULL;
	conv4g = NULL;
	conv4g_top = NULL;
	conv4g_params = NULL;
	relu4 = NULL;
	relu4_top = NULL;
	relu4_params = NULL;
	conv5g = NULL;
	conv5g_top = NULL;
	conv5g_params = NULL;
	relu5 = NULL;
	relu5_top = NULL;
	relu5_params = NULL;
	pool5 = NULL;
	pool5_top = NULL;
	pool5_params = NULL;
	fc6 = NULL;
	fc6_top = NULL;
	fc6_params = NULL;
	relu6 = NULL;
	relu6_top = NULL;
	relu6_params = NULL;
	fc7 = NULL;
	fc7_top = NULL;
	fc7_params = NULL;
	relu7 = NULL;
	relu7_top = NULL;
	relu7_params = NULL;
	fc8 = NULL;
	fc8_top = NULL;
	fc8_params = NULL;
	//		sm1 = NULL;
	//		sm1_top = NULL;
	//		sm1_params = NULL;
	//		argmax1 = NULL;
	//		argmax1_top = NULL;
	//		argmax1_params = NULL;
	//		mlr1 = NULL;
	//		mlr1_top = NULL;
	//		mlr1_params = NULL;
	sml1 = NULL;
	sml1_top = NULL;
	sml1_params = NULL;

	accuracy1 = NULL;
	accuracy1_top = NULL;
	accuracy1_params = NULL;

	conv1_filtersBlob_old = NULL;
	conv1_biasBlob_old = NULL;
	conv2g_filtersBlob_old = NULL;
	conv2g_biasBlob_old = NULL;
	conv3_filtersBlob_old = NULL;
	conv3_biasBlob_old = NULL;
	conv4g_filtersBlob_old = NULL;
	conv4g_biasBlob_old = NULL;
	conv5g_filtersBlob_old = NULL;
	conv5g_biasBlob_old = NULL;
	fc6_filtersBlob_old = NULL;
	fc6_biasBlob_old = NULL;
	fc7_filtersBlob_old = NULL;
	fc7_biasBlob_old = NULL;
	fc8_filtersBlob_old = NULL;
	fc8_biasBlob_old = NULL;

}

AlexNetwork_t::~AlexNetwork_t() {
	DestroyNet();
}

void AlexNetwork_t::DestroyNet() {

	cudaSetDevice(gpu_id);

	delete batch_samples; batch_samples = NULL;
	delete batch_labels; batch_labels = NULL;

	delete conv1; conv1 = NULL;
	delete relu1; relu1 = NULL;
	delete pool1; pool1 = NULL;
	delete conv2g; conv2g = NULL;
	delete relu2; relu2 = NULL;
	delete pool2; pool2 = NULL;
	delete conv3; conv3 = NULL;
	delete relu3; relu3 = NULL;
	delete conv4g; conv4g = NULL;
	delete relu4; relu4 = NULL;
	delete conv5g; conv5g = NULL;
	delete relu5; relu5 = NULL;
	delete pool5; pool5 = NULL;
	delete fc6; fc6 = NULL;
	delete relu6; relu6 = NULL;
	delete fc7; fc7 = NULL;
	delete relu7; relu7 = NULL;
	delete fc8; fc8 = NULL;
	//		delete sm1; sm1 = NULL;
	//      delete argmax1; argmax1 = NULL;
	//		delete mlr1; mlr1 = NULL;
	delete sml1; sml1 = NULL;
	delete accuracy1; accuracy1 = NULL;

	delete conv1_top; conv1_top = NULL;
	delete relu1_top; relu1_top = NULL;
	delete pool1_top; pool1 = NULL;
	delete conv2g_top; conv2g_top = NULL;
	delete relu2_top; relu2_top = NULL;
	delete pool2_top; pool2_top = NULL;
	delete conv3_top; conv3_top = NULL;
	delete relu3_top; relu3_top = NULL;
	delete conv4g_top; conv4g_top = NULL;
	delete relu4_top; relu4_top = NULL;
	delete conv5g_top; conv5g_top = NULL;
	delete relu5_top; relu5_top = NULL;
	delete pool5_top; pool5_top = NULL;
	delete fc6_top; fc6_top = NULL;
	delete relu6_top; relu6_top = NULL;
	delete fc7_top; fc7_top = NULL;
	delete relu7_top; relu7_top = NULL;
	delete fc8_top; fc8_top = NULL;
	//		delete sm1_top; sm1_top = NULL;
	//      delete argmax1_top; argmax1_top = NULL;
	//		delete mlr1_top; mlr1_top = NULL;
	delete sml1_top; sml1_top = NULL;
	delete accuracy1_top; accuracy1_top = NULL;

	delete conv1_params; conv1_params = NULL;
	delete relu1_params; relu1_params = NULL;
	delete pool1_params; pool1_params = NULL;
	delete conv2g_params; conv2g_params = NULL;
	delete relu2_params; relu2_params = NULL;
	delete pool2_params; pool2_params = NULL;
	delete conv3_params; conv3_params = NULL;
	delete relu3_params; relu3_params = NULL;
	delete conv4g_params; conv4g_params = NULL;
	delete relu4_params; relu4_params = NULL;
	delete conv5g_params; conv5g_params = NULL;
	delete relu5_params; relu5_params = NULL;
	delete pool5_params; pool5_params = NULL;
	delete fc6_params; fc6_params = NULL;
	delete relu6_params; relu6_params = NULL;
	delete fc7_params; fc7_params = NULL;
	delete relu7_params; relu7_params = NULL;
	delete fc8_params; fc8_params = NULL;
	//		delete sm1_params; sm1_params = NULL;
	//      delete argmax1_params; argmax1_params = NULL;
	//		delete mlr1_params; mlr1_params = NULL;
	delete sml1_params; sml1_params = NULL;
	delete accuracy1_params; accuracy1_params = NULL;

	delete conv1_filtersBlob_old; conv1_filtersBlob_old = NULL;
	delete conv1_biasBlob_old; conv1_biasBlob_old = NULL;
	delete conv2g_filtersBlob_old; conv2g_filtersBlob_old = NULL;
	delete conv2g_biasBlob_old; conv2g_biasBlob_old = NULL;
	delete conv3_filtersBlob_old; conv3_filtersBlob_old = NULL;
	delete conv3_biasBlob_old; conv3_biasBlob_old = NULL;
	delete conv4g_filtersBlob_old; conv4g_filtersBlob_old = NULL;
	delete conv4g_biasBlob_old; conv4g_biasBlob_old = NULL;
	delete conv5g_filtersBlob_old; conv5g_filtersBlob_old = NULL;
	delete conv5g_biasBlob_old; conv5g_biasBlob_old = NULL;
	delete fc6_filtersBlob_old; fc6_filtersBlob_old = NULL;
	delete fc6_biasBlob_old; fc6_biasBlob_old = NULL;
	delete fc7_filtersBlob_old; fc7_filtersBlob_old = NULL;
	delete fc7_biasBlob_old; fc7_biasBlob_old = NULL;
	delete fc8_filtersBlob_old; fc8_filtersBlob_old = NULL;
	delete fc8_biasBlob_old; fc8_biasBlob_old = NULL;

	CURAND_CHECK( curandDestroyGenerator(curand_generator) );
	CUDA_CHECK( cudaStreamDestroy(curand_stream) );
	CUBLAS_CHECK( cublasDestroy(cublas_handle) );
}

void AlexNetwork_t::BuildNet(int batch_size_, bool is_allocate_top_mem_, const string &net_params_file) {

	is_allocate_top_mem = is_allocate_top_mem_;

	cudaSetDevice(gpu_id);

	CUDA_CHECK( cudaStreamCreate(&curand_stream) );
	curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
	CURAND_CHECK( curandCreateGenerator(&curand_generator, curand_rngtype) );
	CURAND_CHECK( curandSetStream(curand_generator, curand_stream) );
	CUBLAS_CHECK( cublasCreate(&cublas_handle) );

	batch_samples = new Blob_t(batch_size_, 3, 227, 227);
	batch_labels = new Blob_t(batch_size_, 1, 1, 1);

	if(is_allocate_top_mem) {
		batch_samples->allocate_gpu_data();
		batch_samples->allocate_gpu_diff();
		batch_labels->allocate_gpu_data();
		batch_labels->allocate_cpu_data();
	}

	LOG(INFO) << "conv1 setup.\n";
	conv1_top = new Blob_t();
	conv1_params = new ConvolutionParameter_t();
	conv1_params->filter_N = 3;
	conv1_params->filter_C = 96;
	conv1_params->filter_H = 11;
	conv1_params->filter_W = 11;
	conv1_params->pad_h = 0;
	conv1_params->pad_w = 0;
	conv1_params->stride_h = 4;
	conv1_params->stride_w = 4;
	conv1_params->upscale_h = 1;
	conv1_params->upscale_w = 1;
	conv1_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv1 = new ConvolutionLayer_t(conv1_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->filtersBlob->data_gpu, conv1->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->biasBlob->data_gpu, conv1->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv1->biasBlob->count(), 0, conv1->biasBlob->data_gpu);
	conv1->Setup(batch_samples, conv1_top, is_allocate_top_mem);
	LOG(INFO) << "conv1 top: "
			<< conv1_top->N << ", "
			<< conv1_top->C << ", "
			<< conv1_top->H << ", "
			<< conv1_top->W;


	LOG(INFO) << "relu1 setup.\n";
	relu1_top = new Blob_t();
	relu1_params = new ActivationParameter_t();
	relu1_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu1 = new ActivationLayer_t(relu1_params);
	relu1->Setup(conv1_top, relu1_top, is_allocate_top_mem);
	LOG(INFO) << "relu1 top: "
			<< relu1_top->N << ", "
			<< relu1_top->C << ", "
			<< relu1_top->H << ", "
			<< relu1_top->W;

	LOG(INFO) << "pool1 setup.\n";
	pool1_top = new Blob_t();
	pool1_params = new PoolingParameter_t();
	pool1_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	pool1_params->poolsize_h = 3;
	pool1_params->poolsize_w = 3;
	pool1_params->pad_h = 0;
	pool1_params->pad_w = 0;
	pool1_params->stride_h = 2;
	pool1_params->stride_w = 2;
	pool1 = new PoolingLayer_t(pool1_params);
	pool1->Setup(relu1_top, pool1_top, is_allocate_top_mem);
	LOG(INFO) << "pool1 top: "
			<< pool1_top->N << ", "
			<< pool1_top->C << ", "
			<< pool1_top->H << ", "
			<< pool1_top->W;

	LOG(INFO) << "conv2g setup.\n";
	conv2g_top = new Blob_t();
	conv2g_params = new ConvolutionWithGroupParameter_t();
	conv2g_params->group = 2;
	conv2g_params->filter_N = 32;
	conv2g_params->filter_C = 256;
	conv2g_params->filter_H = 5;
	conv2g_params->filter_W = 5;
	conv2g_params->pad_h = 2;
	conv2g_params->pad_w = 2;
	conv2g_params->stride_h = 1;
	conv2g_params->stride_w = 1;
	conv2g_params->upscale_h = 1;
	conv2g_params->upscale_w = 1;
	conv2g_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv2g_params->cudnn_conv_fwd_preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	conv2g = new ConvolutionWithGroupLayer_t(conv2g_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv2g->filtersBlob->data_gpu, conv2g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv2g->biasBlob->count(), float(0.1f), conv2g->biasBlob->data_gpu);
	conv2g->Setup(pool1_top, conv2g_top, is_allocate_top_mem);
	LOG(INFO) << "conv2g top: "
			<< conv2g_top->N << ", "
			<< conv2g_top->C << ", "
			<< conv2g_top->H << ", "
			<< conv2g_top->W;


	LOG(INFO) << "relu2 setup.\n";
	relu2_top = new Blob_t();
	relu2_params = new ActivationParameter_t();
	relu2_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu2 = new ActivationLayer_t(relu2_params);
	relu2->Setup(conv2g_top, relu2_top, is_allocate_top_mem);
	LOG(INFO) << "relu2 top: "
			<< relu2_top->N << ", "
			<< relu2_top->C << ", "
			<< relu2_top->H << ", "
			<< relu2_top->W;

	LOG(INFO) << "pool2 setup.\n";
	pool2_top = new Blob_t();
	pool2_params = new PoolingParameter_t();
	pool2_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	pool2_params->poolsize_h = 3;
	pool2_params->poolsize_w = 3;
	pool2_params->pad_h = 0;
	pool2_params->pad_w = 0;
	pool2_params->stride_h = 2;
	pool2_params->stride_w = 2;
	pool2 = new PoolingLayer_t(pool2_params);
	pool2->Setup(relu2_top, pool2_top, is_allocate_top_mem);
	LOG(INFO) << "pool2 top: "
			<< pool2_top->N << ", "
			<< pool2_top->C << ", "
			<< pool2_top->H << ", "
			<< pool2_top->W;

	LOG(INFO) << "conv3 setup.\n";
	conv3_top = new Blob_t();
	conv3_params = new ConvolutionParameter_t();
	conv3_params->filter_N = 256;
	conv3_params->filter_C = 384;
	conv3_params->filter_H = 3;
	conv3_params->filter_W = 3;
	conv3_params->pad_h = 1;
	conv3_params->pad_w = 1;
	conv3_params->stride_h = 1;
	conv3_params->stride_w = 1;
	conv3_params->upscale_h = 1;
	conv3_params->upscale_w = 1;
	conv3_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv3 = new ConvolutionLayer_t(conv3_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->filtersBlob->data_gpu, conv3->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->biasBlob->data_gpu, conv3->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv3->biasBlob->count(), 0, conv3->biasBlob->data_gpu);
	conv3->Setup(pool2_top, conv3_top, is_allocate_top_mem);
	LOG(INFO) << "conv3 top: "
			<< conv3_top->N << ", "
			<< conv3_top->C << ", "
			<< conv3_top->H << ", "
			<< conv3_top->W;


	LOG(INFO) << "relu3 setup.\n";
	relu3_top = new Blob_t();
	relu3_params = new ActivationParameter_t();
	relu3_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu3 = new ActivationLayer_t(relu3_params);
	relu3->Setup(conv3_top, relu3_top, is_allocate_top_mem);
	LOG(INFO) << "relu3 top: "
			<< relu3_top->N << ", "
			<< relu3_top->C << ", "
			<< relu3_top->H << ", "
			<< relu3_top->W;

	LOG(INFO) << "conv4g setup.\n";
	conv4g_top = new Blob_t();
	conv4g_params = new ConvolutionWithGroupParameter_t();
	conv4g_params->group = 2;
	conv4g_params->filter_N = 384;
	conv4g_params->filter_C = 384;
	conv4g_params->filter_H = 3;
	conv4g_params->filter_W = 3;
	conv4g_params->pad_h = 1;
	conv4g_params->pad_w = 1;
	conv4g_params->stride_h = 1;
	conv4g_params->stride_w = 1;
	conv4g_params->upscale_h = 1;
	conv4g_params->upscale_w = 1;
	conv4g_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv4g_params->cudnn_conv_fwd_preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	conv4g = new ConvolutionWithGroupLayer_t(conv4g_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv4g->filtersBlob->data_gpu, conv4g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv4g->biasBlob->count(), float(0.1f), conv4g->biasBlob->data_gpu);
	conv4g->Setup(relu3_top, conv4g_top, is_allocate_top_mem);
	LOG(INFO) << "conv4g top: "
			<< conv4g_top->N << ", "
			<< conv4g_top->C << ", "
			<< conv4g_top->H << ", "
			<< conv4g_top->W;

	LOG(INFO) << "relu4 setup.\n";
	relu4_top = new Blob_t();
	relu4_params = new ActivationParameter_t();
	relu4_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu4 = new ActivationLayer_t(relu4_params);
	relu4->Setup(conv4g_top, relu4_top, is_allocate_top_mem);
	LOG(INFO) << "relu4 top: "
			<< relu4_top->N << ", "
			<< relu4_top->C << ", "
			<< relu4_top->H << ", "
			<< relu4_top->W;

	LOG(INFO) << ("conv5g setup.\n");
	conv5g_top = new Blob_t();
	conv5g_params = new ConvolutionWithGroupParameter_t();
	conv5g_params->group = 2;
	conv5g_params->filter_N = 384;
	conv5g_params->filter_C = 256;
	conv5g_params->filter_H = 3;
	conv5g_params->filter_W = 3;
	conv5g_params->pad_h = 1;
	conv5g_params->pad_w = 1;
	conv5g_params->stride_h = 1;
	conv5g_params->stride_w = 1;
	conv5g_params->upscale_h = 1;
	conv5g_params->upscale_w = 1;
	conv5g_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv5g_params->cudnn_conv_fwd_preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	conv5g = new ConvolutionWithGroupLayer_t(conv5g_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv5g->filtersBlob->data_gpu, conv5g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv5g->biasBlob->count(), float(0.1f), conv5g->biasBlob->data_gpu);
	conv5g->Setup(relu4_top, conv5g_top, is_allocate_top_mem);
	LOG(INFO) << "conv5g top: "
			<< conv5g_top->N << ", "
			<< conv5g_top->C << ", "
			<< conv5g_top->H << ", "
			<< conv5g_top->W;

	LOG(INFO) << ("relu5 setup.\n");
	relu5_top = new Blob_t();
	relu5_params = new ActivationParameter_t();
	relu5_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu5 = new ActivationLayer_t(relu5_params);
	relu5->Setup(conv5g_top, relu5_top, is_allocate_top_mem);
	LOG(INFO) << "relu5 top: "
			<< relu5_top->N << ", "
			<< relu5_top->C << ", "
			<< relu5_top->H << ", "
			<< relu5_top->W;

	LOG(INFO) << ("pool5 setup.\n");
	pool5_top = new Blob_t();
	pool5_params = new PoolingParameter_t();
	pool5_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	pool5_params->poolsize_h = 3;
	pool5_params->poolsize_w = 3;
	pool5_params->pad_h = 0;
	pool5_params->pad_w = 0;
	pool5_params->stride_h = 2;
	pool5_params->stride_w = 2;
	pool5 = new PoolingLayer_t(pool5_params);
	pool5->Setup(relu5_top, pool5_top, is_allocate_top_mem);
	LOG(INFO) << "pool5 top: "
			<< pool5_top->N << ", "
			<< pool5_top->C << ", "
			<< pool5_top->H << ", "
			<< pool5_top->W;

	LOG(INFO) << ("fc6 setup.\n");
	fc6_top = new Blob_t();
	fc6_params = new FullyConnectedParameter_t();
	fc6_params->hidden_size = 4096;
	fc6 = new FullyConnectedLayer_t(fc6_params);
	fc6->Setup(pool5_top, fc6_top, is_allocate_top_mem);
	CURAND_CHECK( curandGenerateNormal(curand_generator, fc6->filtersBlob->data_gpu, fc6->filtersBlob->count(), (float)0.0f, (float)0.005f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(fc6->biasBlob->count(), float(0.1f), fc6->biasBlob->data_gpu);
	LOG(INFO) << "fc6 top: "
			<< fc6_top->N << ", "
			<< fc6_top->C << ", "
			<< fc6_top->H << ", "
			<< fc6_top->W;

	LOG(INFO) << ("relu6 setup.\n");
	relu6_top = new Blob_t();
	relu6_params = new ActivationParameter_t();
	relu6_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu6 = new ActivationLayer_t(relu6_params);
	relu6->Setup(fc6_top, relu6_top, is_allocate_top_mem);
	LOG(INFO) << "relu6 top: "
			<< relu6_top->N << ", "
			<< relu6_top->C << ", "
			<< relu6_top->H << ", "
			<< relu6_top->W;

	LOG(INFO) << ("fc7 setup.\n");
	fc7_top = new Blob_t();
	fc7_params = new FullyConnectedParameter_t();
	fc7_params->hidden_size = 4096;
	fc7 = new FullyConnectedLayer_t(fc7_params);
	fc7->Setup(relu6_top, fc7_top, is_allocate_top_mem);
	CURAND_CHECK( curandGenerateNormal(curand_generator, fc7->filtersBlob->data_gpu, fc7->filtersBlob->count(), (float)0.0f, (float)0.005f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(fc7->biasBlob->count(), float(0.1f), fc7->biasBlob->data_gpu);
	LOG(INFO) << "fc7 top: "
			<< fc7_top->N << ", "
			<< fc7_top->C << ", "
			<< fc7_top->H << ", "
			<< fc7_top->W;

	LOG(INFO) << ("relu7 setup.\n");
	relu7_top = new Blob_t();
	relu7_params = new ActivationParameter_t();
	relu7_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu7 = new ActivationLayer_t(relu7_params);
	relu7->Setup(fc7_top, relu7_top, is_allocate_top_mem);
	LOG(INFO) << "relu7 top: "
			<< relu7_top->N << ", "
			<< relu7_top->C << ", "
			<< relu7_top->H << ", "
			<< relu7_top->W;

	LOG(INFO) << ("fc8 setup.\n");
	fc8_top = new Blob_t();
	fc8_params = new FullyConnectedParameter_t();
	fc8_params->hidden_size = 1000;
	fc8 = new FullyConnectedLayer_t(fc8_params);
	fc8->Setup(relu7_top, fc8_top, is_allocate_top_mem);
	CURAND_CHECK( curandGenerateNormal(curand_generator, fc8->filtersBlob->data_gpu, fc8->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(fc8->biasBlob->count(), float(0.0f), fc8->biasBlob->data_gpu);
	LOG(INFO) << "fc8 top: "
			<< fc8_top->N << ", "
			<< fc8_top->C << ", "
			<< fc8_top->H << ", "
			<< fc8_top->W;

	//		printf("sm1 setup.\n");
	//		sm1_top = new Blob_t();
	//		sm1_params = new SoftmaxParameter_t();
	//		sm1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
	//		sm1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
	//		sm1 = new SoftmaxLayer_t(sm1_params);
	//		sm1->Setup(ip1_top, sm1_top);
	//
	//		printf("mlr1 setup (in cpu).\n");
	//		mlr1_top = new Blob_t();
	//		mlr1_params = new MultinomialLogisticLossParameter_t();
	//		mlr1 = new MultinomialLogisticLossLayer_t(mlr1_params);
	//		mlr1->Setup(sm1_top, mlr1_top);

	LOG(INFO) << ("sml1 setup.\n");
	sml1_top = new Blob_t();
	sml1_params = new SoftmaxWithLossParameter_t();
	sml1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
	sml1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
	sml1_params->has_ignore_label = false;
	sml1_params->ignore_label = -1;
	sml1_params->normalize = false;
	sml1 = new SoftmaxWithLossLayer_t(sml1_params);
	sml1->Setup(fc8_top, sml1_top, is_allocate_top_mem);
	LOG(INFO) << "sml1 top: "
			<< sml1_top->N << ", "
			<< sml1_top->C << ", "
			<< sml1_top->H << ", "
			<< sml1_top->W;

	//		printf("argmax1 setup.\n");
	//		argmax1_top = new Blob_t();
	//		argmax1_params = new ArgMaxParameter_t();
	//		argmax1_params->out_max_val = true;
	//		argmax1_params->top_k = 1;
	//		argmax1 = new ArgMaxLayer_t(argmax1_params);
	//		argmax1->Setup(sml1_top, argmax1_top);

	LOG(INFO) << ("accuracy1 setup.\n");
	accuracy1_top = new Blob_t();
	accuracy1_params = new AccuracyParameter_t();
	accuracy1_params->top_k = 1;
	accuracy1 = new AccuracyLayer_t(accuracy1_params);
	accuracy1->Setup(fc8_top, accuracy1_top, is_allocate_top_mem);
	LOG(INFO) << "accuracy1 top: "
			<< accuracy1_top->N << ", "
			<< accuracy1_top->C << ", "
			<< accuracy1_top->H << ", "
			<< accuracy1_top->W;

	LOG(INFO) << ("initialize old net params.\n");
	conv1_filtersBlob_old = new Blob_t(conv1->filtersBlob->N, conv1->filtersBlob->C, conv1->filtersBlob->H, conv1->filtersBlob->W);
	conv1_biasBlob_old = new Blob_t(conv1->biasBlob->N, conv1->biasBlob->C, conv1->biasBlob->H, conv1->biasBlob->W);
	conv1_filtersBlob_old->allocate_gpu_data();
	conv1_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv1_filtersBlob_old->data_gpu, conv1_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv1_biasBlob_old->data_gpu, conv1_biasBlob_old->count(), 0.0f, 0.0f);

	conv2g_filtersBlob_old = new Blob_t(conv2g->filtersBlob->N, conv2g->filtersBlob->C, conv2g->filtersBlob->H, conv2g->filtersBlob->W);
	conv2g_biasBlob_old = new Blob_t(conv2g->biasBlob->N, conv2g->biasBlob->C, conv2g->biasBlob->H, conv2g->biasBlob->W);
	conv2g_filtersBlob_old->allocate_gpu_data();
	conv2g_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv2g_filtersBlob_old->data_gpu, conv2g_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv2g_biasBlob_old->data_gpu, conv2g_biasBlob_old->count(), 0.0f, 0.0f);

	conv3_filtersBlob_old = new Blob_t(conv3->filtersBlob->N, conv3->filtersBlob->C, conv3->filtersBlob->H, conv3->filtersBlob->W);
	conv3_biasBlob_old = new Blob_t(conv3->biasBlob->N, conv3->biasBlob->C, conv3->biasBlob->H, conv3->biasBlob->W);
	conv3_filtersBlob_old->allocate_gpu_data();
	conv3_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv3_filtersBlob_old->data_gpu, conv3_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv3_biasBlob_old->data_gpu, conv3_biasBlob_old->count(), 0.0f, 0.0f);

	conv4g_filtersBlob_old = new Blob_t(conv4g->filtersBlob->N, conv4g->filtersBlob->C, conv4g->filtersBlob->H, conv4g->filtersBlob->W);
	conv4g_biasBlob_old = new Blob_t(conv4g->biasBlob->N, conv4g->biasBlob->C, conv4g->biasBlob->H, conv4g->biasBlob->W);
	conv4g_filtersBlob_old->allocate_gpu_data();
	conv4g_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv4g_filtersBlob_old->data_gpu, conv4g_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv4g_biasBlob_old->data_gpu, conv4g_biasBlob_old->count(), 0.0f, 0.0f);

	conv5g_filtersBlob_old = new Blob_t(conv5g->filtersBlob->N, conv5g->filtersBlob->C, conv5g->filtersBlob->H, conv5g->filtersBlob->W);
	conv5g_biasBlob_old = new Blob_t(conv5g->biasBlob->N, conv5g->biasBlob->C, conv5g->biasBlob->H, conv5g->biasBlob->W);
	conv5g_filtersBlob_old->allocate_gpu_data();
	conv5g_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv5g_filtersBlob_old->data_gpu, conv5g_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv5g_biasBlob_old->data_gpu, conv5g_biasBlob_old->count(), 0.0f, 0.0f);

	fc6_filtersBlob_old = new Blob_t(fc6->filtersBlob->N, fc6->filtersBlob->C, fc6->filtersBlob->H, fc6->filtersBlob->W);
	fc6_biasBlob_old = new Blob_t(fc6->biasBlob->N, fc6->biasBlob->C, fc6->biasBlob->H, fc6->biasBlob->W);
	fc6_filtersBlob_old->allocate_gpu_data();
	fc6_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, fc6_filtersBlob_old->data_gpu, fc6_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, fc6_biasBlob_old->data_gpu, fc6_biasBlob_old->count(), 0.0f, 0.0f);

	fc7_filtersBlob_old = new Blob_t(fc7->filtersBlob->N, fc7->filtersBlob->C, fc7->filtersBlob->H, fc7->filtersBlob->W);
	fc7_biasBlob_old = new Blob_t(fc7->biasBlob->N, fc7->biasBlob->C, fc7->biasBlob->H, fc7->biasBlob->W);
	fc7_filtersBlob_old->allocate_gpu_data();
	fc7_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, fc7_filtersBlob_old->data_gpu, fc7_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, fc7_biasBlob_old->data_gpu, fc7_biasBlob_old->count(), 0.0f, 0.0f);

	fc8_filtersBlob_old = new Blob_t(fc8->filtersBlob->N, fc8->filtersBlob->C, fc8->filtersBlob->H, fc8->filtersBlob->W);
	fc8_biasBlob_old = new Blob_t(fc8->biasBlob->N, fc8->biasBlob->C, fc8->biasBlob->H, fc8->biasBlob->W);
	fc8_filtersBlob_old->allocate_gpu_data();
	fc8_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, fc8_filtersBlob_old->data_gpu, fc8_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, fc8_biasBlob_old->data_gpu, fc8_biasBlob_old->count(), 0.0f, 0.0f);

	LOG(INFO) << ("build net (done).\n");
}

void AlexNetwork_t::Forward(float *loss, float *accuracy) {
	cudaSetDevice(gpu_id);

	conv1->Forward(batch_samples, conv1_top);
	relu1->Forward(conv1_top, relu1_top);
	pool1->Forward(relu1_top, pool1_top);

	conv2g->Forward(pool1_top, conv2g_top);
	relu2->Forward(conv2g_top, relu2_top);
	pool2->Forward(relu2_top, pool2_top);

	conv3->Forward(pool2_top, conv3_top);
	relu3->Forward(conv3_top, relu3_top);

	conv4g->Forward(relu3_top, conv4g_top);
	relu4->Forward(conv4g_top, relu4_top);

	conv5g->Forward(relu4_top, conv5g_top);
	relu5->Forward(conv5g_top, relu5_top);
	pool5->Forward(relu5_top, pool5_top);

	fc6->Forward(pool5_top, fc6_top);
	relu6->Forward(fc6_top, relu6_top);

	fc7->Forward(relu6_top, fc7_top);
	relu7->Forward(fc7_top, relu7_top);

	fc8->Forward(relu7_top, fc8_top);

	// sm1->Forward(fc8_top, sm1_top);

	// mlr1->Forward(sm1_top, batch_labels, mlr1_top);

	// loss = mlr1_top->data_cpu[0];

	sml1->Forward(fc8_top, batch_labels, sml1_top, loss);

	//		argmax1->Forward_cpu(sml1_top, argmax1_top);

	accuracy1->Forward_cpu(fc8_top, batch_labels, accuracy1_top);

	*accuracy = accuracy1_top->data_cpu[0];

}

void AlexNetwork_t::Backward() {
	cudaSetDevice(gpu_id);

	sml1->Backward(sml1_top, batch_labels, fc8_top);

	// mlr1->Backward(mlr1_top, batch_labels, sm1_top);

	// sm1->Backward(sm1_top, ip1_top);

	fc8->Backward(fc8_top, relu7_top);

	relu7->Backward(relu7_top, fc7_top);
	fc7->Backward(fc7_top, relu6_top);

	relu6->Backward(relu6_top, fc6_top);
	fc6->Backward(fc6_top, pool5_top);

	pool5->Backward(pool5_top, relu5_top);
	relu5->Backward(relu5_top, conv5g_top);
	conv5g->Backward(conv5g_top, relu4_top);

	relu4->Backward(relu4_top, conv4g_top);
	conv4g->Backward(conv4g_top, relu3_top);

	relu3->Backward(relu3_top, conv3_top);
	conv3->Backward(conv3_top, pool2_top);

	pool2->Backward(pool2_top, relu2_top);
	relu2->Backward(relu2_top, conv2g_top);
	conv2g->Backward(conv2g_top, pool1_top);

	pool1->Backward(pool1_top, relu1_top);
	relu1->Backward(relu1_top, conv1_top);
	conv1->Backward(conv1_top, batch_samples);
}

void AlexNetwork_t::ForwardBackward(float *loss, float *accuracy) {
	Forward(loss, accuracy);
	Backward();
}

void AlexNetwork_t::ComputeUpdateValueSingle(Blob_t *param_gradient_blob, Blob_t *param_blob_old,
		float lr_rate, float momentum, float weight_decay) {
	gpu_axpy(cublas_handle,
			param_gradient_blob->count(), weight_decay,
			param_gradient_blob->data_gpu,
			param_gradient_blob->diff_gpu);

	gpu_axpby(cublas_handle,
			param_gradient_blob->count(), lr_rate,
			param_gradient_blob->diff_gpu, momentum,
			param_blob_old->data_gpu);
	// copy
	gpu_copy(param_gradient_blob->count(),
			param_blob_old->data_gpu,
			param_gradient_blob->diff_gpu);
}
void AlexNetwork_t::ComputeUpdateValue(float lr_rate, float momentum, float weight_decay) {
	cudaSetDevice(gpu_id);
	ComputeUpdateValueSingle(fc8->filtersBlob,   fc8_filtersBlob_old,   lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(fc8->biasBlob,      fc8_biasBlob_old, 		lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(fc7->filtersBlob,   fc7_filtersBlob_old,   lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(fc7->biasBlob,      fc7_biasBlob_old, 		lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(fc6->filtersBlob,   fc6_filtersBlob_old,   lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(fc6->biasBlob,      fc6_biasBlob_old, 		lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv5g->filtersBlob, conv5g_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv5g->biasBlob, 	 conv5g_biasBlob_old, 	lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv4g->filtersBlob, conv4g_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv4g->biasBlob, 	 conv4g_biasBlob_old, 	lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv3->filtersBlob, conv3_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv3->biasBlob, 	 conv3_biasBlob_old, 	lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv2g->filtersBlob, conv2g_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv2g->biasBlob, 	 conv2g_biasBlob_old, 	lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv1->filtersBlob, conv1_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv1->biasBlob, 	 conv1_biasBlob_old, 	lr_rate, momentum, weight_decay);
}

void AlexNetwork_t::UpdateNet(float scale) {
	cudaSetDevice(gpu_id);
	gpu_axpy(cublas_handle, fc8->filtersBlob->count(),   float(scale), fc8->filtersBlob->diff_gpu,   fc8->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, fc8->biasBlob->count(), 	 float(scale), fc8->biasBlob->diff_gpu,      fc8->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, fc7->filtersBlob->count(),   float(scale), fc7->filtersBlob->diff_gpu,   fc7->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, fc7->biasBlob->count(), 	 float(scale), fc7->biasBlob->diff_gpu,      fc7->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, fc6->filtersBlob->count(),   float(scale), fc6->filtersBlob->diff_gpu,   fc6->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, fc6->biasBlob->count(), 	 float(scale), fc6->biasBlob->diff_gpu,      fc6->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, conv5g->filtersBlob->count(), float(scale), conv5g->filtersBlob->diff_gpu, conv5g->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv5g->biasBlob->count(), 	 float(scale), conv5g->biasBlob->diff_gpu, 	  conv5g->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, conv4g->filtersBlob->count(), float(scale), conv4g->filtersBlob->diff_gpu, conv4g->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv4g->biasBlob->count(), 	 float(scale), conv4g->biasBlob->diff_gpu, 	  conv4g->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, conv3->filtersBlob->count(), float(scale), conv3->filtersBlob->diff_gpu, conv3->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv3->biasBlob->count(), 	 float(scale), conv3->biasBlob->diff_gpu, 	  conv3->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, conv2g->filtersBlob->count(), float(scale), conv2g->filtersBlob->diff_gpu, conv2g->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv2g->biasBlob->count(), 	 float(scale), conv2g->biasBlob->diff_gpu, 	  conv2g->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, conv1->filtersBlob->count(), float(scale), conv1->filtersBlob->diff_gpu, conv1->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv1->biasBlob->count(), 	 float(scale), conv1->biasBlob->diff_gpu,    conv1->biasBlob->data_gpu);
}

void AlexNetwork_t::SaveNetParams(int epoch) {
	cudaSetDevice(gpu_id);
	stringstream f1; f1 << net_name << "_c1_weight_e" << epoch << ".mat";
	conv1->filtersBlob->save_cpu_data_and_diff_to_mat(f1.str().c_str());
	stringstream f2; f2 << net_name << "_c1_bias_e" << epoch << ".mat";
	conv1->biasBlob->save_cpu_data_and_diff_to_mat(f2.str().c_str());

	stringstream f3; f3 << net_name << "_c2g_weight_e" << epoch << ".mat";
	conv2g->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
	stringstream f4; f4 << net_name << "_c2g_bias_e" << epoch << ".mat";
	conv2g->biasBlob->save_cpu_data_and_diff_to_mat(f4.str().c_str());

	stringstream f5; f5 << net_name << "_c3_weight_e" << epoch << ".mat";
	conv3->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
	stringstream f6; f6 << net_name << "_c3_bias_e" << epoch << ".mat";
	conv3->biasBlob->save_cpu_data_and_diff_to_mat(f6.str().c_str());

	stringstream f31; f31 << net_name << "_c4g_weight_e" << epoch << ".mat";
	conv4g->filtersBlob->save_cpu_data_and_diff_to_mat(f31.str().c_str());
	stringstream f41; f41 << net_name << "_c4g_bias_e" << epoch << ".mat";
	conv4g->biasBlob->save_cpu_data_and_diff_to_mat(f41.str().c_str());

	stringstream f311; f311 << net_name << "_c5g_weight_e" << epoch << ".mat";
	conv5g->filtersBlob->save_cpu_data_and_diff_to_mat(f311.str().c_str());
	stringstream f411; f411 << net_name << "_c5g_bias_e" << epoch << ".mat";
	conv5g->biasBlob->save_cpu_data_and_diff_to_mat(f411.str().c_str());

	stringstream f7; f7 << net_name << "_fc6_weight_e" << epoch << ".mat";
	fc6->filtersBlob->save_cpu_data_and_diff_to_mat(f7.str().c_str());
	stringstream f8; f8 << net_name << "_fc6_bias_e" << epoch << ".mat";
	fc6->biasBlob->save_cpu_data_and_diff_to_mat(f8.str().c_str());

	stringstream f71; f71<< net_name << "_fc7_weight_e" << epoch << ".mat";
	fc7->filtersBlob->save_cpu_data_and_diff_to_mat(f71.str().c_str());
	stringstream f81; f81 << net_name << "_fc7_bias_e" << epoch << ".mat";
	fc7->biasBlob->save_cpu_data_and_diff_to_mat(f81.str().c_str());

	stringstream f711; f711<< net_name << "_fc8_weight_e" << epoch << ".mat";
	fc8->filtersBlob->save_cpu_data_and_diff_to_mat(f711.str().c_str());
	stringstream f811; f811 << net_name << "_fc8_bias_e" << epoch << ".mat";
	fc8->biasBlob->save_cpu_data_and_diff_to_mat(f811.str().c_str());

}

void AlexNetwork_t::CopyNetParamsFrom(const AlexNetwork_t *other) {
	CopyBlobData_gpu(other->fc8->filtersBlob, 	other->gpu_id, fc8->filtersBlob,   gpu_id);
	CopyBlobData_gpu(other->fc8->biasBlob, 		other->gpu_id, fc8->biasBlob, 	   gpu_id);
	CopyBlobData_gpu(other->fc7->filtersBlob, 	other->gpu_id, fc7->filtersBlob,   gpu_id);
	CopyBlobData_gpu(other->fc7->biasBlob, 		other->gpu_id, fc7->biasBlob, 	   gpu_id);
	CopyBlobData_gpu(other->fc6->filtersBlob, 	other->gpu_id, fc6->filtersBlob,   gpu_id);
	CopyBlobData_gpu(other->fc6->biasBlob, 		other->gpu_id, fc6->biasBlob, 	   gpu_id);
	CopyBlobData_gpu(other->conv5g->filtersBlob, other->gpu_id, conv5g->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv5g->biasBlob, 	other->gpu_id, conv5g->biasBlob,    gpu_id);
	CopyBlobData_gpu(other->conv4g->filtersBlob, other->gpu_id, conv4g->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv4g->biasBlob, 	other->gpu_id, conv4g->biasBlob,    gpu_id);
	CopyBlobData_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv3->biasBlob, 	other->gpu_id, conv3->biasBlob,	   gpu_id);
	CopyBlobData_gpu(other->conv2g->filtersBlob, other->gpu_id, conv2g->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv2g->biasBlob, 	other->gpu_id, conv2g->biasBlob,    gpu_id);
	CopyBlobData_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv1->biasBlob, 	other->gpu_id, conv1->biasBlob,    gpu_id);
}

void AlexNetwork_t::AddNetParamsDiffFrom(const AlexNetwork_t *other) {
	AddBlobDiff_gpu(other->fc8->filtersBlob, 	other->gpu_id, fc8->filtersBlob,   gpu_id);
	AddBlobDiff_gpu(other->fc8->biasBlob, 		other->gpu_id, fc8->biasBlob, 	   gpu_id);
	AddBlobDiff_gpu(other->fc7->filtersBlob, 	other->gpu_id, fc7->filtersBlob,   gpu_id);
	AddBlobDiff_gpu(other->fc7->biasBlob, 		other->gpu_id, fc7->biasBlob, 	   gpu_id);
	AddBlobDiff_gpu(other->fc6->filtersBlob, 	other->gpu_id, fc6->filtersBlob,   gpu_id);
	AddBlobDiff_gpu(other->fc6->biasBlob, 		other->gpu_id, fc6->biasBlob, 	   gpu_id);
	AddBlobDiff_gpu(other->conv5g->filtersBlob, other->gpu_id, conv5g->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv5g->biasBlob, 	other->gpu_id, conv5g->biasBlob,    gpu_id);
	AddBlobDiff_gpu(other->conv4g->filtersBlob, other->gpu_id, conv4g->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv4g->biasBlob, 	other->gpu_id, conv4g->biasBlob,    gpu_id);
	AddBlobDiff_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv3->biasBlob,    other->gpu_id, conv3->biasBlob,    gpu_id);
	AddBlobDiff_gpu(other->conv2g->filtersBlob, other->gpu_id, conv2g->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv2g->biasBlob,    other->gpu_id, conv2g->biasBlob, 	  gpu_id);
	AddBlobDiff_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv1->biasBlob,    other->gpu_id, conv1->biasBlob, 	  gpu_id);
}

void AlexNetwork_t::ClearNetParamsDiff() {
	cudaSetDevice(gpu_id);
	gpu_set(fc8->filtersBlob->count(),   0, fc8->filtersBlob->diff_gpu);
	gpu_set(fc8->biasBlob->count(),      0, fc8->biasBlob->diff_gpu);
	gpu_set(fc7->filtersBlob->count(),   0, fc7->filtersBlob->diff_gpu);
	gpu_set(fc7->biasBlob->count(),      0, fc7->biasBlob->diff_gpu);
	gpu_set(fc6->filtersBlob->count(),   0, fc6->filtersBlob->diff_gpu);
	gpu_set(fc6->biasBlob->count(),      0, fc6->biasBlob->diff_gpu);
	gpu_set(conv5g->filtersBlob->count(), 0, conv5g->filtersBlob->diff_gpu);
	gpu_set(conv5g->biasBlob->count(),    0, conv5g->biasBlob->diff_gpu);
	gpu_set(conv4g->filtersBlob->count(), 0, conv4g->filtersBlob->diff_gpu);
	gpu_set(conv4g->biasBlob->count(),    0, conv4g->biasBlob->diff_gpu);
	gpu_set(conv3->filtersBlob->count(), 0, conv3->filtersBlob->diff_gpu);
	gpu_set(conv3->biasBlob->count(), 	 0, conv3->biasBlob->diff_gpu);
	gpu_set(conv2g->filtersBlob->count(), 0, conv2g->filtersBlob->diff_gpu);
	gpu_set(conv2g->biasBlob->count(),    0, conv2g->biasBlob->diff_gpu);
	gpu_set(conv1->filtersBlob->count(), 0, conv1->filtersBlob->diff_gpu);
	gpu_set(conv1->biasBlob->count(),    0, conv1->biasBlob->diff_gpu);
}
