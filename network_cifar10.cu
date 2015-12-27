#include "network_cifar10.hpp"


Cifar10Network_t::Cifar10Network_t(string net_name_, int gpu_id_) {
	net_name = net_name_;
	gpu_id = gpu_id_;
	curand_stream = NULL;
	curand_generator = NULL;
	curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
	cublas_handle = NULL;

	batch_samples = NULL;
	batch_labels = NULL;

	conv1 = NULL;
	conv1_top = NULL;
	conv1_params = NULL;
	relu1 = NULL;
	relu1_top = NULL;
	relu1_params = NULL;
	mp1 = NULL;
	mp1_top = NULL;
	mp1_params = NULL;
	conv2 = NULL;
	conv2_top = NULL;
	conv2_params = NULL;
	relu2 = NULL;
	relu2_top = NULL;
	relu2_params = NULL;
	mp2 = NULL;
	mp2_top = NULL;
	mp2_params = NULL;
	conv3 = NULL;
	conv3_top = NULL;
	conv3_params = NULL;
	relu3 = NULL;
	relu3_top = NULL;
	relu3_params = NULL;
	mp3 = NULL;
	mp3_top = NULL;
	mp3_params = NULL;
	ip1 = NULL;
	ip1_top = NULL;
	ip1_params = NULL;
	//		sm1 = NULL;
	//		sm1_top = NULL;
	//		sm1_params = NULL;
	//		mlr1 = NULL;
	//		mlr1_top = NULL;
	//		mlr1_params = NULL;
	sml1 = NULL;
	sml1_top = NULL;
	sml1_params = NULL;

	//		argmax1 = NULL;
	//		argmax1_top = NULL;
	//		argmax1_params = NULL;

	accuracy1 = NULL;
	accuracy1_top = NULL;
	accuracy1_params = NULL;

	conv1_filtersBlob_old = NULL;
	conv1_biasBlob_old = NULL;
	conv2_filtersBlob_old = NULL;
	conv2_biasBlob_old = NULL;
	conv3_filtersBlob_old = NULL;
	conv3_biasBlob_old = NULL;
	ip1_filtersBlob_old = NULL;
	ip1_biasBlob_old = NULL;

}

Cifar10Network_t::~Cifar10Network_t() {
	DestroyNet();
}

void Cifar10Network_t::DestroyNet() {

	cudaSetDevice(gpu_id);

	delete batch_samples; batch_samples = NULL;
	delete batch_labels; batch_labels = NULL;

	delete conv1; conv1 = NULL;
	delete relu1; relu1 = NULL;
	delete mp1; mp1 = NULL;
	delete conv2; conv2 = NULL;
	delete relu2; relu2 = NULL;
	delete mp2; mp2 = NULL;
	delete conv3; conv3 = NULL;
	delete relu3; relu3 = NULL;
	delete mp3; mp3 = NULL;
	delete ip1; ip1 = NULL;
	//		delete sm1; sm1 = NULL;
	//		delete argmax1; argmax1 = NULL;
	//		delete mlr1; mlr1 = NULL;
	delete sml1; sml1 = NULL;
	delete accuracy1; accuracy1 = NULL;

	delete conv1_top; conv1_top = NULL;
	delete relu1_top; relu1_top = NULL;
	delete mp1_top; mp1_top = NULL;
	delete conv2_top; conv2_top = NULL;
	delete relu2_top; relu2_top = NULL;
	delete mp2_top; mp2_top = NULL;
	delete conv3_top; conv3_top = NULL;
	delete relu3_top; relu3_top = NULL;
	delete mp3_top; mp3_top = NULL;
	delete ip1_top; ip1_top = NULL;
	//		delete sm1_top; sm1_top = NULL;
	//		delete argmax1_top; argmax1_top = NULL;
	//		delete mlr1_top; mlr1_top = NULL;
	delete sml1_top; sml1_top = NULL;
	delete accuracy1_top; accuracy1_top = NULL;

	delete conv1_params; conv1_params = NULL;
	delete relu1_params; relu1_params = NULL;
	delete mp1_params; mp1_params = NULL;
	delete conv2_params; conv2_params = NULL;
	delete relu2_params; relu2_params = NULL;
	delete mp2_params; mp2_params = NULL;
	delete conv3_params; conv3_params = NULL;
	delete relu3_params; relu3_params = NULL;
	delete mp3_params; mp3_params = NULL;
	delete ip1_params; ip1_params = NULL;
	//		delete sm1_params; sm1_params = NULL;
	//		delete argmax1_params; argmax1_params = NULL;
	//		delete mlr1_params; mlr1_params = NULL;
	delete sml1_params; sml1_params = NULL;
	delete accuracy1_params; accuracy1_params = NULL;

	delete conv1_filtersBlob_old; conv1_filtersBlob_old = NULL;
	delete conv1_biasBlob_old; conv1_biasBlob_old = NULL;
	delete conv2_filtersBlob_old; conv2_filtersBlob_old = NULL;
	delete conv2_biasBlob_old; conv2_biasBlob_old = NULL;
	delete conv3_filtersBlob_old; conv3_filtersBlob_old = NULL;
	delete conv3_biasBlob_old; conv3_biasBlob_old = NULL;
	delete ip1_filtersBlob_old; ip1_filtersBlob_old = NULL;
	delete ip1_biasBlob_old; ip1_biasBlob_old = NULL;

	CURAND_CHECK( curandDestroyGenerator(curand_generator) );
	CUDA_CHECK( cudaStreamDestroy(curand_stream) );
	CUBLAS_CHECK( cublasDestroy(cublas_handle) );
}

void Cifar10Network_t::BuildNet(int batch_size_, const string &net_params_file) {

	cudaSetDevice(gpu_id);

	CUDA_CHECK( cudaStreamCreate(&curand_stream) );
	curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
	CURAND_CHECK( curandCreateGenerator(&curand_generator, curand_rngtype) );
	CURAND_CHECK( curandSetStream(curand_generator, curand_stream) );
	CUBLAS_CHECK( cublasCreate(&cublas_handle) );

	batch_samples = new Blob_t(batch_size_, 3, 32, 32);
	batch_labels = new Blob_t(batch_size_, 1, 1, 1);
	batch_samples->allocate_gpu_data();
	batch_samples->allocate_gpu_diff();
	batch_labels->allocate_gpu_data();
	batch_labels->allocate_cpu_data();

	LOG(INFO) << "conv1 setup.\n";
	conv1_top = new Blob_t();
	conv1_params = new ConvolutionParameter_t();
	conv1_params->filter_N = 3;
	conv1_params->filter_C = 32;
	conv1_params->filter_H = 5;
	conv1_params->filter_W = 5;
	conv1_params->pad_h = 2;
	conv1_params->pad_w = 2;
	conv1_params->stride_h = 1;
	conv1_params->stride_w = 1;
	conv1_params->upscale_h = 1;
	conv1_params->upscale_w = 1;
	conv1_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv1 = new ConvolutionLayer_t(conv1_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->filtersBlob->data_gpu, conv1->filtersBlob->count(), (float)0.0f, (float)0.0001f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->biasBlob->data_gpu, conv1->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv1->biasBlob->count(), 0, conv1->biasBlob->data_gpu);
	conv1->Setup(batch_samples, conv1_top);


	LOG(INFO) << "relu1 setup.\n";
	relu1_top = new Blob_t();
	relu1_params = new ActivationParameter_t();
	relu1_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu1 = new ActivationLayer_t(relu1_params);
	relu1->Setup(conv1_top, relu1_top);

	LOG(INFO) << "mp1 setup.\n";
	mp1_top = new Blob_t();
	mp1_params = new PoolingParameter_t();
	mp1_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	mp1_params->poolsize_h = 3;
	mp1_params->poolsize_w = 3;
	mp1_params->pad_h = 0;
	mp1_params->pad_w = 0;
	mp1_params->stride_h = 2;
	mp1_params->stride_w = 2;
	mp1 = new PoolingLayer_t(mp1_params);
	mp1->Setup(relu1_top, mp1_top);

	LOG(INFO) << "conv2 setup.\n";
	conv2_top = new Blob_t();
	conv2_params = new ConvolutionParameter_t();
	conv2_params->filter_N = 32;
	conv2_params->filter_C = 32;
	conv2_params->filter_H = 5;
	conv2_params->filter_W = 5;
	conv2_params->pad_h = 2;
	conv2_params->pad_w = 2;
	conv2_params->stride_h = 1;
	conv2_params->stride_w = 1;
	conv2_params->upscale_h = 1;
	conv2_params->upscale_w = 1;
	conv2_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv2 = new ConvolutionLayer_t(conv2_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->filtersBlob->data_gpu, conv2->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv2->biasBlob->data_gpu, conv2->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv2->biasBlob->count(), 0, conv2->biasBlob->data_gpu);
	conv2->Setup(mp1_top, conv2_top);


	LOG(INFO) << "relu2 setup.\n";
	relu2_top = new Blob_t();
	relu2_params = new ActivationParameter_t();
	relu2_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu2 = new ActivationLayer_t(relu2_params);
	relu2->Setup(conv2_top, relu2_top);

	LOG(INFO) << "mp2 setup.\n";
	mp2_top = new Blob_t();
	mp2_params = new PoolingParameter_t();
	mp2_params->cudnn_pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
	mp2_params->poolsize_h = 3;
	mp2_params->poolsize_w = 3;
	mp2_params->pad_h = 0;
	mp2_params->pad_w = 0;
	mp2_params->stride_h = 2;
	mp2_params->stride_w = 2;
	mp2 = new PoolingLayer_t(mp2_params);
	mp2->Setup(relu2_top, mp2_top);

	LOG(INFO) << "conv3 setup.\n";
	conv3_top = new Blob_t();
	conv3_params = new ConvolutionParameter_t();
	conv3_params->filter_N = 32;
	conv3_params->filter_C = 64;
	conv3_params->filter_H = 5;
	conv3_params->filter_W = 5;
	conv3_params->pad_h = 2;
	conv3_params->pad_w = 2;
	conv3_params->stride_h = 1;
	conv3_params->stride_w = 1;
	conv3_params->upscale_h = 1;
	conv3_params->upscale_w = 1;
	conv3_params->cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
	conv3 = new ConvolutionLayer_t(conv3_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->filtersBlob->data_gpu, conv3->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, conv3->biasBlob->data_gpu, conv3->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv3->biasBlob->count(), 0, conv3->biasBlob->data_gpu);
	conv3->Setup(mp2_top, conv3_top);


	LOG(INFO) << "relu3 setup.\n";
	relu3_top = new Blob_t();
	relu3_params = new ActivationParameter_t();
	relu3_params->cudnn_activation_mode = CUDNN_ACTIVATION_RELU;
	relu3 = new ActivationLayer_t(relu3_params);
	relu3->Setup(conv3_top, relu3_top);

	LOG(INFO) << "mp3 setup.\n";
	mp3_top = new Blob_t();
	mp3_params = new PoolingParameter_t();
	mp3_params->cudnn_pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
	mp3_params->poolsize_h = 3;
	mp3_params->poolsize_w = 3;
	mp3_params->pad_h = 0;
	mp3_params->pad_w = 0;
	mp3_params->stride_h = 2;
	mp3_params->stride_w = 2;
	mp3 = new PoolingLayer_t(mp3_params);
	mp3->Setup(relu3_top, mp3_top);

	LOG(INFO) << "ip1 setup.\n";
	ip1_top = new Blob_t();
	ip1_params = new FullyConnectedParameter_t();
	ip1_params->hidden_size = 10;
	ip1 = new FullyConnectedLayer_t(ip1_params);
	ip1->Setup(mp3_top, ip1_top);
	CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->filtersBlob->data_gpu, ip1->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	// CURAND_CHECK( curandGenerateNormal(curand_generator, ip1->biasBlob->data_gpu, ip1->biasBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(ip1->biasBlob->count(), 0, ip1->biasBlob->data_gpu);

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

	LOG(INFO) << "sml1 setup.\n";
	sml1_top = new Blob_t();
	sml1_params = new SoftmaxWithLossParameter_t();
	sml1_params->cudnn_softmax_algo = CUDNN_SOFTMAX_ACCURATE;
	sml1_params->cudnn_softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
	sml1_params->has_ignore_label = false;
	sml1_params->ignore_label = -1;
	sml1_params->normalize = false;
	sml1 = new SoftmaxWithLossLayer_t(sml1_params);
	sml1->Setup(ip1_top, sml1_top);

	//		printf("argmax1 setup.\n");
	//		argmax1_top = new Blob_t();
	//		argmax1_params = new ArgMaxParameter_t();
	//		argmax1_params->out_max_val = true;
	//		argmax1_params->top_k = 1;
	//		argmax1 = new ArgMaxLayer_t(argmax1_params);
	//		argmax1->Setup(sml1_top, argmax1_top);

	LOG(INFO) << "accuracy1 setup.\n";
	accuracy1_top = new Blob_t();
	accuracy1_params = new AccuracyParameter_t();
	accuracy1_params->top_k = 1;
	accuracy1 = new AccuracyLayer_t(accuracy1_params);
	accuracy1->Setup(ip1_top, accuracy1_top);

	LOG(INFO) << "initialize old net params.\n";
	conv1_filtersBlob_old = new Blob_t(conv1->filtersBlob->N, conv1->filtersBlob->C, conv1->filtersBlob->H, conv1->filtersBlob->W);
	conv1_biasBlob_old = new Blob_t(conv1->biasBlob->N, conv1->biasBlob->C, conv1->biasBlob->H, conv1->biasBlob->W);
	conv1_filtersBlob_old->allocate_gpu_data();
	conv1_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv1_filtersBlob_old->data_gpu, conv1_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv1_biasBlob_old->data_gpu, conv1_biasBlob_old->count(), 0.0f, 0.0f);

	conv2_filtersBlob_old = new Blob_t(conv2->filtersBlob->N, conv2->filtersBlob->C, conv2->filtersBlob->H, conv2->filtersBlob->W);
	conv2_biasBlob_old = new Blob_t(conv2->biasBlob->N, conv2->biasBlob->C, conv2->biasBlob->H, conv2->biasBlob->W);
	conv2_filtersBlob_old->allocate_gpu_data();
	conv2_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv2_filtersBlob_old->data_gpu, conv2_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv2_biasBlob_old->data_gpu, conv2_biasBlob_old->count(), 0.0f, 0.0f);

	conv3_filtersBlob_old = new Blob_t(conv3->filtersBlob->N, conv3->filtersBlob->C, conv3->filtersBlob->H, conv3->filtersBlob->W);
	conv3_biasBlob_old = new Blob_t(conv3->biasBlob->N, conv3->biasBlob->C, conv3->biasBlob->H, conv3->biasBlob->W);
	conv3_filtersBlob_old->allocate_gpu_data();
	conv3_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, conv3_filtersBlob_old->data_gpu, conv3_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, conv3_biasBlob_old->data_gpu, conv3_biasBlob_old->count(), 0.0f, 0.0f);

	ip1_filtersBlob_old = new Blob_t(ip1->filtersBlob->N, ip1->filtersBlob->C, ip1->filtersBlob->H, ip1->filtersBlob->W);
	ip1_biasBlob_old = new Blob_t(ip1->biasBlob->N, ip1->biasBlob->C, ip1->biasBlob->H, ip1->biasBlob->W);
	ip1_filtersBlob_old->allocate_gpu_data();
	ip1_biasBlob_old->allocate_gpu_data();
	gpu_fill(NULL, ip1_filtersBlob_old->data_gpu, ip1_filtersBlob_old->count(), 0.0f, 0.0f);
	gpu_fill(NULL, ip1_biasBlob_old->data_gpu, ip1_biasBlob_old->count(), 0.0f, 0.0f);

	LOG(INFO) << "build net (done).\n";
}

void Cifar10Network_t::Forward(float *loss, float *accuracy) {
	cudaSetDevice(gpu_id);

	// printf("conv1 forward.\n");
	conv1->Forward(batch_samples, conv1_top);

	// printf("relu1 forward.\n");
	relu1->Forward(conv1_top, relu1_top);

	// printf("mp1 forward.\n");
	mp1->Forward(relu1_top, mp1_top);

	// printf("conv2 forward.\n");
	conv2->Forward(mp1_top, conv2_top);

	// printf("relu2 forward.\n");
	relu2->Forward(conv2_top, relu2_top);

	// printf("mp2 forward.\n");
	mp2->Forward(relu2_top, mp2_top);

	// printf("conv3 forward.\n");
	conv3->Forward(mp2_top, conv3_top);

	// printf("relu3 forward.\n");
	relu3->Forward(conv3_top, relu3_top);

	// printf("mp2 forward.\n");
	mp3->Forward(relu3_top, mp3_top);

	// printf("ip1 forward.\n");
	ip1->Forward(mp3_top, ip1_top);

	// printf("sm1 forward.\n");
	// sm1->Forward(ip1_top, sm1_top);

	// printf("mlr1 forward.\n");
	// mlr1->Forward(sm1_top, batch_labels, mlr1_top);

	// loss = mlr1_top->data_cpu[0];

	sml1->Forward(ip1_top, batch_labels, sml1_top, loss);

	//		argmax1->Forward_cpu(sml1_top, argmax1_top);

	accuracy1->Forward_cpu(ip1_top, batch_labels, accuracy1_top);

	*accuracy = accuracy1_top->data_cpu[0];

}

void Cifar10Network_t::Backward() {
	cudaSetDevice(gpu_id);

	// printf("sml1 backward.\n");
	sml1->Backward(sml1_top, batch_labels, ip1_top);

	// printf("mlr1 backward.\n");
	// mlr1->Backward(mlr1_top, batch_labels, sm1_top);

	// printf("sm1 backward.\n");
	// sm1->Backward(sm1_top, ip1_top);

	// printf("ip1 backward.\n");
	ip1->Backward(ip1_top, mp3_top);

	// printf("mp3 backward.\n");
	mp3->Backward(mp3_top, relu3_top);

	// printf("relu3 backward.\n");
	relu3->Backward(relu3_top, conv3_top);

	// printf("conv3 backward.\n");
	conv3->Backward(conv3_top, mp2_top);

	// printf("mp2 backward.\n");
	mp2->Backward(mp2_top, relu2_top);

	// printf("relu2 backward.\n");
	relu2->Backward(relu2_top, conv2_top);

	// printf("conv2 backward.\n");
	conv2->Backward(conv2_top, mp1_top);

	// printf("mp1 backward.\n");
	mp1->Backward(mp1_top, relu1_top);

	// printf("relu1 backward.\n");
	relu1->Backward(relu1_top, conv1_top);

	// printf("conv1 backward.\n");
	conv1->Backward(conv1_top, batch_samples);
}

void Cifar10Network_t::ForwardBackward(float *loss, float *accuracy) {
	Forward(loss, accuracy);
	Backward();
}

void Cifar10Network_t::ComputeUpdateValueSingle(Blob_t *param_gradient_blob, Blob_t *param_blob_old,
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
void Cifar10Network_t::ComputeUpdateValue(float lr_rate, float momentum, float weight_decay) {
	cudaSetDevice(gpu_id);
	ComputeUpdateValueSingle(conv3->filtersBlob, conv3_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv3->biasBlob, 	 conv3_biasBlob_old, 	lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv2->filtersBlob, conv2_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv2->biasBlob, 	 conv2_biasBlob_old, 	lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv1->filtersBlob, conv1_filtersBlob_old, lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(conv1->biasBlob, 	 conv1_biasBlob_old, 	lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(ip1->filtersBlob,   ip1_filtersBlob_old,   lr_rate, momentum, weight_decay);
	ComputeUpdateValueSingle(ip1->biasBlob,      ip1_biasBlob_old, 		lr_rate, momentum, weight_decay);
}

void Cifar10Network_t::UpdateNet(float scale) {
	cudaSetDevice(gpu_id);
	gpu_axpy(cublas_handle, conv3->filtersBlob->count(), float(scale), conv3->filtersBlob->diff_gpu, conv3->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv3->biasBlob->count(), 	 float(scale), conv3->biasBlob->diff_gpu, 	  conv3->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, conv2->filtersBlob->count(), float(scale), conv2->filtersBlob->diff_gpu, conv2->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv2->biasBlob->count(), 	 float(scale), conv2->biasBlob->diff_gpu, 	  conv2->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, conv1->filtersBlob->count(), float(scale), conv1->filtersBlob->diff_gpu, conv1->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, conv1->biasBlob->count(), 	 float(scale), conv1->biasBlob->diff_gpu,    conv1->biasBlob->data_gpu);
	gpu_axpy(cublas_handle, ip1->filtersBlob->count(),   float(scale), ip1->filtersBlob->diff_gpu,   ip1->filtersBlob->data_gpu);
	gpu_axpy(cublas_handle, ip1->biasBlob->count(), 	 float(scale), ip1->biasBlob->diff_gpu,      ip1->biasBlob->data_gpu);
}

void Cifar10Network_t::SaveNetParams(int epoch) {
	cudaSetDevice(gpu_id);
	stringstream f1; f1 << net_name << "_c1_weight_e" << epoch << ".mat";
	conv1->filtersBlob->save_cpu_data_and_diff_to_mat(f1.str().c_str());

	stringstream f2; f2 << net_name << "_c1_bias_e" << epoch << ".mat";
	conv1->biasBlob->save_cpu_data_and_diff_to_mat(f2.str().c_str());

	stringstream f3; f3 << net_name << "_c2_weight_e" << epoch << ".mat";
	conv2->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
	stringstream f4; f4 << net_name << "_c2_bias_e" << epoch << ".mat";
	conv2->biasBlob->save_cpu_data_and_diff_to_mat(f4.str().c_str());

	stringstream f5; f5 << net_name << "_c3_weight_e" << epoch << ".mat";
	conv3->filtersBlob->save_cpu_data_and_diff_to_mat(f3.str().c_str());
	stringstream f6; f6 << net_name << "_c3_bias_e" << epoch << ".mat";
	conv3->biasBlob->save_cpu_data_and_diff_to_mat(f6.str().c_str());

	stringstream f7; f7 << net_name << "_ip1_weight_e" << epoch << ".mat";
	ip1->filtersBlob->save_cpu_data_and_diff_to_mat(f7.str().c_str());
	stringstream f8; f8 << net_name << "_ip1_bias_e" << epoch << ".mat";
	ip1->biasBlob->save_cpu_data_and_diff_to_mat(f8.str().c_str());

}

void Cifar10Network_t::CopyNetParamsFrom(const Cifar10Network_t *other) {
	CopyBlobData_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv3->biasBlob, 	other->gpu_id, conv3->biasBlob,	   gpu_id);
	CopyBlobData_gpu(other->conv2->filtersBlob, other->gpu_id, conv2->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv2->biasBlob, 	other->gpu_id, conv2->biasBlob,    gpu_id);
	CopyBlobData_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
	CopyBlobData_gpu(other->conv1->biasBlob, 	other->gpu_id, conv1->biasBlob,    gpu_id);
	CopyBlobData_gpu(other->ip1->filtersBlob, 	other->gpu_id, ip1->filtersBlob,   gpu_id);
	CopyBlobData_gpu(other->ip1->biasBlob, 		other->gpu_id, ip1->biasBlob, 	   gpu_id);
}

void Cifar10Network_t::AddNetParamsDiffFrom(const Cifar10Network_t *other) {
	AddBlobDiff_gpu(other->conv3->filtersBlob, other->gpu_id, conv3->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv3->biasBlob,    other->gpu_id, conv3->biasBlob,    gpu_id);
	AddBlobDiff_gpu(other->conv2->filtersBlob, other->gpu_id, conv2->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv2->biasBlob,    other->gpu_id, conv2->biasBlob, 	  gpu_id);
	AddBlobDiff_gpu(other->conv1->filtersBlob, other->gpu_id, conv1->filtersBlob, gpu_id);
	AddBlobDiff_gpu(other->conv1->biasBlob,    other->gpu_id, conv1->biasBlob, 	  gpu_id);
	AddBlobDiff_gpu(other->ip1->filtersBlob,   other->gpu_id, ip1->filtersBlob,   gpu_id);
	AddBlobDiff_gpu(other->ip1->biasBlob,      other->gpu_id, ip1->biasBlob, 	  gpu_id);
}

void Cifar10Network_t::ClearNetParamsDiff() {
	cudaSetDevice(gpu_id);
	gpu_set(conv3->filtersBlob->count(), 0, conv3->filtersBlob->diff_gpu);
	gpu_set(conv3->biasBlob->count(), 	 0, conv3->biasBlob->diff_gpu);
	gpu_set(conv2->filtersBlob->count(), 0, conv2->filtersBlob->diff_gpu);
	gpu_set(conv2->biasBlob->count(),    0, conv2->biasBlob->diff_gpu);
	gpu_set(conv1->filtersBlob->count(), 0, conv1->filtersBlob->diff_gpu);
	gpu_set(conv1->biasBlob->count(),    0, conv1->biasBlob->diff_gpu);
	gpu_set(ip1->filtersBlob->count(),   0, ip1->filtersBlob->diff_gpu);
	gpu_set(ip1->biasBlob->count(),      0, ip1->biasBlob->diff_gpu);
}
