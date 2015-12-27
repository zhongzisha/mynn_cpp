
#include <glog/logging.h>
#include <pthread.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
using namespace std;

#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <boost/filesystem.hpp>
using namespace boost;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include "common.hpp"
#include "blob.hpp"
#include "common_layer.hpp"
#include "data_layer.hpp"
#include "conv_layer.hpp"
#include "loss_layer.hpp"
#include "network_cifar10.hpp"
#include "network_alex.hpp"

pthread_barrier_t barr;
struct cifar10_thread_data_t
{
public:
	Blob_t *batch_samples;
	Blob_t *batch_labels;
	Cifar10Network_t *net;
	int main_gpu_id;
	int net_gpu_id;
	float lr_rate;
	float momentum;
	float weight_decay;
};

void cifar10_do_slave(void *data_)
{
	cifar10_thread_data_t *data = (cifar10_thread_data_t *)data_;
	cudaSetDevice(data->net_gpu_id);
	// CUDA_CHECK( cudaMemcpy(data->net->batch_samples->data_gpu, data->batch_samples->data_cpu, data->batch_samples->count() * sizeof(float), cudaMemcpyHostToDevice) );
	// CUDA_CHECK( cudaMemcpy(data->net->batch_labels->data_gpu, data->batch_labels->data_cpu, data->batch_labels->count() * sizeof(float), cudaMemcpyHostToDevice) );
	float trn_loss, trn_acc;
	data->net->ForwardBackward(&trn_loss, &trn_acc);
	data->net->ComputeUpdateValue(data->lr_rate, data->momentum, data->weight_decay);
	// printf("gpuid[%d]: trn_loss=%.6f, trn_acc=%.6f\n", data->net_gpu_id, trn_loss, trn_acc);

	pthread_barrier_wait(&barr);
}

struct alexnet_thread_data_t
{
public:
	Blob_t *batch_samples;
	Blob_t *batch_labels;
	AlexNetwork_t *net;
	int main_gpu_id;
	int net_gpu_id;
	float lr_rate;
	float momentum;
	float weight_decay;
};

void alexnet_do_slave(void *data_)
{
	alexnet_thread_data_t *data = (alexnet_thread_data_t *)data_;
	cudaSetDevice(data->net_gpu_id);
	// CUDA_CHECK( cudaMemcpy(data->net->batch_samples->data_gpu, data->batch_samples->data_cpu, data->batch_samples->count() * sizeof(float), cudaMemcpyHostToDevice) );
	// CUDA_CHECK( cudaMemcpy(data->net->batch_labels->data_gpu, data->batch_labels->data_cpu, data->batch_labels->count() * sizeof(float), cudaMemcpyHostToDevice) );
	float trn_loss, trn_acc;
	data->net->ForwardBackward(&trn_loss, &trn_acc);
	data->net->ComputeUpdateValue(data->lr_rate, data->momentum, data->weight_decay);
	// printf("gpuid[%d]: trn_loss=%.6f, trn_acc=%.6f\n", data->net_gpu_id, trn_loss, trn_acc);

	pthread_barrier_wait(&barr);
}


int main_test_data_layer_ok(int argc, char *argv[]) {
	if(argc != 12) {
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

	Blob_t *batch_samples = new Blob_t();
	Blob_t *batch_labels = new Blob_t();
	DataLayerParameter_t *data_param = new DataLayerParameter_t();
	data_param->backend = "lmdb";
	data_param->batch_size = trn_batch_size;
	data_param->source = trn_db_filename;
	data_param->mean_file = mean_file;
	DataLayer_t *trn_data_layer = new DataLayer_t(data_param);
	trn_data_layer->Setup();
	printf("forward datalayer.\n");
	trn_data_layer->Forward_cpu(batch_samples, batch_labels);
	printf("forward datalayer(done).\n");

	batch_samples->print_cpu_data(100);
	batch_labels->print_cpu_data(100);

	delete data_param; data_param = NULL;
	delete batch_samples; batch_samples = NULL;
	delete batch_labels; batch_labels = NULL;
	delete trn_data_layer; trn_data_layer = NULL;
	return 0;
}

int main_test_multigpu_ok(int argc, char *argv[]) {
	if(argc != 12) {
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


	int main_gpu_id;
	cudaGetDevice(&main_gpu_id);
	printf("current gpu id: %d\n", main_gpu_id);

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	printf("number of manually-set gpus: %ld, total %d gpus.\n", gpus.size(), num_gpus);

	if(num_gpus >= gpus.size()) {
		printf("enable P2P: ");
		EnableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	} else {
		gpus.clear();
		gpus.push_back(main_gpu_id);
	}

	cudaSetDevice(main_gpu_id);

	vector<Cifar10Network_t *> trn_nets(gpus.size());
	for(int i = 0; i < gpus.size(); i++) {
		trn_nets[i] = NULL;
	}
	printf("initialize nets for each gpu ...\n");
	for(int i = 0; i < gpus.size(); i++)
	{
		printf("=========== gpu [%d] ==============\n", gpus[i]);
		cudaSetDevice(gpus[i]);
		trn_nets[i] = new Cifar10Network_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(trn_batch_size, "");
		trn_nets[i]->batch_labels->allocate_cpu_data();
	}
	printf("initialize nets for each gpu (done) ...\n");

	cudaSetDevice(main_gpu_id);

	pthread_t *threads;
	pthread_attr_t pta;
	threads = (pthread_t *) malloc(sizeof(pthread_t) * gpus.size());
	int ret_count = pthread_attr_init(&pta);
	cifar10_thread_data_t thread_data[gpus.size()];

	// prepare batch data, should use blocking queue
	Blob_t *batch_samples = new Blob_t(trn_batch_size, 3, 32, 32);
	Blob_t *batch_labels  = new Blob_t(trn_batch_size, 1, 1, 1);
	batch_samples->allocate_cpu_data();
	batch_labels->allocate_cpu_data();
	for(int n = 0; n < batch_samples->N; n++) {
		for(int c = 0; c < batch_samples->C; c++) {
			for(int h = 0; h < batch_samples->H; h++) {
				for(int w = 0; w < batch_samples->W; w++) {
					int index = (((n)*batch_samples->C+c)*batch_samples->H+h)*batch_samples->W + w;
					batch_samples->data_cpu[index] = (float)rand() / (float)RAND_MAX;
				}
			}
		}
		batch_labels->data_cpu[n] = n;
	}

	for(int i = 0; i < gpus.size(); i++) {
		thread_data[i].lr_rate = lr_rate;
		thread_data[i].momentum = momentum;
		thread_data[i].weight_decay = weight_decay;
		thread_data[i].main_gpu_id = main_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples;
		thread_data[i].batch_labels = batch_labels;

		ret_count = pthread_create(&threads[i], &pta, (void*(*)(void*))cifar10_do_slave, (void*)(&(thread_data[i])));
	}

	for(int i = 0; i < gpus.size(); i++) {
		ret_count = pthread_join(threads[i], NULL);
	}

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(main_gpu_id);
	delete batch_samples;
	delete batch_labels;

	if(num_gpus >= gpus.size()) {
		printf("disable P2P: ");
		DisableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	}
	cudaDeviceReset();
	return 0;
}

int main_mgpu_ok_loss_is_decreasing(int argc, char *argv[]) {
	if(argc != 12) {
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


	int main_gpu_id;
	cudaGetDevice(&main_gpu_id);
	printf("current gpu id: %d\n", main_gpu_id);

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	printf("number of manually-set gpus: %ld, total %d gpus.\n", gpus.size(), num_gpus);

	if(num_gpus >= gpus.size()) {
		printf("enable P2P: ");
		EnableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	} else {
		gpus.clear();
		gpus.push_back(main_gpu_id);
	}

	cudaSetDevice(main_gpu_id);

	vector<Cifar10Network_t *> trn_nets(gpus.size());
	vector<Blob_t *> batch_samples_slices(gpus.size());
	vector<Blob_t *> batch_labels_slices(gpus.size());
	vector<int> batch_sizes(gpus.size());
	for(int i = 0; i < gpus.size(); i++) {
		trn_nets[i] = NULL;
		batch_samples_slices[i] = NULL;
		batch_labels_slices[i] = NULL;
		batch_sizes[i] = 0;
	}
	printf("initialize nets for each gpu ...\n");
	for(int i = 0; i < gpus.size(); i++)
	{
		printf("=========== gpu [%d] ==============\n", gpus[i]);
		cudaSetDevice(gpus[i]);

		batch_samples_slices[i] = new Blob_t();
		batch_labels_slices[i] = new Blob_t();
		batch_sizes[i] = trn_batch_size / gpus.size();

		trn_nets[i] = new Cifar10Network_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(batch_sizes[i], "");
		trn_nets[i]->batch_labels->allocate_cpu_data();
	}
	printf("initialize nets for each gpu (done) ...\n");

	cudaSetDevice(main_gpu_id);

	Blob_t *trn_batch_samples = new Blob_t();
	Blob_t *trn_batch_labels = new Blob_t();
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	Blob_t *tst_batch_samples = new Blob_t();
	Blob_t *tst_batch_labels = new Blob_t();
	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	Cifar10Network_t *trn_net = new Cifar10Network_t("trn_net", main_gpu_id);
	trn_net->BuildNet(trn_batch_size, "");
	trn_net->batch_labels->allocate_cpu_data();
	Cifar10Network_t *tst_net = new Cifar10Network_t("tst_net", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");
	tst_net->batch_labels->allocate_cpu_data();

	pthread_t *threads;
	pthread_attr_t pta;
	threads = (pthread_t *) malloc(sizeof(pthread_t) * gpus.size());
	int ret_count = pthread_attr_init(&pta);
	cifar10_thread_data_t thread_data[gpus.size()];
	for(int i = 0; i < gpus.size(); i++) {
		thread_data[i].lr_rate = lr_rate;
		thread_data[i].momentum = momentum;
		thread_data[i].weight_decay = weight_decay;
		thread_data[i].main_gpu_id = main_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples_slices[i];
		thread_data[i].batch_labels = batch_labels_slices[i];
	}

	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_acc = 0.0f;
		tst_net->CopyNetParamsFrom(trn_net);
		for(int iter = 0; iter < floor(10000 / tst_batch_size); iter++) {
			tst_data_layer->Forward_cpu(tst_batch_samples, tst_batch_labels);
			tst_net->Forward(&tst_loss, &tst_acc);
		}

		// training net
		for(int iter = 0; iter < floor(50000 / trn_batch_size); iter++) {
			trn_data_layer->Forward_cpu_multi(batch_samples_slices, batch_labels_slices, batch_sizes);

			trn_net->ClearNetParamsDiff();

			// copy trn_net params into trn_nets_i
			for(int i = 0; i < gpus.size(); i++) {
				trn_nets[i]->CopyNetParamsFrom(trn_net);
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_create(&threads[i], &pta, (void*(*)(void*))cifar10_do_slave, (void*)(&(thread_data[i])));
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_join(threads[i], NULL);
			}

			cudaDeviceSynchronize();
			cudaSetDevice(main_gpu_id);
			// copy update values from each sub nets to the main trn_net
			for(int i = 0; i < gpus.size(); i++) {
				trn_net->AddNetParamsDiffFrom(trn_nets[i]);
			}
			trn_net->UpdateNet();
		}
	}

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(main_gpu_id);
	delete trn_batch_samples;
	delete trn_batch_labels;
	delete tst_batch_samples;
	delete tst_batch_labels;
	delete trn_net;
	delete tst_net;

	delete trn_data_param; trn_data_param = NULL;
	delete trn_data_layer; trn_data_layer = NULL;
	delete tst_data_param; tst_data_param = NULL;
	delete tst_data_layer; tst_data_layer = NULL;

	if(num_gpus >= gpus.size()) {
		printf("disable P2P: ");
		DisableP2P(gpus);
		printf("%s \n", cudaGetErrorString(cudaGetLastError()));
	}
	free(threads); threads = NULL;
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}

int main_test_memcpy(int argc, char **argv) {

	int N = 64;
	float *data_h = NULL;
	MallocHost((void **)&data_h, N * sizeof(float));
	for(int i = 0; i < N; i++) {
		data_h[i] = (float)rand() / (float)RAND_MAX;
	}

	cudaSetDevice(1);
	float *data_d = NULL;
	CUDA_CHECK( cudaMalloc((void **)&data_d, N * sizeof(float)) );
	cudaSetDevice(2);
	CUDA_CHECK( cudaMemset(data_d, 0, N * sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(data_d, data_h, N * sizeof(float), cudaMemcpyHostToDevice) );
	float *data_d_copy = NULL;
	CUDA_CHECK( cudaMalloc((void **)&data_d_copy, N * sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(data_d_copy, data_d, N * sizeof(float), cudaMemcpyDeviceToDevice) );

	float *data_h2 = new float[N];
	CUDA_CHECK( cudaMemcpy(data_h2, data_d_copy, N * sizeof(float), cudaMemcpyDeviceToHost) );
	bool isyes = true;
	for(int i = 0; i < N; i++) {
		if(abs(data_h[i] - data_h2[i]) > 1e-6) {
			isyes = false;
			break;
		}
	}
	printf("data_h %s data_h2\n", isyes ? "==" : "!=");

	cudaSetDevice(2);
	float *gpu2_data_d = NULL;
	CUDA_CHECK( cudaMalloc((void**)&gpu2_data_d, N *sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(gpu2_data_d, data_d, N * sizeof(float), cudaMemcpyDefault) );
	CUDA_CHECK( cudaMemcpy(data_h2, data_d_copy, N * sizeof(float), cudaMemcpyDeviceToHost) );
	isyes = true;
	for(int i = 0; i < N; i++) {
		if(abs(data_h[i] - data_h2[i]) > 1e-6) {
			isyes = false;
			break;
		}
	}
	printf("data_h %s data_h2\n", isyes ? "==" : "!=");

	cudaSetDevice(2);
	CUDA_CHECK( cudaFree(gpu2_data_d) );

	cudaSetDevice(1);
	CUDA_CHECK( cudaFree(data_d) );
	CUDA_CHECK( cudaFree(data_d_copy) );
	FreeHost(data_h);
	delete[] data_h2;
	cudaDeviceReset();
	return 0;
}

int main_cifar10_single_gpu_ok(int argc, char **argv) {
	if(argc != 12) {
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

	int main_gpu_id = 0;
	cudaSetDevice(main_gpu_id);
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	Cifar10Network_t *trn_net = new Cifar10Network_t("trn_net", main_gpu_id);
	trn_net->BuildNet(trn_batch_size, "");

	Cifar10Network_t *tst_net = new Cifar10Network_t("tst_net", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");

	int num_tst_iters = ceil(10000 / tst_batch_size);
	int num_trn_iters = ceil(50000 / trn_batch_size);
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		tst_net->CopyNetParamsFrom(trn_net);
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(tst_net->batch_samples, tst_net->batch_labels);
			tst_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;

		// training net
		float trn_loss = 0.0f, trn_loss_batch = 0.0f;
		float trn_acc  = 0.0f, trn_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network(trn_net->batch_samples, trn_net->batch_labels);
			trn_net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
			trn_loss += trn_loss_batch;
			trn_acc  += trn_acc_batch;
			trn_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);
			trn_net->UpdateNet();
		}
		trn_loss /= num_trn_iters;
		trn_acc  /= num_trn_iters;

		// update learning rate
		if((epoch != 0) && (epoch % lr_stepsize == 0))
		{
			lr_rate /= 10;
			trn_net->SaveNetParams(epoch);
		}
		printf("epoch[%d]: trn_loss=%.6f, trn_acc=%.6f, tst_loss=%.6f, tst_acc=%.6f\n",
				epoch, trn_loss, trn_acc, tst_loss, tst_acc);
	}

	delete trn_net;
	delete tst_net;

	delete trn_data_layer;
	delete tst_data_layer;
	delete trn_data_param;
	delete tst_data_param;

	cudaDeviceReset();
	return 0;
}

int main_cifar10_multi_gpu_ok(int argc, char **argv) {
	if(argc != 14) {
		LOG(FATAL) << ("Usage: <filename> main_gpu_id db_backend trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n");
		return -1;
	}
	int main_gpu_id = atoi(argv[1]);
	string db_backend = string(argv[2]);
	string trn_db_filename = string(argv[3]);
	string tst_db_filename = string(argv[4]);
	string mean_file = string(argv[5]);
	float lr_rate = atof(argv[6]);
	int lr_stepsize = atoi(argv[7]);
	float momentum = atof(argv[8]);
	float weight_decay = atof(argv[9]);
	int trn_batch_size = atoi(argv[10]);
	int tst_batch_size = atoi(argv[11]);
	int max_epoch_num = atoi(argv[12]);
	string gpu_ids_str = string(argv[13]);

	cudaSetDevice(main_gpu_id);
	LOG(INFO) << "current gpu id: " << main_gpu_id;

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	LOG(INFO) << "number of manually-set gpus: " << gpus.size() <<
			"total " << num_gpus << " gpus.";

	if(num_gpus >= gpus.size()) {
		LOG(INFO) << ("enable P2P: ");
		EnableP2P(gpus);
	} else {
		gpus.clear();
		gpus.push_back(main_gpu_id);
	}

	if(trn_batch_size % gpus.size() != 0) {
		LOG(FATAL) << "trn_batch_size: " << trn_batch_size
				<< ", number of given gpus: " << gpus.size()
				<< ", trn_batch_size must be times of the number of given gpus.";
		return -1;
	}

	cudaSetDevice(main_gpu_id);
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = db_backend;
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = db_backend;
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	cudaSetDevice(main_gpu_id);
	Cifar10Network_t *tst_net = new Cifar10Network_t("tst_net", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");
	// tst_net->SaveNetParams(0);

	vector<Cifar10Network_t *> trn_nets(gpus.size());
	vector<Blob_t *> batch_samples_slices(gpus.size());
	vector<Blob_t *> batch_labels_slices(gpus.size());
	vector<int> batch_sizes(gpus.size());
	for(int i = 0; i < gpus.size(); i++) {
		trn_nets[i] = NULL;
		batch_samples_slices[i] = NULL;
		batch_labels_slices[i] = NULL;
		batch_sizes[i] = 0;
	}
	LOG(INFO) << ("initialize nets for each gpu ...\n");
	for(int i = 0; i < gpus.size(); i++)
	{
		LOG(INFO) << "gpu[" <<  gpus[i] << "]:\n";
		cudaSetDevice(main_gpu_id);
		batch_sizes[i] = trn_batch_size / gpus.size();
		trn_nets[i] = new Cifar10Network_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(batch_sizes[i], "");

		batch_samples_slices[i] = trn_nets[i]->batch_samples;
		batch_labels_slices[i] = trn_nets[i]->batch_labels;
	}
	LOG(INFO) << ("initialize nets for each gpu (done) ...\n");

	pthread_t *threads;
	pthread_attr_t pta;
	threads = (pthread_t *) malloc(sizeof(pthread_t) * gpus.size());
	int ret_count = pthread_attr_init(&pta);
	pthread_attr_setdetachstate(&pta, PTHREAD_CREATE_JOINABLE);
	pthread_barrier_init(&barr, NULL, gpus.size());

	cifar10_thread_data_t thread_data[gpus.size()];
	for(int i = 0; i < gpus.size(); i++) {
		thread_data[i].lr_rate = lr_rate;
		thread_data[i].momentum = momentum;
		thread_data[i].weight_decay = weight_decay;
		thread_data[i].main_gpu_id = main_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples_slices[i];
		thread_data[i].batch_labels = batch_labels_slices[i];
	}

	int num_tst_iters = ceil(10000 / tst_batch_size);
	int num_trn_iters = ceil(50000 / trn_batch_size);
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(tst_net->batch_samples, tst_net->batch_labels);
			tst_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc  += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;
		LOG(INFO) << "epoch[" << epoch << "]: tst_loss=" << tst_loss << ", tst_acc=" << tst_acc << "\n";

		// training net
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network_multi(gpus, batch_sizes, batch_samples_slices, batch_labels_slices);

			// copy trn_net params into trn_nets_i
			for(int i = 0; i < gpus.size(); i++) {
				trn_nets[i]->CopyNetParamsFrom(tst_net);
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_create(&(threads[i]), &pta, (void*(*)(void*))cifar10_do_slave, (void*)(&(thread_data[i])));
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_join(threads[i], NULL);
			}

			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			tst_net->ClearNetParamsDiff();
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			for(int i = 0; i < gpus.size(); i++) {
				tst_net->AddNetParamsDiffFrom(trn_nets[i]);
			}
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			tst_net->UpdateNet(-1.0f / (gpus.size()));
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			// update learning rate
			if((epoch != 0) && (epoch % lr_stepsize == 0))
			{
				lr_rate /= 10;
				// tst_net->SaveNetParams(epoch);
			}
		}
	}

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(main_gpu_id);
	batch_samples_slices.clear();
	batch_labels_slices.clear();

	delete tst_net;

	delete trn_data_param; trn_data_param = NULL;
	delete trn_data_layer; trn_data_layer = NULL;
	delete tst_data_param; tst_data_param = NULL;
	delete tst_data_layer; tst_data_layer = NULL;

	if(num_gpus >= gpus.size()) {
		LOG(INFO) << ("disable P2P: ");
		DisableP2P(gpus);
	}

	pthread_barrier_destroy(&barr);
	ret_count = pthread_attr_destroy(&pta);
	free(threads); threads = NULL;
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}

int main_test_conv_wigh_group_seems_ok(int argc, char **argv) {

	cudaStream_t curand_stream;
	curandRngType_t curand_rngtype;
	curandGenerator_t curand_generator;
	cublasHandle_t cublas_handle;
	CUDA_CHECK( cudaStreamCreate(&curand_stream) );
	curand_rngtype = CURAND_RNG_PSEUDO_DEFAULT;
	CURAND_CHECK( curandCreateGenerator(&curand_generator, curand_rngtype) );
	CURAND_CHECK( curandSetStream(curand_generator, curand_stream) );
	CUBLAS_CHECK( cublasCreate(&cublas_handle) );

	Blob_t *batch_samples = new Blob_t(16, 3, 227, 227);
	Blob_t *batch_labels = new Blob_t(16, 1, 1, 1);
	batch_samples->allocate_gpu_data();
	batch_samples->allocate_gpu_diff();
	batch_labels->allocate_gpu_data();
	batch_labels->allocate_cpu_data();

	CURAND_CHECK( curandGenerateNormal(curand_generator, batch_samples->data_gpu, batch_samples->count(), (float)0.0f, (float)0.1f) );
	for(int i=0; i<batch_labels->count(); i++) {
		batch_labels->data_cpu[i] = i%10;
	}
	batch_labels->data_to_gpu();

	printf("conv1 setup.\n");
	Blob_t *conv1_top = new Blob_t();
	ConvolutionParameter_t *conv1_params = new ConvolutionParameter_t();
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
	ConvolutionLayer_t *conv1 = new ConvolutionLayer_t(conv1_params);
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv1->filtersBlob->data_gpu, conv1->filtersBlob->count(), (float)0.0f, (float)0.0001f) );
	gpu_set(conv1->biasBlob->count(), 0, conv1->biasBlob->data_gpu);
	conv1->Setup(batch_samples, conv1_top);

	printf("mp1 setup.\n");
	Blob_t *mp1_top = new Blob_t();
	PoolingParameter_t *mp1_params = new PoolingParameter_t();
	mp1_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	mp1_params->poolsize_h = 3;
	mp1_params->poolsize_w = 3;
	mp1_params->pad_h = 0;
	mp1_params->pad_w = 0;
	mp1_params->stride_h = 2;
	mp1_params->stride_w = 2;
	PoolingLayer_t *mp1 = new PoolingLayer_t(mp1_params);
	mp1->Setup(conv1_top, mp1_top);

	printf("conv2g setup.\n");
	Blob_t *conv2g_top = new Blob_t();
	ConvolutionWithGroupParameter_t *conv2g_params = new ConvolutionWithGroupParameter_t();
	conv2g_params->group = 2;
	conv2g_params->filter_N = 96;
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
	ConvolutionWithGroupLayer_t *conv2g = new ConvolutionWithGroupLayer_t(conv2g_params);
	printf("init\n");
	CURAND_CHECK( curandGenerateNormal(curand_generator, conv2g->filtersBlob->data_gpu, conv2g->filtersBlob->count(), (float)0.0f, (float)0.01f) );
	gpu_set(conv2g->biasBlob->count(), 0, conv2g->biasBlob->data_gpu);
	conv2g->Setup(mp1_top, conv2g_top);

	printf("mp2 setup.\n");
	Blob_t *mp2_top = new Blob_t();
	PoolingParameter_t *mp2_params = new PoolingParameter_t();
	mp2_params->cudnn_pooling_mode = CUDNN_POOLING_MAX;
	mp2_params->poolsize_h = 3;
	mp2_params->poolsize_w = 3;
	mp2_params->pad_h = 0;
	mp2_params->pad_w = 0;
	mp2_params->stride_h = 2;
	mp2_params->stride_w = 2;
	PoolingLayer_t *mp2 = new PoolingLayer_t(mp1_params);
	mp2->Setup(conv2g_top, mp2_top);

	conv1->Forward(batch_samples, conv1_top);
	printf("conv1: (%d, %d, %d, %d)\n", conv1_top->N, conv1_top->C, conv1_top->H, conv1_top->W);

	mp1->Forward(conv1_top, mp1_top);
	printf("mp1: (%d, %d, %d, %d)\n", mp1_top->N, mp1_top->C, mp1_top->H, mp1_top->W);

	conv2g->Forward(mp1_top, conv2g_top);
	printf("conv2g_top: (%d, %d, %d, %d)\n", conv2g_top->N, conv2g_top->C, conv2g_top->H, conv2g_top->W);

	mp2->Forward(conv2g_top, mp2_top);
	printf("mp2: (%d, %d, %d, %d)\n", mp2_top->N, mp2_top->C, mp2_top->H, mp2_top->W);

	mp2->Backward(mp2_top, conv2g_top);

	conv2g->Backward(conv2g_top, mp1_top);

	mp1->Backward(mp1_top, conv1_top);

	conv1->Backward(conv1_top, batch_samples);


	delete conv1;
	delete conv1_top;
	delete conv1_params;
	delete mp1;
	delete mp1_top;
	delete mp1_params;
	delete conv2g;
	delete conv2g_top;
	delete conv2g_params;
	delete mp2;
	delete mp2_top;
	delete mp2_params;

	CURAND_CHECK( curandDestroyGenerator(curand_generator) );
	CUDA_CHECK( cudaStreamDestroy(curand_stream) );
	CUBLAS_CHECK( cublasDestroy(cublas_handle) );
	return 0;
}


// test alex net
int main_alex_net_single_gpu(int argc, char **argv) {
	if(argc != 14) {
		LOG(FATAL) << "Usage: <filename> main_gpu_id db_backend trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n";
		return -1;
	}
	int main_gpu_id = atoi(argv[1]);
	string db_backend = string(argv[2]);
	string trn_db_filename = string(argv[3]);
	string tst_db_filename = string(argv[4]);
	string mean_file = string(argv[5]);
	float lr_rate = atof(argv[6]);
	int lr_stepsize = atoi(argv[7]);
	float momentum = atof(argv[8]);
	float weight_decay = atof(argv[9]);
	int trn_batch_size = atoi(argv[10]);
	int tst_batch_size = atoi(argv[11]);
	int max_epoch_num = atoi(argv[12]);
	string gpu_ids_str = string(argv[13]);

	cudaSetDevice(main_gpu_id);
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = db_backend;
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 227;
	trn_data_param->scale = 1.0f;
	trn_data_param->mirror = true;
	trn_data_param->has_mean_file = true;
	trn_data_param->phase = "train";
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = db_backend;
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 227;
	tst_data_param->scale = 1.0f;
	tst_data_param->mirror = false;
	tst_data_param->has_mean_file = true;
	tst_data_param->phase = "test";
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	AlexNetwork_t *trn_net = new AlexNetwork_t("alexnet_trn", main_gpu_id);
	trn_net->BuildNet(trn_batch_size, "");

	AlexNetwork_t *tst_net = new AlexNetwork_t("alexnet_tst", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");

	int num_tst_iters = ceil(50000 / tst_batch_size);
	int num_trn_iters = ceil(1281167 / trn_batch_size);
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		tst_net->CopyNetParamsFrom(trn_net);
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(tst_net->batch_samples, tst_net->batch_labels);
			tst_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;
		LOG(INFO) << "epoch[" << epoch << "]: tst_loss=" << tst_loss << ", tst_acc= " << tst_acc;

		// training net
		float trn_loss = 0.0f, trn_loss_batch = 0.0f;
		float trn_acc  = 0.0f, trn_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network(trn_net->batch_samples, trn_net->batch_labels);
			trn_net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
			trn_loss += trn_loss_batch;
			trn_acc  += trn_acc_batch;
			trn_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);
			trn_net->UpdateNet();
		}
		trn_loss /= num_trn_iters;
		trn_acc  /= num_trn_iters;

		// update learning rate
		if((epoch != 0) && (epoch % lr_stepsize == 0))
		{
			lr_rate /= 10;
			// trn_net->SaveNetParams(epoch);
		}
		LOG(INFO) << "epoch[" << epoch << "]: trn_loss=" << trn_loss << ", trn_acc= " << trn_acc;
	}

	delete trn_net;
	delete tst_net;

	delete trn_data_layer;
	delete tst_data_layer;
	delete trn_data_param;
	delete tst_data_param;

	cudaDeviceReset();
	return 0;
}


int main_alexnet_multi_gpu(int argc, char **argv) {
	if(argc != 14) {
		LOG(FATAL) << ("Usage: <filename> main_gpu_id db_backend trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay trn_batch_size tst_batch_size max_epoch_num gpu_ids\n");
		return -1;
	}
	int main_gpu_id = atoi(argv[1]);
	string db_backend = string(argv[2]);
	string trn_db_filename = string(argv[3]);
	string tst_db_filename = string(argv[4]);
	string mean_file = string(argv[5]);
	float lr_rate = atof(argv[6]);
	int lr_stepsize = atoi(argv[7]);
	float momentum = atof(argv[8]);
	float weight_decay = atof(argv[9]);
	int trn_batch_size = atoi(argv[10]);
	int tst_batch_size = atoi(argv[11]);
	int max_epoch_num = atoi(argv[12]);
	string gpu_ids_str = string(argv[13]);

	cudaSetDevice(main_gpu_id);
	LOG(INFO) << "current gpu id: " << main_gpu_id;

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	LOG(INFO) << "number of manually-set gpus: " << gpus.size() <<
			"total " << num_gpus << " gpus.";

	if(num_gpus >= gpus.size()) {
		LOG(INFO) << ("enable P2P: ");
		EnableP2P(gpus);
	} else {
		gpus.clear();
		gpus.push_back(main_gpu_id);
	}

	if(trn_batch_size % gpus.size() != 0) {
		LOG(FATAL) << "trn_batch_size: " << trn_batch_size
				<< ", number of given gpus: " << gpus.size()
				<< ", trn_batch_size must be times of the number of given gpus.";
		return -1;
	}

	cudaSetDevice(main_gpu_id);
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = db_backend;
	trn_data_param->batch_size = trn_batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 227;
	trn_data_param->scale = 1.0f;
	trn_data_param->mirror = true;
	trn_data_param->has_mean_file = true;
	trn_data_param->phase = "train";
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = db_backend;
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 227;
	tst_data_param->scale = 1.0f;
	tst_data_param->mirror = false;
	tst_data_param->has_mean_file = true;
	tst_data_param->phase = "test";
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();

	cudaSetDevice(main_gpu_id);
	AlexNetwork_t *tst_net = new AlexNetwork_t("tst_net", main_gpu_id);
	tst_net->BuildNet(tst_batch_size, "");
	// tst_net->SaveNetParams(0);

	vector<AlexNetwork_t *> trn_nets(gpus.size());
	vector<Blob_t *> batch_samples_slices(gpus.size());
	vector<Blob_t *> batch_labels_slices(gpus.size());
	vector<int> batch_sizes(gpus.size());
	for(int i = 0; i < gpus.size(); i++) {
		trn_nets[i] = NULL;
		batch_samples_slices[i] = NULL;
		batch_labels_slices[i] = NULL;
		batch_sizes[i] = 0;
	}
	LOG(INFO) << ("initialize nets for each gpu ...\n");
	for(int i = 0; i < gpus.size(); i++)
	{
		LOG(INFO) << "gpu[" <<  gpus[i] << "]:\n";
		cudaSetDevice(main_gpu_id);
		batch_sizes[i] = trn_batch_size / gpus.size();
		trn_nets[i] = new AlexNetwork_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(batch_sizes[i], "");

		batch_samples_slices[i] = trn_nets[i]->batch_samples;
		batch_labels_slices[i] = trn_nets[i]->batch_labels;
	}
	LOG(INFO) << ("initialize nets for each gpu (done) ...\n");

	pthread_t *threads;
	pthread_attr_t pta;
	threads = (pthread_t *) malloc(sizeof(pthread_t) * gpus.size());
	int ret_count = pthread_attr_init(&pta);
	pthread_attr_setdetachstate(&pta, PTHREAD_CREATE_JOINABLE);
	pthread_barrier_init(&barr, NULL, gpus.size());

	alexnet_thread_data_t thread_data[gpus.size()];
	for(int i = 0; i < gpus.size(); i++) {
		thread_data[i].lr_rate = lr_rate;
		thread_data[i].momentum = momentum;
		thread_data[i].weight_decay = weight_decay;
		thread_data[i].main_gpu_id = main_gpu_id;
		thread_data[i].net = trn_nets[i];
		thread_data[i].net_gpu_id = gpus[i];
		thread_data[i].batch_samples = batch_samples_slices[i];
		thread_data[i].batch_labels = batch_labels_slices[i];
	}

	int num_tst_iters = ceil(50000 / tst_batch_size);
	int num_trn_iters = ceil(1281167 / trn_batch_size);
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {

		// testing net
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(tst_net->batch_samples, tst_net->batch_labels);
			tst_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc  += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;
		LOG(INFO) << "epoch[" << epoch << "]: tst_loss=" << tst_loss << ", tst_acc=" << tst_acc << "\n";

		// training net
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network_multi(gpus, batch_sizes, batch_samples_slices, batch_labels_slices);

			// copy trn_net params into trn_nets_i
			for(int i = 0; i < gpus.size(); i++) {
				trn_nets[i]->CopyNetParamsFrom(tst_net);
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_create(&(threads[i]), &pta, (void*(*)(void*))alexnet_do_slave, (void*)(&(thread_data[i])));
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_join(threads[i], NULL);
			}

			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			tst_net->ClearNetParamsDiff();
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			for(int i = 0; i < gpus.size(); i++) {
				tst_net->AddNetParamsDiffFrom(trn_nets[i]);
			}
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			tst_net->UpdateNet(-1.0f / (gpus.size()));
			cudaDeviceSynchronize();

			cudaSetDevice(main_gpu_id);
			// update learning rate
			if((epoch != 0) && (epoch % lr_stepsize == 0))
			{
				lr_rate /= 10;
				// tst_net->SaveNetParams(epoch);
			}
		}
	}

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(main_gpu_id);
	batch_samples_slices.clear();
	batch_labels_slices.clear();

	delete tst_net;

	delete trn_data_param; trn_data_param = NULL;
	delete trn_data_layer; trn_data_layer = NULL;
	delete tst_data_param; tst_data_param = NULL;
	delete tst_data_layer; tst_data_layer = NULL;

	if(num_gpus >= gpus.size()) {
		LOG(INFO) << ("disable P2P: ");
		DisableP2P(gpus);
	}

	pthread_barrier_destroy(&barr);
	ret_count = pthread_attr_destroy(&pta);
	free(threads); threads = NULL;
	cudaDeviceReset();
	return 0;
}

int main(int argc, char **argv) {
	// main_cifar10_multi_gpu_ok(argc, argv);
	main_alex_net_single_gpu(argc, argv);
	// main_alexnet_multi_gpu(argc, argv);
	return 0;
}
