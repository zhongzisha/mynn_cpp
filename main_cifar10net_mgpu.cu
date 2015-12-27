

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
struct thread_data_t
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

void thread_func(void *data_)
{
	thread_data_t *data = (thread_data_t *)data_;
	cudaSetDevice(data->net_gpu_id);
	// CUDA_CHECK( cudaMemcpy(data->net->batch_samples->data_gpu, data->batch_samples->data_cpu, data->batch_samples->count() * sizeof(float), cudaMemcpyHostToDevice) );
	// CUDA_CHECK( cudaMemcpy(data->net->batch_labels->data_gpu, data->batch_labels->data_cpu, data->batch_labels->count() * sizeof(float), cudaMemcpyHostToDevice) );
	float trn_loss, trn_acc;
	data->net->ForwardBackward(&trn_loss, &trn_acc);
	data->net->ComputeUpdateValue(data->lr_rate, data->momentum, data->weight_decay);
	// printf("gpuid[%d]: trn_loss=%.6f, trn_acc=%.6f\n", data->net_gpu_id, trn_loss, trn_acc);

	pthread_barrier_wait(&barr);
}


int main(int argc, char **argv) {
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

	thread_data_t thread_data[gpus.size()];
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
				ret_count = pthread_create(&(threads[i]), &pta, (void*(*)(void*))thread_func, (void*)(&(thread_data[i])));
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

