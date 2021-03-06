#include <glog/logging.h>
#include <pthread.h>
#include <mpi.h>

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

#include "hdf5.h"
#include "hdf5_hl.h"

#include "common.hpp"
#include "blob.hpp"
#include "common_layer.hpp"
#include "data_layer.hpp"
#include "conv_layer.hpp"
#include "loss_layer.hpp"
#include "network_cifar10.hpp"
#include "network_alex.hpp"

bool is_number_in_set(const int *arr, int arr_size, int number) {
	bool isin = false;
	for(int i=0; i<arr_size; i++) {
		if(arr[i] == number) {
			isin = true;
			break;
		}
	}
	return isin;
}

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

pthread_mutex_t mutex_trn = PTHREAD_MUTEX_INITIALIZER;
float trn_loss = 0.0f;
float trn_acc  = 0.0f;
void thread_func_fwbw(void *data_)
{
	thread_data_t *data = (thread_data_t *)data_;
	cudaSetDevice(data->net_gpu_id);
	// CUDA_CHECK( cudaMemcpy(data->net->batch_samples->data_gpu, data->batch_samples->data_cpu, data->batch_samples->count() * sizeof(float), cudaMemcpyHostToDevice) );
	// CUDA_CHECK( cudaMemcpy(data->net->batch_labels->data_gpu, data->batch_labels->data_cpu, data->batch_labels->count() * sizeof(float), cudaMemcpyHostToDevice) );
	float trn_loss_batch, trn_acc_batch;
	data->net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
	data->net->ComputeUpdateValue(data->lr_rate, data->momentum, data->weight_decay);
	// printf("gpuid[%d]: trn_loss=%.6f, trn_acc=%.6f\n", data->net_gpu_id, trn_loss, trn_acc);

	pthread_mutex_lock(&mutex_trn);
	trn_loss += trn_loss_batch;
	trn_acc  += trn_acc_batch;
	pthread_mutex_unlock(&mutex_trn);

	pthread_barrier_wait(&barr);
}

pthread_mutex_t mutex_tst = PTHREAD_MUTEX_INITIALIZER;
float tst_loss = 0.0f;
float tst_acc  = 0.0f;
void thread_func_fw(void *data_)
{
	thread_data_t *data = (thread_data_t *)data_;
	cudaSetDevice(data->net_gpu_id);
	// CUDA_CHECK( cudaMemcpy(data->net->batch_samples->data_gpu, data->batch_samples->data_cpu, data->batch_samples->count() * sizeof(float), cudaMemcpyHostToDevice) );
	// CUDA_CHECK( cudaMemcpy(data->net->batch_labels->data_gpu, data->batch_labels->data_cpu, data->batch_labels->count() * sizeof(float), cudaMemcpyHostToDevice) );
	float tst_loss_batch, tst_acc_batch;
	data->net->Forward(&tst_loss_batch, &tst_acc_batch);

	pthread_mutex_lock(&mutex_tst);
	tst_loss += tst_loss_batch;
	tst_acc  += tst_acc_batch;
	pthread_mutex_unlock(&mutex_tst);

	pthread_barrier_wait(&barr);
}


int main(int argc, char **argv) {
	// another thought
	// test MPI_Allreduce
	if(argc != 13) {
		printf("Usage: <filename> main_gpu_id db_backend trn_db_filename tst_db_filename mean_file lr_rate lr_stepsize momentum weight_decay batch_size max_epoch_num gpu_ids\n");
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
	int batch_size = atoi(argv[10]);
	int max_epoch_num = atoi(argv[11]);
	string gpu_ids_str = string(argv[12]);

	MPI_Init(&argc, &argv);
	int rank_id, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	char myname[MPI_MAX_PROCESSOR_NAME];
	int namelen;
	MPI_Get_processor_name(myname, &namelen);
	const int net_params_tags[8] = {31, 32, 33, 34, 35, 36, 37, 38};

	cudaSetDevice(main_gpu_id);
	std::cout << "current gpu id: " << main_gpu_id;

	vector<int> gpus;
	vector<string> strings;
	boost::split(strings, gpu_ids_str, boost::is_any_of(","));
	for (int i = 0; i < strings.size(); ++i) {
		gpus.push_back(boost::lexical_cast<int>(strings[i]));
	}
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	std::cout << "number of manually-set gpus: " << gpus.size() <<
			"total " << num_gpus << " gpus.";

	if(num_gpus >= gpus.size()) {
		std::cout << ("enable P2P: ");
		EnableP2P(gpus);
	} else {
		gpus.clear();
		gpus.push_back(main_gpu_id);
	}

	if(batch_size % gpus.size() != 0) {
		std::cout << "batch_size: " << batch_size
				<< ", number of given gpus: " << gpus.size()
				<< ", batch_size must be times of the number of given gpus.";
		return -1;
	}

	printf("init trn data_layer ...\n");
	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = db_backend;
	trn_data_param->batch_size = batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	trn_data_param->scale = 1.0f;
	trn_data_param->mirror = true;
	trn_data_param->has_mean_file = true;
	trn_data_param->phase = "train";
	trn_data_param->cursor_start = rank_id * batch_size;
	trn_data_param->cursor_step = rank_size * batch_size;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();
	printf("init trn data_layer (done) ...\n");

	printf("init tst data_layer ...\n");
	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = db_backend;
	tst_data_param->batch_size = batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	tst_data_param->scale = 1.0f;
	tst_data_param->mirror = false;
	tst_data_param->has_mean_file = true;
	tst_data_param->phase = "test";
	tst_data_param->cursor_start = 0;
	tst_data_param->cursor_step = 1;
	DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
	tst_data_layer->Setup();
	printf("init tst data_layer (done) ...\n");

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
		batch_sizes[i] = batch_size / gpus.size();
		trn_nets[i] = new Cifar10Network_t(string("trn_nets_"+i), gpus[i]);
		trn_nets[i]->BuildNet(batch_sizes[i], true, "");

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

	printf("init master_net and params_net ...\n");
	Cifar10Network_t *master_net = new Cifar10Network_t("master_net", main_gpu_id);
	master_net->BuildNet(1, false, "");
	Cifar10Network_t *params_net = new Cifar10Network_t("params_net", main_gpu_id);
	params_net->BuildNet(1, false, "");

	printf("get master_net_params_cpu_data.\n");
	vector<std::pair<float *, int> > master_net_params_cpu_data;
	master_net_params_cpu_data.push_back(std::make_pair(master_net->conv1->filtersBlob->cpu_data(), master_net->conv1->filtersBlob->count()));
	master_net_params_cpu_data.push_back(std::make_pair(master_net->conv1->biasBlob->cpu_data(),    master_net->conv1->biasBlob->count()));
	master_net_params_cpu_data.push_back(std::make_pair(master_net->conv2->filtersBlob->cpu_data(), master_net->conv2->filtersBlob->count()));
	master_net_params_cpu_data.push_back(std::make_pair(master_net->conv2->biasBlob->cpu_data(),    master_net->conv2->biasBlob->count()));
	master_net_params_cpu_data.push_back(std::make_pair(master_net->conv3->filtersBlob->cpu_data(), master_net->conv3->filtersBlob->count()));
	master_net_params_cpu_data.push_back(std::make_pair(master_net->conv3->biasBlob->cpu_data(),    master_net->conv3->biasBlob->count()));
	master_net_params_cpu_data.push_back(std::make_pair(master_net->ip1->filtersBlob->cpu_data(),   master_net->ip1->filtersBlob->count()));
	master_net_params_cpu_data.push_back(std::make_pair(master_net->ip1->biasBlob->cpu_data(),      master_net->ip1->biasBlob->count()));

	printf("get master_net_params_cpu_diff.\n");
	vector<std::pair<float *, int> > master_net_params_cpu_diff;
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->conv1->filtersBlob->cpu_diff(), master_net->conv1->filtersBlob->count()));
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->conv1->biasBlob->cpu_diff(),    master_net->conv1->biasBlob->count()));
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->conv2->filtersBlob->cpu_diff(), master_net->conv2->filtersBlob->count()));
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->conv2->biasBlob->cpu_diff(),    master_net->conv2->biasBlob->count()));
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->conv3->filtersBlob->cpu_diff(), master_net->conv3->filtersBlob->count()));
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->conv3->biasBlob->cpu_diff(),    master_net->conv3->biasBlob->count()));
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->ip1->filtersBlob->cpu_diff(),   master_net->ip1->filtersBlob->count()));
	master_net_params_cpu_diff.push_back(std::make_pair(master_net->ip1->biasBlob->cpu_diff(),      master_net->ip1->biasBlob->count()));

	printf("get master_net_params_gpu_diff.\n");
	vector<std::pair<float *, int> > master_net_params_gpu_diff;
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->conv1->filtersBlob->diff_gpu, master_net->conv1->filtersBlob->count()));
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->conv1->biasBlob->diff_gpu,    master_net->conv1->biasBlob->count()));
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->conv2->filtersBlob->diff_gpu, master_net->conv2->filtersBlob->count()));
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->conv2->biasBlob->diff_gpu,    master_net->conv2->biasBlob->count()));
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->conv3->filtersBlob->diff_gpu, master_net->conv3->filtersBlob->count()));
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->conv3->biasBlob->diff_gpu,    master_net->conv3->biasBlob->count()));
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->ip1->filtersBlob->diff_gpu,   master_net->ip1->filtersBlob->count()));
	master_net_params_gpu_diff.push_back(std::make_pair(master_net->ip1->biasBlob->diff_gpu,      master_net->ip1->biasBlob->count()));

	//	vector<std::pair<float *, int> > params_net_params_cpu_data;
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->conv1->filtersBlob->cpu_data(), params_net->conv1->filtersBlob->count()));
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->conv1->biasBlob->cpu_data(),    params_net->conv1->biasBlob->count()));
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->conv2->filtersBlob->cpu_data(), params_net->conv2->filtersBlob->count()));
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->conv2->biasBlob->cpu_data(),    params_net->conv2->biasBlob->count()));
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->conv3->filtersBlob->cpu_data(), params_net->conv3->filtersBlob->count()));
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->conv3->biasBlob->cpu_data(),    params_net->conv3->biasBlob->count()));
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->ip1->filtersBlob->cpu_data(),   params_net->ip1->filtersBlob->count()));
	//	params_net_params_cpu_data.push_back(std::make_pair(params_net->ip1->biasBlob->cpu_data(),      params_net->ip1->biasBlob->count()));

	printf("get params_net_params_cpu_diff.\n");
	vector<std::pair<float *, int> > params_net_params_cpu_diff;
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->conv1->filtersBlob->cpu_diff(), params_net->conv1->filtersBlob->count()));
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->conv1->biasBlob->cpu_diff(),    params_net->conv1->biasBlob->count()));
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->conv2->filtersBlob->cpu_diff(), params_net->conv2->filtersBlob->count()));
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->conv2->biasBlob->cpu_diff(),    params_net->conv2->biasBlob->count()));
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->conv3->filtersBlob->cpu_diff(), params_net->conv3->filtersBlob->count()));
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->conv3->biasBlob->cpu_diff(),    params_net->conv3->biasBlob->count()));
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->ip1->filtersBlob->cpu_diff(),   params_net->ip1->filtersBlob->count()));
	params_net_params_cpu_diff.push_back(std::make_pair(params_net->ip1->biasBlob->cpu_diff(),      params_net->ip1->biasBlob->count()));

	if(rank_id == 0) {
		printf("send net params into slaves.\n");
		for(int i=1; i<rank_size; i++) {
			for(int j=0; j<master_net_params_cpu_data.size(); j++) {
				MPI_Send(master_net_params_cpu_data[j].first,
						master_net_params_cpu_data[j].second,
						MPI_FLOAT, i, net_params_tags[j], MPI_COMM_WORLD);
			}
		}
	} else {
		printf("recv net params from master.\n");
		int done_count = 0;
		while(done_count != 8) {
			MPI_Status status;
			MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			int tag = status.MPI_TAG;
			if(is_number_in_set(net_params_tags, 8, tag)) {
				int index = tag - 31;
				MPI_Recv(master_net_params_cpu_data[index].first,
						master_net_params_cpu_data[index].second,
						MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				done_count++;
			}
		}

		printf("copy net params into gpu.\n");
		master_net->conv1->filtersBlob->data_to_gpu();
		master_net->conv1->biasBlob->data_to_gpu();
		master_net->conv2->filtersBlob->data_to_gpu();
		master_net->conv2->biasBlob->data_to_gpu();
		master_net->conv3->filtersBlob->data_to_gpu();
		master_net->conv3->biasBlob->data_to_gpu();
		master_net->ip1->filtersBlob->data_to_gpu();
		master_net->ip1->biasBlob->data_to_gpu();
	}


	int num_tst_iters = ceil(10000 / batch_size);
	int num_trn_iters = ceil(50000 / (batch_size * rank_size));

	float trn_local_results[2];
	float trn_global_results[2];

	printf("begin iteration: \n");
	for(int epoch = 0; epoch < max_epoch_num; epoch++) {
		tst_loss = 0.0f;
		tst_acc  = 0.0f;
		// copy trn_net params into trn_nets_i
		for(int i = 0; i < gpus.size(); i++) {
			trn_nets[i]->CopyNetParamsFrom(master_net);
		}
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network_multi(gpus, batch_sizes, batch_samples_slices, batch_labels_slices);

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_create(&(threads[i]), &pta, (void*(*)(void*))thread_func_fw, (void*)(&(thread_data[i])));
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_join(threads[i], NULL);
			}

			cudaDeviceSynchronize();
		}
		tst_loss /= num_tst_iters * gpus.size();
		tst_acc  /= num_tst_iters * gpus.size();

		if(rank_id == 0)
			printf("rank[%d]-epoch[%d]: tst_loss=%.6f, tst_acc=%.6f\n", rank_id, epoch, tst_loss, tst_acc);

		// training net
		trn_loss = 0.0f;
		trn_acc  = 0.0f;
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network_multi(gpus, batch_sizes, batch_samples_slices, batch_labels_slices);

			// copy trn_net params into trn_nets_i
			for(int i = 0; i < gpus.size(); i++) {
				trn_nets[i]->CopyNetParamsFrom(master_net);
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_create(&(threads[i]), &pta, (void*(*)(void*))thread_func_fwbw, (void*)(&(thread_data[i])));
			}

			for(int i = 0; i < gpus.size(); i++) {
				ret_count = pthread_join(threads[i], NULL);
			}
			cudaDeviceSynchronize();
			cudaSetDevice(main_gpu_id);
			master_net->ClearNetParamsDiff();
			cudaDeviceSynchronize();
			cudaSetDevice(main_gpu_id);
			for(int i = 0; i < gpus.size(); i++) {
				master_net->AddNetParamsDiffFrom(trn_nets[i]);
			}
			cudaDeviceSynchronize();

			// copy the diff into cpu
			master_net->conv1->filtersBlob->diff_to_cpu();
			master_net->conv1->biasBlob->diff_to_cpu();
			master_net->conv2->filtersBlob->diff_to_cpu();
			master_net->conv2->biasBlob->diff_to_cpu();
			master_net->conv3->filtersBlob->diff_to_cpu();
			master_net->conv3->biasBlob->diff_to_cpu();
			master_net->ip1->filtersBlob->diff_to_cpu();
			master_net->ip1->biasBlob->diff_to_cpu();

			for(int j=0; j<master_net_params_cpu_diff.size(); j++) {
				MPI_Allreduce(master_net_params_cpu_diff[j].first,
						params_net_params_cpu_diff[j].first,
						master_net_params_cpu_diff[j].second,
						MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			}

			// copy params_net_params_cpu_diff into master_net_params_gpu_diff
			for(int j=0; j<master_net_params_cpu_diff.size(); j++) {
				CUDA_CHECK( cudaMemcpy(master_net_params_gpu_diff[j].first,
						params_net_params_cpu_diff[j].first,
						params_net_params_cpu_diff[j].second * sizeof(float),
						cudaMemcpyHostToDevice) );
			}

			// update the master_net in each nodes
			master_net->UpdateNet(-(1.0f / (rank_size * gpus.size())));

		}
		trn_local_results[0] = trn_loss / (num_trn_iters * gpus.size());
		trn_local_results[1] = trn_acc / (num_trn_iters * gpus.size());
		MPI_Allreduce(trn_local_results, trn_global_results, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		trn_loss = trn_global_results[0] / rank_size;
		trn_acc  = trn_global_results[1] / rank_size;

		if(rank_id == 0)
			printf("rank[%d]-epoch[%d]: trn_loss=%.6f, trn_acc=%.6f\n", rank_id, epoch, trn_loss, trn_acc);

		// update learning rate
		if((epoch != 0) && (epoch % lr_stepsize == 0)) {
			lr_rate /= 10;
		}
	}

	if(rank_id == 0)
		master_net->SaveNetParams(100);

	for(int i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		delete trn_nets[i]; trn_nets[i] = NULL;
	}

	cudaSetDevice(main_gpu_id);
	batch_samples_slices.clear();
	batch_labels_slices.clear();

	delete master_net;
	delete params_net;
	delete trn_data_layer;
	delete tst_data_layer;
	delete trn_data_param;
	delete tst_data_param;

	if(num_gpus >= gpus.size()) {
		std::cout << ("disable P2P: ");
		DisableP2P(gpus);
	}

	pthread_barrier_destroy(&barr);
	ret_count = pthread_attr_destroy(&pta);
	free(threads); threads = NULL;
	cudaDeviceReset();

	MPI_Finalize();

	return 0;
}
