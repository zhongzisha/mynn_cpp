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


	printf("init master_net and params_net ...\n");
	Cifar10Network_t *master_net = new Cifar10Network_t("master_net", main_gpu_id);
	master_net->BuildNet(batch_size, true, "");
	Cifar10Network_t *params_net = new Cifar10Network_t("params_net", main_gpu_id);
	params_net->BuildNet(batch_size, false, "");

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
		float tst_loss = 0.0f, tst_loss_batch = 0.0f;
		float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_tst_iters; iter++) {
			tst_data_layer->Forward_to_Network(master_net->batch_samples, master_net->batch_labels);
			master_net->Forward(&tst_loss_batch, &tst_acc_batch);
			tst_loss += tst_loss_batch;
			tst_acc += tst_acc_batch;
		}
		tst_loss /= num_tst_iters;
		tst_acc  /= num_tst_iters;

		if(rank_id == 0)
			printf("rank[%d]-epoch[%d]: tst_loss=%.6f, tst_acc=%.6f\n", rank_id, epoch, tst_loss, tst_acc);

		// training net
		float trn_loss = 0.0f, trn_loss_batch = 0.0f;
		float trn_acc  = 0.0f, trn_acc_batch  = 0.0f;
		for(int iter = 0; iter < num_trn_iters; iter++) {
			trn_data_layer->Forward_to_Network(master_net->batch_samples, master_net->batch_labels);
			master_net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
			trn_loss += trn_loss_batch;
			trn_acc  += trn_acc_batch;
			master_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);

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
			master_net->UpdateNet(-(1.0f / rank_size));

		}
		trn_local_results[0] = trn_loss / num_trn_iters;
		trn_local_results[1] = trn_acc / num_trn_iters;
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

	delete master_net;
	delete params_net;
	delete trn_data_layer;
	delete tst_data_layer;
	delete trn_data_param;
	delete tst_data_param;
	cudaDeviceReset();

	MPI_Finalize();

	return 0;
}
