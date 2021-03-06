
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


int main_basic_send_recv(int argc, char **argv) {

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

	printf("yes1\n");
	MPI_Init(&argc, &argv);
	int rank_id, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	char myname[MPI_MAX_PROCESSOR_NAME];
	int namelen;
	MPI_Get_processor_name(myname, &namelen);
	int key_tag = 1;
	int name_tag = 2;
	printf("yes2\n");

	if(rank_id == 0) {
		for(int rank = 1 ; rank < rank_size; rank++) {

			int key = rand() % 50000;
			char key_str[5];
			int key_len = snprintf(key_str, 6, "%05d", key);
			printf("send key %s(%d) to slave %d.\n", key_str, key_len, rank);
			MPI_Send(key_str, key_len, MPI_CHAR, rank, key_tag, MPI_COMM_WORLD);
		}

		printf("receive hostnames from slaves.\n");
		for(int rank_ = 1; rank_ < rank_size; rank_++) {
			MPI_Status status;
			int message_size;
			printf("probe rank %d\n", rank_);
			MPI_Probe(rank_, name_tag, MPI_COMM_WORLD, &status);
			printf("get count rank %d\n", rank_);
			MPI_Get_count(&status, MPI_CHAR, &message_size);
			char *message_buf = (char*)malloc(sizeof(char) * message_size);
			printf("receive from rank %d\n", rank_);
			MPI_Recv(message_buf, message_size, MPI_CHAR, rank_, name_tag, MPI_COMM_WORLD, &status);
			printf("rank %d(%d): (%d), %s\n", rank_, status.MPI_ERROR, message_size, message_buf);
			free(message_buf);
		}
	} else {

		MPI_Status status;
		MPI_Probe(0, key_tag, MPI_COMM_WORLD, &status);
		int key_size;
		MPI_Get_count(&status, MPI_CHAR, &key_size);
		char *message_buf = (char*)malloc(sizeof(char) * key_size);
		MPI_Recv(message_buf, key_size, MPI_CHAR, 0, key_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("rank %d: %s\n", rank_id, message_buf);
		string key = string(message_buf);
		free(message_buf);

		// send the hostname to the master;
		stringstream ss;
		ss << myname << "_" << key;
		char *ss_str = const_cast<char *>(ss.str().c_str());
		MPI_Send(ss_str, strlen(ss_str), MPI_CHAR, 0, name_tag, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}

int main_test_cursor_start_and_step_ok(int argc, char **argv) {
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

	cudaSetDevice(main_gpu_id);

	DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
	trn_data_param->backend = "lmdb";
	trn_data_param->batch_size = batch_size;
	trn_data_param->source = trn_db_filename;
	trn_data_param->mean_file = mean_file;
	trn_data_param->crop_size = 0;
	trn_data_param->scale = 1.0f;
	trn_data_param->mirror = true;
	trn_data_param->has_mean_file = true;
	trn_data_param->phase = "train";
	trn_data_param->cursor_start = 100;
	trn_data_param->cursor_step = 2;
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();
	printf("key: %s\n", trn_data_layer->cursor_->key().c_str());

	Blob_t *batch_samples = new Blob_t(batch_size, 3, 32, 32);
	Blob_t *batch_labels = new Blob_t(batch_size, 1,1,1);
	batch_samples->allocate_gpu_data();
	batch_labels->allocate_gpu_data();
	trn_data_layer->Forward_to_Network(batch_samples, batch_labels);
	printf("key: %s\n", trn_data_layer->cursor_->key().c_str());
	trn_data_layer->Forward_to_Network(batch_samples, batch_labels);
	printf("key: %s\n", trn_data_layer->cursor_->key().c_str());

	delete batch_samples;
	delete batch_labels;

	delete trn_data_param;
	delete trn_data_layer;

	return 0;
}


int main_independent_net_for_each_slave_ok(int argc, char **argv) {

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
	const int net_params_tag = 3;
	const int net_tst_tag = 4;
	const int net_trn_tag = 5;
	const int net_done_tag = 6;

	if(rank_id == 0) {

		cudaSetDevice(main_gpu_id);

		printf("build master_net.\n");
		Cifar10Network_t *master_net = new Cifar10Network_t("master_net", main_gpu_id);
		master_net->BuildNet(batch_size, false, "");

		printf("get net params.\n");
		vector<std::pair<float *, int> > net_params_buffers;
		net_params_buffers.push_back(std::make_pair(master_net->conv1->filtersBlob->cpu_data(), master_net->conv1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv1->biasBlob->cpu_data(),    master_net->conv1->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv2->filtersBlob->cpu_data(), master_net->conv2->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv2->biasBlob->cpu_data(),    master_net->conv2->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv3->filtersBlob->cpu_data(), master_net->conv3->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv3->biasBlob->cpu_data(),    master_net->conv3->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->ip1->filtersBlob->cpu_data(),   master_net->ip1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->ip1->biasBlob->cpu_data(),      master_net->ip1->biasBlob->count()));

		printf("send net params into slaves.\n");
		for (int i=1; i<rank_size; i++) {
			for(int j=0; j<8; j++) {
				MPI_Send(net_params_buffers[j].first, net_params_buffers[j].second, MPI_FLOAT, i, net_params_tag, MPI_COMM_WORLD);
			}
		}

		printf("begin to receiving messages from slaves ...\n");
		MPI_Status  recv_status;
		float result[3];
		int done_count = 0;
		while(true) {
			MPI_Recv(result, 3, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
			int recv_rank = recv_status.MPI_SOURCE;
			switch(recv_status.MPI_TAG) {
			case net_trn_tag:
				printf("rank[%d]-epoch[%d]: trn_loss=%.6f, trn_acc=%.6f\n", recv_rank, (int)result[0], result[1], result[2]);
				break;
			case net_tst_tag:
				printf("rank[%d]-epoch[%d]: tst_loss=%.6f, tst_acc=%.6f\n", recv_rank, (int)result[0], result[1], result[2]);
				break;
			case net_done_tag:
				done_count+=1;
				break;
			default:
				printf("No, the received tag %d is not a correct tag.\n", recv_status.MPI_TAG);
				break;
			}

			if(done_count == rank_size - 1)
				break;
		}

		delete master_net;
		net_params_buffers.clear();

	} else {

		cudaSetDevice(main_gpu_id);

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
		DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
		trn_data_layer->Setup();

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
		DataLayer_t *tst_data_layer = new DataLayer_t(tst_data_param);
		tst_data_layer->Setup();

		Cifar10Network_t *slave_net = new Cifar10Network_t("slave_net", main_gpu_id);
		slave_net->BuildNet(batch_size, true, "");

		vector<std::pair<float *, int> > net_params_buffers;
		net_params_buffers.push_back(std::make_pair(slave_net->conv1->filtersBlob->cpu_data(), slave_net->conv1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv1->biasBlob->cpu_data(),    slave_net->conv1->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv2->filtersBlob->cpu_data(), slave_net->conv2->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv2->biasBlob->cpu_data(),    slave_net->conv2->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv3->filtersBlob->cpu_data(), slave_net->conv3->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv3->biasBlob->cpu_data(),    slave_net->conv3->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->ip1->filtersBlob->cpu_data(),   slave_net->ip1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->ip1->biasBlob->cpu_data(),      slave_net->ip1->biasBlob->count()));

		for(int j=0; j<8; j++) {
			MPI_Status status;
			MPI_Probe(0, net_params_tag, MPI_COMM_WORLD, &status);
			int msg_size;
			MPI_Get_count(&status, MPI_CHAR, &msg_size);
			MPI_Recv(net_params_buffers[j].first, msg_size, MPI_FLOAT, 0, net_params_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// copy net params into gpu
		slave_net->conv1->filtersBlob->data_to_gpu();
		slave_net->conv1->biasBlob->data_to_gpu();
		slave_net->conv2->filtersBlob->data_to_gpu();
		slave_net->conv2->biasBlob->data_to_gpu();
		slave_net->conv3->filtersBlob->data_to_gpu();
		slave_net->conv3->biasBlob->data_to_gpu();
		slave_net->ip1->filtersBlob->data_to_gpu();
		slave_net->ip1->biasBlob->data_to_gpu();

		int num_tst_iters = ceil(10000 / batch_size);
		int num_trn_iters = ceil(50000 / batch_size);
		float result[3];
		for(int epoch = 0; epoch < max_epoch_num; epoch++) {

			// testing net
			float tst_loss = 0.0f, tst_loss_batch = 0.0f;
			float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
			for(int iter = 0; iter < num_tst_iters; iter++) {
				tst_data_layer->Forward_to_Network(slave_net->batch_samples, slave_net->batch_labels);
				slave_net->Forward(&tst_loss_batch, &tst_acc_batch);
				tst_loss += tst_loss_batch;
				tst_acc += tst_acc_batch;
			}
			tst_loss /= num_tst_iters;
			tst_acc  /= num_tst_iters;

			result[0] = epoch;
			result[1] = tst_loss;
			result[2] = tst_acc;
			MPI_Request tst_request;
			MPI_Isend(result, 3, MPI_FLOAT, 0, net_tst_tag, MPI_COMM_WORLD, &tst_request);

			// training net
			float trn_loss = 0.0f, trn_loss_batch = 0.0f;
			float trn_acc  = 0.0f, trn_acc_batch  = 0.0f;
			for(int iter = 0; iter < num_trn_iters; iter++) {
				trn_data_layer->Forward_to_Network(slave_net->batch_samples, slave_net->batch_labels);
				slave_net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
				trn_loss += trn_loss_batch;
				trn_acc  += trn_acc_batch;
				slave_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);
				slave_net->UpdateNet();
			}
			trn_loss /= num_trn_iters;
			trn_acc  /= num_trn_iters;

			result[0] = epoch;
			result[1] = trn_loss;
			result[2] = trn_acc;
			MPI_Request trn_request;
			MPI_Isend(result, 3, MPI_FLOAT, 0, net_trn_tag, MPI_COMM_WORLD, &trn_request);

			// update learning rate
			if((epoch != 0) && (epoch % lr_stepsize == 0)) {
				lr_rate /= 10;
			}
		}

		delete slave_net;
		delete trn_data_layer;
		delete tst_data_layer;
		delete trn_data_param;
		delete tst_data_param;
		cudaDeviceReset();

		result[0] = 0;
		result[1] = 0;
		result[2] = 0;
		MPI_Request done_request;
		MPI_Isend(result, 3, MPI_FLOAT, 0, net_done_tag, MPI_COMM_WORLD, &done_request);
	}

	MPI_Finalize();
	return 0;
}

int main_test_mpi_allreduce(int argc, char **argv) {
	// test MPI_Allreduce small example
	MPI_Init(&argc, &argv);
	int rank_id, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	char myname[MPI_MAX_PROCESSOR_NAME];
	int namelen;
	MPI_Get_processor_name(myname, &namelen);

	const int N = 10;
	float *arr = NULL;
	const int data_tag = 1;
	const int result_tag = 2;

	// data transfer
	if(rank_id == 0) {
		// send data into slaves
		arr = new float[10];
		printf("rank 0 arr data: ");
		for(int i=0; i<N; i++) {
			arr[i] = (float)rand() / (float)RAND_MAX;
			printf("%.6f ", arr[i]);
		}
		printf("\n");

		for(int i=1; i<rank_size; i++) {
			MPI_Send(arr, N, MPI_FLOAT, i, data_tag, MPI_COMM_WORLD);
		}
	} else {
		// rece data from master
		MPI_Status status;
		MPI_Probe(0, data_tag, MPI_COMM_WORLD, &status);
		int arr_size;
		MPI_Get_count(&status, MPI_FLOAT, &arr_size);
		arr = new float[arr_size];
		MPI_Recv(arr, arr_size, MPI_FLOAT, 0, data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	float *global_sum = new float[N];
	for(int i=0; i<N; i++)
		global_sum[i] = 0;
	MPI_Allreduce(arr, global_sum, N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

	if(rank_id == 0) {
		printf("rank 0 global_sum: ");
		for(int i=0; i<N; i++)
			printf("%.6f ", global_sum[i]);
		printf("\n");

		for(int i=1; i<rank_size; i++) {
			MPI_Recv(arr, N, MPI_FLOAT, i, result_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("rank %d global_sum: ", i);
			for(int i=0; i<N; i++)
				printf("%.6f ", arr[i]);
			printf("\n");
		}
	} else {
		MPI_Send(global_sum, N, MPI_FLOAT, 0, result_tag, MPI_COMM_WORLD);
	}

	if(arr != NULL)
		delete[] arr;
	if(global_sum != NULL)
		delete[] global_sum;

	return 0;

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

int main_still_debugging(int argc, char **argv) {

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
	const int cursor_tag = 1;
	const int net_params_tag = 3;
	const int net_tst_tag = 4;
	const int net_tst_cursor_tag = 41;
	const int net_trn_tag = 5;
	const int net_trn_cursor_tag = 51;
	const int net_done_tag = 6;

	if(rank_id == 0) {
		// send cursor_start and cursor_step into slaves
		int cursor_info[2] = {0,0};
		for(int i=1; i<rank_size; i++) {
			cursor_info[0] = (i-1) * batch_size; // cursor_start
			cursor_info[1] = (rank_size - 1) * batch_size; // cursor_step
			MPI_Send(cursor_info, 2, MPI_INT, i, cursor_tag, MPI_COMM_WORLD);
		}

		cudaSetDevice(main_gpu_id);

		printf("build master_net.\n");
		Cifar10Network_t *master_net = new Cifar10Network_t("master_net", main_gpu_id);
		master_net->BuildNet(batch_size, false, "");

		printf("get net params.\n");
		vector<std::pair<float *, int> > net_params_buffers;
		net_params_buffers.push_back(std::make_pair(master_net->conv1->filtersBlob->cpu_data(), master_net->conv1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv1->biasBlob->cpu_data(),    master_net->conv1->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv2->filtersBlob->cpu_data(), master_net->conv2->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv2->biasBlob->cpu_data(),    master_net->conv2->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv3->filtersBlob->cpu_data(), master_net->conv3->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->conv3->biasBlob->cpu_data(),    master_net->conv3->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->ip1->filtersBlob->cpu_data(),   master_net->ip1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(master_net->ip1->biasBlob->cpu_data(),      master_net->ip1->biasBlob->count()));

		printf("send net params into slaves.\n");
		for (int i=1; i<rank_size; i++) {
			for(int j=0; j<8; j++) {
				MPI_Send(net_params_buffers[j].first, net_params_buffers[j].second, MPI_FLOAT, i, net_params_tag, MPI_COMM_WORLD);
			}
		}

		printf("begin to receiving messages from slaves ...\n");
		MPI_Status  recv_status;
		float result[3];
		int done_count = 0;
		while(true) {
			MPI_Status status;
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			int msg_tag = status.MPI_TAG;
			int msg_source = status.MPI_SOURCE;
			if(msg_tag == net_tst_cursor_tag || msg_tag == net_trn_cursor_tag) {
				int msg_size;
				MPI_Get_count(&status, MPI_CHAR, &msg_size);
				char *msg_str = new char[msg_size];
				MPI_Recv(msg_str, msg_size, MPI_CHAR, msg_source, msg_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				printf("rank %d, %s: %s\n", msg_source, msg_tag == net_tst_cursor_tag ? "tst_msg" : "trn_msg", msg_str);
				delete[] msg_str;
			} else {
				MPI_Recv(result, 3, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
				int recv_rank = recv_status.MPI_SOURCE;
				switch(recv_status.MPI_TAG) {
				case net_trn_tag:
					printf("rank[%d]-epoch[%d]: trn_loss=%.6f, trn_acc=%.6f\n", recv_rank, (int)result[0], result[1], result[2]);
					break;
				case net_tst_tag:
					printf("rank[%d]-epoch[%d]: tst_loss=%.6f, tst_acc=%.6f\n", recv_rank, (int)result[0], result[1], result[2]);
					break;
				case net_done_tag:
					done_count+=1;
					break;
				default:
					printf("No, the received tag %d is not a correct tag.\n", recv_status.MPI_TAG);
					break;
				}
			}

			if(done_count == rank_size - 1)
				break;
		}

		delete master_net;
		net_params_buffers.clear();


	} else {
		// recv cursor_start and cursor_step from master
		MPI_Status status;
		MPI_Probe(0, cursor_tag, MPI_COMM_WORLD, &status);
		int msg_size;
		MPI_Get_count(&status, MPI_INT, &msg_size);
		int *cursor_info = new int[msg_size];
		MPI_Recv(cursor_info, msg_size, MPI_INT, 0, cursor_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		cudaSetDevice(main_gpu_id);

		DataLayerParameter_t *trn_data_param = new DataLayerParameter_t();
		trn_data_param->backend = "lmdb";
		trn_data_param->batch_size = batch_size;
		trn_data_param->source = trn_db_filename;
		trn_data_param->mean_file = mean_file;
		trn_data_param->crop_size = 0;
		trn_data_param->scale = 1.0f;
		trn_data_param->mirror = true;
		trn_data_param->has_mean_file = true;
		trn_data_param->phase = "train";
		trn_data_param->cursor_start = cursor_info[0];
		trn_data_param->cursor_step = cursor_info[1];
		DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
		trn_data_layer->Setup();
		delete[] cursor_info;

		DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
		tst_data_param->backend = "lmdb";
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

		Cifar10Network_t *slave_net = new Cifar10Network_t("slave_net", main_gpu_id);
		slave_net->BuildNet(batch_size, true, "");

		vector<std::pair<float *, int> > net_params_buffers;
		net_params_buffers.push_back(std::make_pair(slave_net->conv1->filtersBlob->cpu_data(), slave_net->conv1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv1->biasBlob->cpu_data(),    slave_net->conv1->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv2->filtersBlob->cpu_data(), slave_net->conv2->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv2->biasBlob->cpu_data(),    slave_net->conv2->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv3->filtersBlob->cpu_data(), slave_net->conv3->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->conv3->biasBlob->cpu_data(),    slave_net->conv3->biasBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->ip1->filtersBlob->cpu_data(),   slave_net->ip1->filtersBlob->count()));
		net_params_buffers.push_back(std::make_pair(slave_net->ip1->biasBlob->cpu_data(),      slave_net->ip1->biasBlob->count()));

		for(int j=0; j<8; j++) {
			MPI_Status status;
			MPI_Probe(0, net_params_tag, MPI_COMM_WORLD, &status);
			int msg_size;
			MPI_Get_count(&status, MPI_CHAR, &msg_size);
			MPI_Recv(net_params_buffers[j].first, msg_size, MPI_FLOAT, 0, net_params_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// copy net params into gpu
		slave_net->conv1->filtersBlob->data_to_gpu();
		slave_net->conv1->biasBlob->data_to_gpu();
		slave_net->conv2->filtersBlob->data_to_gpu();
		slave_net->conv2->biasBlob->data_to_gpu();
		slave_net->conv3->filtersBlob->data_to_gpu();
		slave_net->conv3->biasBlob->data_to_gpu();
		slave_net->ip1->filtersBlob->data_to_gpu();
		slave_net->ip1->biasBlob->data_to_gpu();

		int num_tst_iters = ceil(10000 / batch_size);
		int num_trn_iters = ceil(50000 / batch_size);

		float result[3];
		float tst_local_results[2];
		float tst_global_results[2];
		float trn_local_results[2];
		float trn_global_results[2];
		for(int epoch = 0; epoch < max_epoch_num; epoch++) {
			float tst_loss = 0.0f, tst_loss_batch = 0.0f;
			float tst_acc  = 0.0f, tst_acc_batch  = 0.0f;
			for(int iter = 0; iter < num_tst_iters; iter++) {
				tst_data_layer->Forward_to_Network(slave_net->batch_samples, slave_net->batch_labels);
				slave_net->Forward(&tst_loss_batch, &tst_acc_batch);
				tst_loss += tst_loss_batch;
				tst_acc += tst_acc_batch;
			}
			tst_local_results[0] = tst_loss;
			tst_local_results[1] =tst_acc;
			MPI_Allreduce(tst_local_results, tst_global_results, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

			tst_loss = tst_global_results[0] / num_tst_iters;
			tst_acc  = tst_global_results[1] / num_tst_iters;

			result[0] = epoch;
			result[1] = tst_loss;
			result[2] = tst_acc;
			MPI_Request tst_request;
			MPI_Isend(result, 3, MPI_FLOAT, 0, net_tst_tag, MPI_COMM_WORLD, &tst_request);

			// training net
			float trn_loss = 0.0f, trn_loss_batch = 0.0f;
			float trn_acc  = 0.0f, trn_acc_batch  = 0.0f;
			for(int iter = 0; iter < num_trn_iters; iter++) {
				trn_data_layer->Forward_to_Network(slave_net->batch_samples, slave_net->batch_labels);
				slave_net->ForwardBackward(&trn_loss_batch, &trn_acc_batch);
				trn_loss += trn_loss_batch;
				trn_acc  += trn_acc_batch;
				slave_net->ComputeUpdateValue(lr_rate, momentum, weight_decay);
				slave_net->UpdateNet();
			}
			trn_loss /= num_trn_iters;
			trn_acc  /= num_trn_iters;

			result[0] = epoch;
			result[1] = trn_loss;
			result[2] = trn_acc;
			MPI_Request trn_request;
			MPI_Isend(result, 3, MPI_FLOAT, 0, net_trn_tag, MPI_COMM_WORLD, &trn_request);

			// update learning rate
			if((epoch != 0) && (epoch % lr_stepsize == 0)) {
				lr_rate /= 10;
			}
		}


		delete slave_net;
		delete trn_data_layer;
		delete tst_data_layer;
		delete trn_data_param;
		delete tst_data_param;
		cudaDeviceReset();

		result[0] = 0;
		result[1] = 0;
		result[2] = 0;
		MPI_Request done_request;
		MPI_Isend(result, 3, MPI_FLOAT, 0, net_done_tag, MPI_COMM_WORLD, &done_request);
	}

	return 0;
}

