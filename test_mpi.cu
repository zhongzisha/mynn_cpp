
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


int main(int argc, char **argv) {

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

		float local_lr_rate = lr_rate;

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
		DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
		trn_data_layer->Setup();

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
				slave_net->ComputeUpdateValue(local_lr_rate, momentum, weight_decay);
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
				local_lr_rate /= 10;
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
