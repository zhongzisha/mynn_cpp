

#include "common.hpp"
#include "blob.hpp"
#include "common_layer.hpp"
#include "data_layer.hpp"
#include "conv_layer.hpp"
#include "loss_layer.hpp"
#include "network_cifar10.hpp"

int main(int argc, char **argv) {
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
	trn_data_param->scale = 1.0f;
	trn_data_param->mirror = true;
	trn_data_param->has_mean_file = true;
	trn_data_param->phase = "train";
	DataLayer_t *trn_data_layer = new DataLayer_t(trn_data_param);
	trn_data_layer->Setup();

	DataLayerParameter_t *tst_data_param = new DataLayerParameter_t();
	tst_data_param->backend = "lmdb";
	tst_data_param->batch_size = tst_batch_size;
	tst_data_param->source = tst_db_filename;
	tst_data_param->mean_file = mean_file;
	tst_data_param->crop_size = 0;
	tst_data_param->scale = 1.0f;
	tst_data_param->mirror = false;
	tst_data_param->has_mean_file = true;
	tst_data_param->phase = "test";
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




















