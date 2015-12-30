
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

//#include "common.hpp"
//#include "blob.hpp"
//#include "common_layer.hpp"
//#include "data_layer.hpp"
//#include "conv_layer.hpp"
//#include "loss_layer.hpp"
//#include "network_cifar10.hpp"
//#include "network_alex.hpp"


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
			printf("send key to slave %d.\n", rank);
			MPI_Send(&key, 1, MPI_INT, rank, key_tag, MPI_COMM_WORLD);
		}

		printf("receive hostnames from slaves.\n");
		for(int rank = 1; rank < rank_size; rank++) {
			MPI_Status status;
			int name_size;
			printf("probe rank %d\n", rank);
			MPI_Probe(rank, name_tag, MPI_COMM_WORLD, &status);
			printf("get count rank %d\n", rank);
			MPI_Get_count(&status, MPI_CHAR, &name_size);
			char *message_buf = (char*)malloc(sizeof(char) * name_size);
			printf("receive from rank %d\n", rank);
			MPI_Recv(message_buf, name_size, MPI_CHAR, rank, name_tag, MPI_COMM_WORLD, &status);
			printf("rank %d: %s\n", rank, message_buf);

			free(message_buf);
		}
	} else {
		MPI_Status status;
		int key;
		/* Now receive the message into the allocated buffer */
		MPI_Recv(&key, 1, MPI_INT, 0, key_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


		// send the hostname to the master
		stringstream ss;
		ss << myname << "_" << key;
		char *ss_str = const_cast<char *>(ss.str().c_str());
		int ss_str_len = strlen(ss_str);
		MPI_Send(ss_str, ss_str_len, MPI_CHAR, 0, name_tag, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}
