/* http://www.lam-mpi.org/tutorials/one-step/ezstart.php */
/* Master/Slave Algorithmus, automatische Lastverteilung */
/* Umbenennungen myrank -> pid , ntasks -> np , rank -> id */

#include <mpi.h>


#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <map>
#include <list>
#include <stack>
#include <deque>
using namespace std;

#include "myproto.pb.h"
using namespace myproto;

#include <cuda_runtime.h>
#include <driver_types.h>  // cuda driver types

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <boost/filesystem.hpp>
using namespace boost;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#define CUDA_CALL(x) { \
		const cudaError_t a=(x); \
		if(a!=cudaSuccess) { \
			printf("CUDA Error: %s (error_num: %d)\n",cudaGetErrorString(a),a); \
			cudaDeviceReset(); \
			assert(0); \
		} \
}
//Demo: CUDA_CALL(cudaSetDevice(0));



using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

bool approximatelyEqual(float a, float b, float epsilon) {
	return fabs(a-b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}
bool essentiallyEqual(float a, float b, float epsilon) {
	return fabs(a-b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}
bool definitelyGreaterThan(float a, float b, float epsilon) {
	return (a-b) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}
bool definitelyLessThan(float a, float b, float epsilon) {
	return (b-a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<typename Dtype>
bool feature_extraction_pipeline(
		string &src_imgname,
		string &pretrained_net_param,
		string &feature_extraction_proto_file,
		string &extract_feature_blob_names,
		string &save_feature_dataset_names,
		int padHeight,
		int padWidth,
		int half_width) {

	// get the devices
	int devCount = 0;
	cudaGetDeviceCount(&devCount);
	int device_id = 0;
	float maximum_free_mem = FLT_MIN;
	if(devCount>0) {
		size_t gpuTotalMem;
		size_t gpuFreeMem;
		int deviceid;
		for(int i=0;i<devCount;i++)
		{
			cudaSetDevice(i);
			cudaDeviceProp device_prop;
			cudaGetDeviceProperties(&device_prop, i);
			cudaGetDevice(&deviceid);
			CUDA_CALL(cudaMemGetInfo(&gpuFreeMem, &gpuTotalMem));

			float free_mem = (float)gpuFreeMem/1000000.0f;
			if(definitelyGreaterThan(free_mem, maximum_free_mem, 1e-6)) {
				maximum_free_mem = free_mem;
				device_id = i;
			}
		}
	} else {
		// no GPU device
		return false;
	}

	// get the batch_size according to the size of free GPU memory
	int batch_size = 1024;
	if(maximum_free_mem>=4000) {
		batch_size = 4096;
	} else if(maximum_free_mem<4000 && maximum_free_mem>=2000) {
		batch_size = 2048;
	} else if(maximum_free_mem<2000 && maximum_free_mem>=1000) {
		batch_size = 1024;
	}

	// set the GPU device
	Caffe::SetDevice(device_id);
	Caffe::set_mode(Caffe::GPU);

	// resotre the net
	std::string pretrained_binary_proto(pretrained_net_param);
	std::string feature_extraction_proto(feature_extraction_proto_file);
	shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto, caffe::TEST));
	feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

	std::vector<std::string> blob_names;
	boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));
	std::vector<std::string> dataset_names;
	boost::split(dataset_names, save_feature_dataset_names, boost::is_any_of(","));
	size_t num_features = blob_names.size();

	std::vector<FILE *> feature_txts;
	for (size_t i = 0; i < num_features; ++i) {
		FILE *fp = fopen((dataset_names[i]+".txt").c_str(), "w");
		feature_txts.push_back(fp);
	}

	boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layer_by_name("data"))->set_batch_size(batch_size);
	string data_layer_type = string(boost::static_pointer_cast<caffe::Layer<Dtype> >(feature_extraction_net->layer_by_name("data"))->type());
	if (data_layer_type == "MemoryData") {
		// get the inputs
		// read the source image
		Mat srcImg = imread(src_imgname);
		// pad the image
		Mat padImg;
		int top=padHeight;
		int bottom=padHeight;
		int left=padWidth;
		int right=padWidth;
		int borderType=cv::BORDER_REFLECT_101;
		copyMakeBorder(srcImg,padImg,top,bottom,left,right,borderType);
		std::vector<cv::Mat> mat_vector;
		std::vector<int> lab_vector;
		std::vector<Blob<float>*> input_vec;
		int count = 0;
		for(int x = padWidth; x < padWidth+srcImg.cols; x++) {
			for(int y = padHeight; y < padHeight+srcImg.rows; y++) {
				cv::Range rows_range(y-half_width,y+half_width);
				cv::Range cols_range(x-half_width,x+half_width);
				Mat blockImg=padImg(rows_range,cols_range);
				mat_vector.push_back(blockImg);
				lab_vector.push_back(0);

				if (++count % batch_size == 0) {
					//feed the mat_vector and lab_vector to the Net
					boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layer_by_name("data"))->AddMatVector(mat_vector, lab_vector);

					// Forward the Net and save the output features
					feature_extraction_net->Forward(input_vec);
					for (int i = 0; i < num_features; ++i) {
						const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[i]);
						const shared_ptr<Blob<Dtype> > label_blob = feature_extraction_net->blob_by_name("label");
						int batch_size = feature_blob->num();
						int dim_features = feature_blob->count() / batch_size;
						const Dtype* feature_blob_data;
						for (int n = 0; n < batch_size; ++n) {
							feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
							fprintf((FILE*)feature_txts[i], "%d ", (int)(label_blob->cpu_data()[n])+1);//because the start label is 0
							for (int d = 0; d < dim_features; ++d) {
								if(!approximatelyEqual(feature_blob_data[d], 0.0f, 1e-6)) {
									fprintf((FILE*)feature_txts[i], "%d:%.6f ", d, feature_blob_data[d]);
								}
							}
							fprintf((FILE*)feature_txts[i], "\n");
						}  // for (int n = 0; n < batch_size; ++n)
					}  // for (int i = 0; i < num_features; ++i)

					//clear the mat_vector and lab_vector
					mat_vector.clear();
					lab_vector.clear();
					LOG(ERROR) << "Processed " << count << " pixels.";
				}
			}
		}

		// for the left pixels
		int mat_vector_size = mat_vector.size();
		if(mat_vector_size != 0) {
			Mat blockImg = mat_vector[mat_vector_size - 1];
			int need_size = batch_size - mat_vector_size;
			for(int j = 0; j < need_size; j++) {
				mat_vector.push_back(blockImg);
				lab_vector.push_back(0);
			}
			//feed the mat_vector and lab_vector to the Net
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layer_by_name("data"))->AddMatVector(mat_vector, lab_vector);

			// Forward the Net and save the output features
			feature_extraction_net->Forward(input_vec);
			for (int i = 0; i < num_features; ++i) {
				const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[i]);
				const shared_ptr<Blob<Dtype> > label_blob = feature_extraction_net->blob_by_name("label");
				int batch_size = feature_blob->num();
				int dim_features = feature_blob->count() / batch_size;
				const Dtype* feature_blob_data;
				for (int n = 0; n < batch_size; ++n) {
					feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
					fprintf((FILE*)feature_txts[i], "%d ", (int)(label_blob->cpu_data()[n])+1);//because the start label is 0
					for (int d = 0; d < dim_features; ++d) {
						if(!approximatelyEqual(feature_blob_data[d], 0.0f, 1e-6)) {
							fprintf((FILE*)feature_txts[i], "%d:%.6f ", d, feature_blob_data[d]);
						}
					}
					fprintf((FILE*)feature_txts[i], "\n");
				}  // for (int n = 0; n < batch_size; ++n)
			}  // for (int i = 0; i < num_features; ++i)

			//clear the mat_vector and lab_vector
			mat_vector.clear();
			lab_vector.clear();
		}
	}

	// close the FILE pointer handles
	for (int i = 0; i < num_features; ++i) {
		fclose(feature_txts[i]); feature_txts[i] = NULL;
	}

	return true;
}



#define WORKTAG 1
#define DIETAG 2

struct caffenet_params {
	string image_filename;
	string pretrained_net_param;
	string feature_extraction_proto_file;
	string extract_feature_blob_names;
	int pad_height;
	int pad_width;
	int half_width;
};

WorkUnit get_next_work_item(std::deque<std::string> &image_lists, caffenet_params &params)
{
	/* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a slave. */
	WorkUnit work;

	if(!image_lists.empty()) {

		string image_filename = image_lists.front();
		work.set_image_filename(image_filename);
		int dotpos = image_filename.find_last_of(".");
		work.set_extract_feature_blob_names(params.extract_feature_blob_names);
		work.set_pretrained_net_param(params.pretrained_net_param);
		work.set_feature_extraction_proto_file(params.feature_extraction_proto_file);
		work.set_save_feature_dataset_names(image_filename.substr(0,dotpos));
		work.set_half_width(params.half_width);
		work.set_pad_height(params.pad_height);
		work.set_pad_width(params.pad_width);


		image_lists.pop_front();
	} else {
		work.set_image_filename("");
	}
	return work;

}


void process_results(ResultUnit &result, std::deque<std::string> &image_lists)
{
	/* Fill in with whatever is relevant to process the results returned
     by the slave */

	if(result.result_type() == ResultUnit::SUCCESS) {
		std::cout << result.image_filename() << " is done with success. \n ";
		std::cout << "result_string: " << result.result_string() << "\n";
	} else {
		std::cout << result.image_filename() << " is done with failure. \n";
		image_lists.push_back(result.image_filename());
	}

}


ResultUnit do_work(WorkUnit &work)
{
	/* Fill in with whatever is necessary to process the work and
     generate a result */

	ResultUnit result;
	result.set_image_filename(work.image_filename());

	string image_filename = work.image_filename();
	string pretrained_net_param = work.pretrained_net_param();
	string feature_extraction_proto_file = work.feature_extraction_proto_file();
	string extract_feature_blob_names = work.extract_feature_blob_names();
	string save_feature_dataset_names = work.save_feature_dataset_names();
	int pad_height = work.pad_height();
	int pad_width = work.pad_width();
	int half_width = work.half_width();

	bool success = false;
	try {
		success = feature_extraction_pipeline<float>(image_filename, pretrained_net_param,
				feature_extraction_proto_file, extract_feature_blob_names,
				save_feature_dataset_names, pad_height, pad_width, half_width);
	} catch(...) {
		success = false;
	}

	result.set_result_type(success?ResultUnit::SUCCESS:ResultUnit::FAILURE);
	return result;
}


void master(std::deque<std::string> &image_lists, caffenet_params &params)
{
	int np, id;
	MPI_Status status;

	/* Find out how many processes there are in the default
     communicator */

	MPI_Comm_size(MPI_COMM_WORLD, &np);

	/* Seed the slaves; send one unit of work to each slave. */

	for (id = 1; id < np; ++id) {

		/* Find the next item of work to do */
		if(!image_lists.empty()) {

			WorkUnit work;
			work = get_next_work_item(image_lists, params);
			/* Send it to each id */
			string work_string;
			work.SerializeToString(&work_string);
			char *msgdata = const_cast<char*>(work_string.c_str());
			int   msgdata_len = strlen(msgdata);
			MPI_Send(msgdata,             /* message buffer */
					msgdata_len,                 /* one data item */
					MPI_CHAR,           /* data item is an integer */
					id,              /* destination process id */
					WORKTAG,           /* user chosen message tag */
					MPI_COMM_WORLD);   /* default communicator */
		}
	}

	/* Loop over getting new work requests until there is no more work
     to be done */

	while(1) {
		WorkUnit work;
		work = get_next_work_item(image_lists, params);
		while(!work.image_filename().empty()) {
			std::cout << "the stack size is " << image_lists.size() << "\n\n";

			const int maximum_msgdata_len = 1024;
			char msgdata[maximum_msgdata_len];
			int msgdata_len;
			/* Receive results from a slave */
			MPI_Recv(&msgdata,           /* message buffer */
					maximum_msgdata_len,                 /* one data item */
					MPI_CHAR,        /* of type double real */
					MPI_ANY_SOURCE,    /* receive from any sender */
					MPI_ANY_TAG,       /* any type of message */
					MPI_COMM_WORLD,    /* default communicator */
					&status);          /* info about the received message */
			MPI_Get_count(&status, MPI_CHAR, &msgdata_len);
			string msg = string(msgdata, msgdata+msgdata_len);
			ResultUnit result;
			result.ParseFromString(msg);


			// process result
			process_results(result, image_lists);

			/* Send the slave a new work unit */
			string work_string;
			work.SerializeToString(&work_string);
			char *workmsgdata = const_cast<char*>(work_string.c_str());
			int   workmsg_len = strlen(workmsgdata);
			MPI_Send(workmsgdata,             /* message buffer */
					workmsg_len,                 /* one data item */
					MPI_CHAR,           /* data item is an integer */
					status.MPI_SOURCE, /* to who we just received from */
					WORKTAG,           /* user chosen message tag */
					MPI_COMM_WORLD);   /* default communicator */

			work = get_next_work_item(image_lists, params);
		}

		/* There's no more work to be done, so receive all the outstanding results from the slaves. */
		for (id = 1; id < np; ++id) {
			const int maximum_msgdata_len = 1024;
			char msgdata[maximum_msgdata_len];
			int msgdata_len;
			/* Receive results from a slave */
			MPI_Recv(&msgdata,           /* message buffer */
					maximum_msgdata_len,                 /* one data item */
					MPI_CHAR,        /* of type double real */
					MPI_ANY_SOURCE,    /* receive from any sender */
					MPI_ANY_TAG,       /* any type of message */
					MPI_COMM_WORLD,    /* default communicator */
					&status);          /* info about the received message */
			MPI_Get_count(&status, MPI_CHAR, &msgdata_len);
			string msg = string(msgdata, msgdata+msgdata_len);
			ResultUnit result;
			result.ParseFromString(msg);

			// process result
			process_results(result, image_lists);
		}

		if(image_lists.empty()) {
			break;
		}

	}

	/* Tell all the slaves to exit by sending an empty message with the
     DIETAG. */

	for (id = 1; id < np; ++id) {
		MPI_Send(0, 0, MPI_INT, id, DIETAG, MPI_COMM_WORLD);
	}
}


void slave(void)
{
	while (1) {

		ResultUnit result;
		MPI_Status status;

		/* Receive a message from the master */
		//		MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG,
		//				MPI_COMM_WORLD, &status);
		const int maximum_msgdata_len = 1024;
		char msgdata[maximum_msgdata_len];
		int msgdata_len;
		/* Receive results from a slave */
		MPI_Recv(&msgdata,           /* message buffer */
				maximum_msgdata_len,                 /* one data item */
				MPI_CHAR,        /* of type double real */
				0,    /* receive from any sender */
				MPI_ANY_TAG,       /* any type of message */
				MPI_COMM_WORLD,    /* default communicator */
				&status);          /* info about the received message */
		MPI_Get_count(&status, MPI_CHAR, &msgdata_len);

		/* Check the tag of the received message. */
		if (status.MPI_TAG == DIETAG) {
			break;
		}

		/* Do the work */
		string msg = string(msgdata, msgdata+msgdata_len);
		WorkUnit work;
		work.ParseFromString(msg);

		result = do_work(work);

		/* Send the result back */
		string result_string;
		result.SerializeToString(&result_string);
		char *result_msgdata = const_cast<char*>(result_string.c_str());
		int   msg_len = strlen(result_msgdata);
		MPI_Send(result_msgdata, msg_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
}



int main(int argc, char **argv) {

	if(argc!=8) {
		std::cout << "The input parameters are as follows:\n";
		std::cout <<
				"\timages_list_filename " <<
				"pretrained_net_param " <<
				"feature_extraction_proto_file " <<
				"extract_feature_blob_names " <<
				"pad_height " <<
				"pad_width " <<
				"half_width " << "\n";
		return -1;
	}

	std::string image_lists_filename = string(argv[1]);

	caffenet_params params;
	params.pretrained_net_param = string(argv[2]);
	params.feature_extraction_proto_file = string(argv[3]);
	params.extract_feature_blob_names = string(argv[4]);
	params.pad_height = atoi(argv[5]);
	params.pad_width = atoi(argv[6]);
	params.half_width = atoi(argv[7]);

	std::ifstream infile(image_lists_filename.c_str());
	std::deque<std::string> image_lists;
	std::string image_filename;
	while (infile >> image_filename) {
		image_lists.push_back(image_filename);
	}
	infile.close();

	// begin the MPI procedure
	int pid;
	/* Initialize MPI */
	MPI_Init(&argc, &argv);

	/* Find out my identity in the default communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	if (pid == 0) {
		master(image_lists, params);
	} else {
		slave();
	}

	/* Shut down MPI */
	MPI_Finalize();
	return 0;
}

