
#include "data_layer.hpp"
//#include "myproto.pb.h"
//#include "common.hpp"
//#include "blob.hpp"

void DataLayer_t::Setup() {
	// Initialize DB
	db_.reset(db::GetDB(data_params->backend));
	db_->OpenForReadOnly(data_params->source, db::READ);
	cursor_.reset(db_->NewCursor());

	// Read a data point, and use it to initialize the top blob.
	Datum datum;
	datum.ParseFromString(cursor_->value());

	if(datum.encoded()) {
		cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
		datum_channels_ = cv_img.channels();
		datum_height_ = cv_img.rows;
		datum_width_ = cv_img.cols;
	} else {
		datum_channels_ = datum.channels();
		datum_height_ = datum.height();
		datum_width_ = datum.width();
	}


	LOG(INFO) << "datum height_: " << datum_height_ << "\n"
			<< "datum_width_: " << datum_width_ << "\n";

	if(crop_size && crop_size < datum_height_ && crop_size < datum_width_) {
		prefetch_data_ = new Blob_t(data_params->batch_size, datum_channels_, crop_size, crop_size);
		prefetch_label_ = new Blob_t(data_params->batch_size, 1, 1, 1);

	} else {
		prefetch_data_ = new Blob_t(data_params->batch_size, datum_channels_, datum_height_, datum_width_);
		prefetch_label_ = new Blob_t(data_params->batch_size, 1, 1, 1);
	}
	top_height_ = prefetch_data_->H;
	top_width_ = prefetch_data_->W;
	top_datum_size_ = top_height_ * top_width_ * prefetch_data_->C;
	prefetch_data_->allocate_cpu_data();
	prefetch_label_->allocate_cpu_data();
	LOG(INFO) << "prefetch_data_ size: "
			<< prefetch_data_->N << ", "
			<< prefetch_data_->C << ", "
			<< prefetch_data_->H << ", "
			<< prefetch_data_->W;

	if(has_mean_file) {
		mean_ = new Blob_t(1, datum_channels_, datum_height_, datum_width_);
		mean_->allocate_cpu_data();
		mean_data = mean_->data_cpu;
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(data_params->mean_file.c_str(), &blob_proto);
		for (int i = 0; i < mean_->count(); ++i) {
			mean_data[i] = (float)blob_proto.data(i);
		}
		LOG(INFO) << "mean_ size: "
				<< mean_->N << ", "
				<< mean_->C << ", "
				<< mean_->H << ", "
				<< mean_->W;
	}

	CreatePrefetchThread();
}

void DataLayer_t::SetCursor(const string& key_str) {
	if(data_params->backend != "rocksdb") {
		return;
	}
	if (db_ == NULL) {
		db_.reset(db::GetDB(data_params->backend));
		db_->Open(data_params->source, db::READ);
		cursor_.reset(db_->NewCursor());
	}
	cursor_->Seek(key_str);
}

void DataLayer_t::Forward_cpu(Blob_t *top_data, Blob_t *top_label) {
	// printf("First, join the thread.\n");
	JoinPrefetchThread();

	// printf("copy data to top_data.\n");
	memcpy(top_data->data_cpu, prefetch_data_->data_cpu, prefetch_data_->count() * sizeof(float));

	// printf("copy label to top_label.\n");
	memcpy(top_label->data_cpu, prefetch_label_->data_cpu, prefetch_label_->count() * sizeof(float));

	// printf("Start a new prefetch thread.\n");
	CreatePrefetchThread();
}

void DataLayer_t::Forward_to_Network(Blob_t *top_data, Blob_t *top_label) {

	JoinPrefetchThread();

	CUDA_CHECK( cudaMemcpy(top_data->data_gpu, prefetch_data_->data_cpu, prefetch_data_->count() * sizeof(float), cudaMemcpyDefault) );

	CUDA_CHECK( cudaMemcpy(top_label->data_gpu, prefetch_label_->data_cpu, prefetch_label_->count() * sizeof(float), cudaMemcpyDefault) );

	CreatePrefetchThread();
}

void DataLayer_t::Forward_cpu_multi(vector<Blob_t *> &top_data, vector<Blob_t *> &top_label, vector<int> &batch_sizes) {
	// printf("First, join the thread.\n");
	JoinPrefetchThread();

	for(int i = 0; i < batch_sizes.size(); i++) {
		int start_index = 0;
		for(int j = 0; j < i; j++) {
			start_index += batch_sizes[j];
		}
		// printf("copy data to top_data.\n");
		memcpy(top_data[i]->data_cpu,
				prefetch_data_->data_cpu + start_index * top_data[i]->C * top_data[i]->H * top_data[i]->W,
				top_data[i]->count() * sizeof(float));

		// printf("copy label to top_label.\n");
		memcpy(top_label[i]->data_cpu,
				prefetch_label_->data_cpu + start_index * top_label[i]->C * top_label[i]->H * top_label[i]->W,
				top_label[i]->count() * sizeof(float));
	}
	// printf("Start a new prefetch thread.\n");
	CreatePrefetchThread();
}

void DataLayer_t::Forward_to_Network_multi(vector<int> &gpus, vector<int> &batch_sizes, vector<Blob_t *> &top_data, vector<Blob_t *> &top_label) {
	JoinPrefetchThread();

	for(int i = 0; i < batch_sizes.size(); i++) {
		int start_index = 0;
		for(int j = 0; j < i; j++) {
			start_index += batch_sizes[j];
		}

		cudaSetDevice(gpus[i]);
		CUDA_CHECK( cudaMemcpy(top_data[i]->data_gpu,
				prefetch_data_->data_cpu + start_index * top_data[i]->C * top_data[i]->H * top_data[i]->W,
				top_data[i]->count() * sizeof(float), cudaMemcpyDefault) );

		CUDA_CHECK( cudaMemcpy(top_label[i]->data_gpu,
				prefetch_label_->data_cpu + start_index * top_label[i]->C * top_label[i]->H * top_label[i]->W,
				top_label[i]->count() * sizeof(float), cudaMemcpyDefault) );
	}
	CreatePrefetchThread();
}

void DataLayer_t::Transform(const cv::Mat& cv_img, float* transformed_data) {
	const int img_channels = cv_img.channels();
	const int img_height = cv_img.rows;
	const int img_width = cv_img.cols;

	int h_off = 0;
	int w_off = 0;
	cv::Mat cv_cropped_img = cv_img;
	if (crop_size) {
		// We only do random crop when we do training.
		if (phase == "train") {
			h_off = rand() % (img_height - crop_size + 1);
			w_off = rand() % (img_width - crop_size + 1);
		} else {
			h_off = (img_height - crop_size) / 2;
			w_off = (img_width - crop_size) / 2;
		}
		cv::Rect roi(w_off, h_off, crop_size, crop_size);
		cv_cropped_img = cv_img(roi);
	}

	CHECK(cv_cropped_img.data);

	int top_index;
	for (int h = 0; h < top_height_; ++h) {
		const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < top_width_; ++w) {
			for (int c = 0; c < datum_channels_; ++c) {
				if (do_mirror) {
					top_index = (c * top_height_ + h) * top_width_ + (top_width_ - 1 - w);
				} else {
					top_index = (c * top_height_ + h) * top_width_ + w;
				}
				// int top_index = (c * height + h) * width + w;
				float pixel = static_cast<float>(ptr[img_index++]);
				if (has_mean_file) {
					int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
					transformed_data[top_index] = (pixel - mean_data[mean_index]) * scale;
				} else {
					if (has_mean_values) {
						transformed_data[top_index] = (pixel - data_params->mean_values[c]) * scale;
					} else {
						transformed_data[top_index] = pixel * scale;
					}
				}
			}
		}
	}
}

void DataLayer_t::CreatePrefetchThread() {
	CHECK(StartInternalThread()) << "Thread execution failed";
}
void DataLayer_t::JoinPrefetchThread() {
	CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}
void DataLayer_t::InternalThreadEntry(){

	float *top_data = prefetch_data_->data_cpu;
	float *top_label = prefetch_label_->data_cpu;
	for (int item_id = 0; item_id < data_params->batch_size; ++item_id) {

		// get a blob
		Datum datum;
		datum.ParseFromString(cursor_->value());

		if (datum.encoded()) {
			cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
			// data augmentation
			Transform(cv_img, top_data + item_id * top_datum_size_);
		} else {
			const string& data = datum.data();
			if(crop_size && crop_size < datum_height_ && crop_size < datum_width_) {
				int h_offset = rand() % (datum_height_ - crop_size);
				int w_offset = rand() % (datum_width_ - crop_size);
				if(data.size()) {
					for (int c = 0; c < datum_channels_; ++c) {
						for (int h = 0; h < crop_size; ++h) {
							for (int w = 0; w < crop_size; ++w) {
								int index = (c * datum_height_ + h + h_offset) * datum_width_ + w + w_offset;
								top_data[((item_id * datum_channels_ + c) * crop_size + h) * crop_size + w] =
										static_cast<float>((uint8_t)data[index]) - mean_data[index];
							}
						}
					}
				} else {
					for (int c = 0; c < datum_channels_; ++c) {
						for (int h = 0; h < crop_size; ++h) {
							for (int w = 0; w < crop_size; ++w) {
								int index = (c * datum_height_ + h + h_offset) * datum_width_ + w + w_offset;
								top_data[((item_id * datum_channels_ + c) * crop_size + h) * crop_size + w] =
										data[index] - mean_data[index];
							}
						}
					}
				}
			} else {
				if (data.size()) {
					for (int j = 0; j < top_datum_size_; ++j) {
						top_data[item_id * top_datum_size_ + j] = (static_cast<float>((uint8_t)data[j])) - mean_data[j];
					}
				} else {
					for (int j = 0; j < top_datum_size_; ++j) {
						top_data[item_id * top_datum_size_ + j] = (datum.float_data(j)) - mean_data[j];
					}
				}
			}
		}

		// read the label
		top_label[item_id] = datum.label();

		// go to the next iter
		cursor_->Next();
		if (!cursor_->valid()) {
			cursor_->SeekToFirst();
		}
	}
}
