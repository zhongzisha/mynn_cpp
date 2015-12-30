/*
 * layers.hpp
 *
 *  Created on: Dec 27, 2015
 *      Author: ubuntu
 */

#ifndef LAYERS_HPP_
#define LAYERS_HPP_

#include "myproto.pb.h"
#include "db.hpp"
#include "io.hpp"
#include "internal_thread.hpp"
#include "common.hpp"
#include "blob.hpp"

class DataLayerParameter_t
{
public:
	DataLayerParameter_t() {
		backend = "lmdb";
		source = "";
		mean_file = "";
		batch_size = 1;
		crop_size = 0;
		mirror = false;
		scale = 1.0f;
		has_mean_file = false;
		phase = "train";
		cursor_start = 0;
		cursor_step = 1;
	}
	string backend;
	string source;
	string mean_file;
	int batch_size;
	int crop_size;
	bool mirror;
	float scale;
	bool has_mean_file;
	vector<float> mean_values;
	string phase;
	int cursor_start;
	int cursor_step;
};

class DataLayer_t : public InternalThread
{
public:
	DataLayerParameter_t *data_params;
	Blob_t *prefetch_data_;
	Blob_t *prefetch_label_;
	Blob_t *mean_;
	float *mean_data;

	int crop_size;
	float scale;
	bool do_mirror;
	bool has_mean_file;
	bool has_mean_values;
	int datum_channels_;
	int datum_height_;
	int datum_width_;
	int top_height_;
	int top_width_;
	int top_datum_size_;
	int cursor_start;
	int cursor_step;

	string phase;
	boost::shared_ptr<db::DB> db_;
	boost::shared_ptr<db::Cursor> cursor_;

	DataLayer_t(const DataLayerParameter_t *data_params_) {
		data_params = const_cast<DataLayerParameter_t *>(data_params_);

		crop_size = data_params->crop_size;
		scale = data_params->scale;
		do_mirror = data_params->mirror && (rand() % 2);
		has_mean_file = data_params->has_mean_file;
		has_mean_values = data_params->mean_values.size() > 0;
		phase = data_params->phase;
		cursor_start = data_params->cursor_start;
		cursor_step = data_params->cursor_step;

		LOG(INFO) << "crop_size: " << crop_size << "\n"
				<< "scale: " << scale << "\n"
				<< "mirror: " << do_mirror << "\n"
				<< "has_mean_file: " << "\n"
				<< "has_mean_values: " << "\n";

		prefetch_data_ = NULL;
		prefetch_label_ = NULL;
		mean_ = NULL;
		mean_data = NULL;
		datum_channels_ = 0;
		datum_height_ = 0;
		datum_width_ = 0;
		top_height_ = 0;
		top_width_ = 0;
		top_datum_size_ = 0;
	}

	~DataLayer_t() {
		JoinPrefetchThread(); // here, we should stop the final thread, when we delete the class instance
		delete prefetch_data_;
		delete prefetch_label_;
		delete mean_;
	}

	void Setup();

	void SetCursor(const string &key_str);

	void Forward_cpu(Blob_t *top_data, Blob_t *top_label);

	void Forward_to_Network(Blob_t *top_data, Blob_t *top_label);

	void Forward_cpu_multi(vector<Blob_t *> &top_data, vector<Blob_t *> &top_label, vector<int> &batch_sizes);

	void Forward_to_Network_multi(vector<int> &gpus, vector<int> &batch_sizes, vector<Blob_t *> &top_data, vector<Blob_t *> &top_label);

protected:
	void Transform(const cv::Mat& cv_img, float* transformed_data);

	void CreatePrefetchThread();
	void JoinPrefetchThread();
	void InternalThreadEntry();
};



#endif /* LAYERS_HPP_ */
