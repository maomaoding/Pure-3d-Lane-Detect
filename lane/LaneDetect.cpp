#include <crmw/crmw.h>
#include <msgs/image.pb.h>
#include <msgs/pointcloud.pb.h>
#include <math/extrinsics.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <msgs/inner_camera.h>
#include "lane/PluginFactoryLane.h"
#include "inference/camera/tensorNet.h"
#include "common/frame.h"
#define PROFILE_ENABLE
#include <crmw/profile.h>
#include <math/intrinsics.h>

DEFINE_uint32(LANE_IMG_H, 256, "image x size");
DEFINE_uint32(LANE_IMG_W, 512, "image y size");
DEFINE_uint32(ROIY_START, 80, "image y size");
DEFINE_double(LINE_EXIST_THRESHOLD, 0.3, "confidecnce");
DEFINE_double(LANE_QUALITY, 4, "qulity");
DEFINE_double(CANNY_THRES1, 182, "canny threshold1 value");
DEFINE_double(CANNY_THRES2, 182, "canny threshold2 value");

namespace COWA
{
//         ======================  7
// ||         |    |    |    |         ||
// ||         |    |    |    |         ||
// ||         |    |    |    |         ||
// ||         |    |    |    |         ||
// ||         |    |    |    |         ||
// ||         |    |    |    |         ||
// ||         |    | xx |    |         ||
// ||         |    | xx |    |         ||
// ||         |    | xx |    |         ||
// ||         |    |    |    |         ||
// 1		  2    3    4    5         6
class LaneDetectBase
{
private:
	nvinfer1::IPluginFactory *lane_plugin_factory_;
	TensorNet *tensorNet_;
	DimsCHW dims_input0_;
	DimsCHW dims_output0_;
	DimsCHW dims_output1_;
	float *output0_cpu_ = nullptr;
	float *output1_cpu_ = nullptr;
	float *input0_ = nullptr;
	float *output0_ = nullptr;
	float *output1_ = nullptr;
	void *buffers_[3];
	std::vector<uint8_t> img_data;
	std::string image_name;

	std::vector<cv::Mat> map;
	cv::Mat finalmask;
	cv::Mat intrinsics;
	cv::Mat distortion;
	cv::Mat canny_edge;
	std::vector<std::vector<cv::Point> > contours, canny_edge_contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point2f> pts_raw, pts_rectified;
	std::vector<char> label;
public:
	LaneDetectBase(std::string model_path, cv::Mat ins, cv::Mat dist)
	{
		PROFILE_FILE_START();

		intrinsics = ins;
		distortion = dist;

		lane_plugin_factory_ = new PluginFactoryLane();
		tensorNet_ = new TensorNet(lane_plugin_factory_);
		const char *INPUT0_BLOB_NAME = "data";
		const char *OUTPUT0_BLOB_NAME = "label";
		const char *OUTPUT1_BLOB_NAME = "spatial";

		tensorNet_->LoadNetwork(model_path.c_str());
		dims_input0_ = tensorNet_->getTensorDims(INPUT0_BLOB_NAME);
		dims_output0_ = tensorNet_->getTensorDims(OUTPUT0_BLOB_NAME);
		dims_output1_ = tensorNet_->getTensorDims(OUTPUT1_BLOB_NAME);
		input0_ = tensorNet_->allocateMemory(dims_input0_, (char *)"input0 blob");
		output0_ = tensorNet_->allocateMemory(dims_output0_, (char *)"output0 blob");
		output1_ = tensorNet_->allocateMemory(dims_output1_, (char *)"output1 blob");
		output0_cpu_ = new float[dims_output0_.c() * dims_output0_.h() * dims_output0_.w()];
		output1_cpu_ = new float[dims_output1_.c() * dims_output1_.h() * dims_output1_.w()];

		buffers_[0] = input0_;
		buffers_[1] = output0_;
		buffers_[2] = output1_;

		int size = FLAGS_LANE_IMG_W * FLAGS_LANE_IMG_H * sizeof(float3);
		img_data.resize(size);
		
		map.resize(7);
		for(int i = 0; i < map.size(); i++)
			map[i] = cv::Mat::zeros(cv::Size(FLAGS_LANE_IMG_W, FLAGS_LANE_IMG_H), CV_8UC1);
		finalmask = cv::Mat::zeros(cv::Size(FLAGS_LANE_IMG_W, FLAGS_LANE_IMG_H), CV_8UC1);
		canny_edge = cv::Mat::zeros(cv::Size(FLAGS_LANE_IMG_W, FLAGS_LANE_IMG_H), CV_8UC1);
		contours.reserve(10), hierarchy.reserve(10), pts_raw.reserve(1024);
	}
	~LaneDetectBase()
	{
		PROFILE_FILE_SAVE("ding_time_test.prof");
		delete output0_cpu_;
		delete output1_cpu_;
		delete tensorNet_;
		delete lane_plugin_factory_;
		cudaFree(input0_);
		cudaFree(output0_);
		cudaFree(output1_);
	}

	cv::Mat detect(const cv::Mat &frame, std::shared_ptr<SensorMsg::PointCloud> pts_out)
	{
		PROFILE_FUNCTION();

		PROFILE_BLOCK("preprocess");
		cv::Mat resize_frame;
		cv::Mat frame_roi = frame(cv::Rect(0, FLAGS_ROIY_START, frame.cols, frame.rows - FLAGS_ROIY_START));
		cv::resize(frame_roi, resize_frame, cv::Size(FLAGS_LANE_IMG_W, FLAGS_LANE_IMG_H), (0.0), (0.0), cv::INTER_LINEAR);
		utils::loadImg(resize_frame, (float *)(&img_data[0]),
					   make_float3(126.519748 / 255.0, 131.187286 / 255.0, 127.388270 / 255.0),
					   make_float3(11.248100 / 255.0, 11.453702 / 255.0, 11.2866413 / 255.0));
		cudaMemcpyAsync(input0_, &img_data[0], img_data.size(), cudaMemcpyHostToDevice);
		PROFILE_END_BLOCK();

		PROFILE_BLOCK("inference");
		tensorNet_->imageInference(buffers_, 3, 1);
		PROFILE_END_BLOCK();
		
		PROFILE_BLOCK("memory post");
		cudaDeviceSynchronize();
		cudaMemcpy(output0_cpu_, output0_, dims_output0_.c() * dims_output0_.h() * dims_output0_.w() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(output1_cpu_, output1_, dims_output1_.c() * dims_output1_.h() * dims_output1_.w() * sizeof(float), cudaMemcpyDeviceToHost);
		
		for(auto &i: map)
			memset(i.data, 0, i.cols * i.rows);
		for (int i = 0; i < dims_output1_.h() * dims_output1_.w(); i++)
		{
			float max_arg = 0.0;
			int max_index = 0;

			for (int c = 0; c < dims_output1_.c(); c++)
			{
				float temp_arg = output1_cpu_[c * dims_output1_.h() * dims_output1_.w() + i];
				if (temp_arg > max_arg)
				{
					max_index = c;
					max_arg = temp_arg;
				}
			}
			if (output0_cpu_[max_index - 1] > FLAGS_LINE_EXIST_THRESHOLD)
				map[max_index-1].data[i] = max_index;
		}
		PROFILE_END_BLOCK();


		PROFILE_BLOCK("cluster");
		memset(finalmask.data, 0, finalmask.cols * finalmask.rows);
		memset(canny_edge.data, 0, canny_edge.cols * canny_edge.rows);
		canny_edge_contours.clear();
		hierarchy.clear();
		cv::Canny(resize_frame, canny_edge, FLAGS_CANNY_THRES1, FLAGS_CANNY_THRES2);
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
		cv::morphologyEx(canny_edge, canny_edge, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);
		cv::findContours(canny_edge, canny_edge_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		cv::drawContours(canny_edge, canny_edge_contours, -1, cv::Scalar(1), cv::FILLED);

		finalmask = finalmask + map[0] + map[5];
		for(int i = 1; i < 5; i++)
		{
			contours.clear();
			hierarchy.clear();
			float base_line = 10;

			std::vector<std::vector<cv::Point> > finalcontour(1);
			cv::findContours(map[i], contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for(int c = 0; c < contours.size(); c++)
			{
				cv::RotatedRect box = cv::minAreaRect(contours[c]);
				finalcontour[0] = max(box.size.width, box.size.height) > base_line ? contours[c] : finalcontour[0];
				base_line = max(box.size.width, box.size.height) > base_line ? \
								max(box.size.width, box.size.height) : base_line;
			}
			
			cv::Mat mask = cv::Mat::zeros(resize_frame.size(), CV_8UC1);
			if(finalcontour[0].size() > 0)
				cv::drawContours(mask, finalcontour, -1, cv::Scalar(i+1), cv::FILLED);

			map[i] = mask;
			// canny_edge.copyTo(map[i], mask);
			// finalmask += map[i]*(i+1);
			finalmask += mask;
		}
		finalmask += map[6];
		PROFILE_END_BLOCK();

		PROFILE_BLOCK("result collect");
		pts_raw.clear();
		pts_rectified.clear();
		label.clear();
		cv::Mat argmax_img = cv::Mat::zeros(resize_frame.size(), CV_8UC3);
		for (int i = 0; i < resize_frame.rows * resize_frame.cols; i++)
		{
			if (finalmask.data[i] > 0)
			{
				float u = i % dims_output1_.w() / (double)FLAGS_LANE_IMG_W * frame_roi.cols;
				float v = i / dims_output1_.w() / (double)FLAGS_LANE_IMG_H * frame_roi.rows + FLAGS_ROIY_START;
				pts_raw.emplace_back(cv::Point2f(u, v));
				label.emplace_back(finalmask.data[i]);
				uint8_t colors[] = {
						255, 0, 0,	 // 1
						0, 255, 0,	 // 2
						0, 0, 255,	 // 3
						255, 255, 0, // 4
						127, 127, 0, // 5
						127, 0, 127, // 6
						0, 127, 127, // 7
				};
				argmax_img.data[i * 3] = colors[finalmask.data[i] * 3 - 3];
				argmax_img.data[i * 3 + 1] = colors[finalmask.data[i] * 3 - 2];
				argmax_img.data[i * 3 + 2] = colors[finalmask.data[i] * 3 - 1];
			}
		}

		cv::addWeighted(resize_frame, 0.8, argmax_img, 0.5, 0, resize_frame);
		if (pts_raw.size())
			cv::undistortPoints(pts_raw, pts_rectified, intrinsics, distortion, cv::noArray(), intrinsics);

		pts_out->mutable_point()->Reserve(pts_rectified.size());
		for(int i = 0; i < pts_rectified.size(); ++i)
		{
			cv::Mat p = (cv::Mat_<double>(3, 1) << pts_rectified[i].x, pts_rectified[i].y, 1);
			p = intrinsics.inv() * p;
			auto pt = pts_out->add_point();
			pt->set_x(p.at<double>(0));
			pt->set_y(p.at<double>(1));
			pt->set_z(1);
			pt->set_id(label[i]);
		}
		PROFILE_END_BLOCK();
		return resize_frame;
	}
};
class LaneDetectInterface
{
private:
	std::shared_ptr<COWA::Writer<SensorMsg::Image>> lane_det_pub_;
	std::shared_ptr<COWA::Writer<lidar::Frame>> inner_points_pub_;
	std::shared_ptr<COWA::Writer<SensorMsg::PointCloud>> lane_det_points_pub_;
	std::shared_ptr<LaneDetectBase> detect_;
	util::ObjectPool<SensorMsg::Image>::ObjectPoolPtr proto_img_pool_;
	util::ObjectPool<SensorMsg::PointCloud>::ObjectPoolPtr proto_pt_pool_;
	util::ObjectPool<lidar::Frame>::ObjectPoolPtr inner_pt_pool_;
	std::string camera_type_;
	uint32_t seq_;
	std::vector<cv::Point2f> pts;
public:
	bool Init(std::shared_ptr<COWA::Node> node_)
	{
		seq_=1;
		//get param
		auto param = std::make_shared<COWA::Parameter>(node_);
		//std::string pub_name = param->value<std::string>("output");
		std::string model_path = param->value<std::string>("model");
		camera_type_ = param->value<std::string>("camera");

		//obj pool init 
		//image
		std::function<void(SensorMsg::Image *)> init_img = [this](SensorMsg::Image *image) -> void {
			image->set_frame_id(camera_type_);
		};
		proto_img_pool_ = std::make_shared<util::ObjectPool<SensorMsg::Image>>(48, init_img);
		//outer point clouds
		std::function<void(SensorMsg::PointCloud *)> init_pc = [this](SensorMsg::PointCloud *pc) -> void {
			pc->set_frame_id(camera_type_);
		};
		proto_pt_pool_ = std::make_shared<util::ObjectPool<SensorMsg::PointCloud>>(48, init_pc);

		//get cam param
		COWA::math::Intrinsics x(param, camera_type_);
		detect_ = std::make_shared<LaneDetectBase>(model_path, x.intrinsics, x.distortion);

		//writer init
		std::string debug = param->value<std::string>("debug", "");
		// if(debug != "")
			lane_det_pub_ = node_->CreateWriter<SensorMsg::Image>("/image/debug");
		std::string output = param->value<std::string>("output", "/lane/points");
		lane_det_points_pub_ = node_->CreateWriter<SensorMsg::PointCloud>(output);
		return true;
	}
	bool Proc(cv::Mat& frame, uint64_t timestamp, const std::string& frame_id)
	{
		//write point clouds on camera plane where z = 1
		auto msg_pt_out=proto_pt_pool_->GetObject();
		msg_pt_out->set_timestamp(timestamp);
		msg_pt_out->set_frame_id(frame_id);
		msg_pt_out->set_sequence(seq_);
		msg_pt_out->clear_point();
		cv::Mat img_detected = detect_->detect(frame, msg_pt_out);
		lane_det_points_pub_->Write(msg_pt_out);

		//write image msg
		if(lane_det_pub_ && lane_det_pub_->HasReader())
		{
			auto msg_out = proto_img_pool_->GetObject();
			msg_out->set_timestamp(timestamp);
			msg_out->set_frame_id(frame_id);
			msg_out->set_sequence(seq_++);
			msg_out->set_width(img_detected.cols);
			msg_out->set_height(img_detected.rows);
			msg_out->set_data(img_detected.data, img_detected.cols * img_detected.rows * img_detected.channels());
			msg_out->set_encoding(SensorMsg::BGR8);
			lane_det_pub_->Write(msg_out);
		}
		// cv::imwrite("/datastore/data/hyw/lane_result/"+std::to_string(timestamp)+".jpg", img_detected);
	}
};
/******************InnerMsg without multiprocess support**********/
class InnerLaneDetect : public Component<SensorMsg::InnerImage>,public LaneDetectInterface
{
public:
	bool Init() override
	{
	   return LaneDetectInterface::Init(node_);
	}
	bool Proc(const std::shared_ptr<SensorMsg::InnerImage> &msg) override
	{
		LaneDetectInterface::Proc(msg->image, msg->timestamp, msg->frame_id);
	}
};
NODE_REGISTER_COMPONENT(InnerLaneDetect);

/******************Proto msg with multiprocess support**********/
class LaneDetect : public Component<SensorMsg::Image>,public LaneDetectInterface
{
private:
	
public:
	bool Init() override
	{
		return LaneDetectInterface::Init(node_);
	}
	bool Proc(const std::shared_ptr<SensorMsg::Image> &msg) override
	{
		cv::Mat frame(msg->height(),msg->width(),CV_8UC3);
		if(msg->encoding()==SensorMsg::TYPE_JPEG)
		{
			std::vector<uint8_t> msg_data(msg->data().begin(), msg->data().end());
			frame=cv::imdecode(msg_data, CV_LOAD_IMAGE_COLOR);
		}
		else if(msg->encoding()==SensorMsg::BGR8)
		{
			frame.data=(uchar *)msg->data().data();
		}     
		LaneDetectInterface::Proc(frame, msg->timestamp(), msg->frame_id());
		return true;
	}
};
NODE_REGISTER_COMPONENT(LaneDetect);
} // namespace COWA