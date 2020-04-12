#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <crmw/crmw.h>
#include <pose_provider.h>
#include <math/extrinsics.h>
#include <math/intrinsics.h>
#include <msgs/image.pb.h>
#include <msgs/pointcloud.pb.h>
#include "lane/PluginFactoryLane.h"
#include "inference/camera/tensorNet.h"

DEFINE_uint32(LANE_IMG_H, 256, "image x size");
DEFINE_uint32(LANE_IMG_W, 512, "image y size");
DEFINE_uint32(ROIY_START, 80, "image y size");
DEFINE_double(LINE_EXIST_THRESHOLD, 0.3, "confidecnce");
DEFINE_string(frame_id_img, "/camera", "img topic name");

template<typename T>
std::vector<T> polyfit_Eigen(const std::vector<T> &xValues, const std::vector<T> &yValues, const int degree)
{
	int numCoefficients = degree + 1;
	size_t nCount = xValues.size();

	Eigen::MatrixXf X(nCount, numCoefficients);
	Eigen::MatrixXf Y(nCount, 1);

	for(size_t i = 0; i < nCount; i++)
		Y(i,0) = yValues[i];

	for(size_t nRow = 0; nRow < nCount; nRow++)
	{
		T nVal = 1.0f;
		for(int nCol = 0; nCol < numCoefficients; nCol++)
		{
			X(nRow, nCol) = nVal;
			nVal *= xValues[nRow];
		}
	}

	Eigen::VectorXf coefficients;
	coefficients = X.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
	return std::vector<T>(coefficients.data(), coefficients.data() + numCoefficients);
}

template<typename T>
T polyeval(std::vector<T> &coefficient, T x)
{
	T result = 0.;
	for(int i = 0; i < coefficient.size(); i++)
	{
		result += coefficient[i]*pow(x, i);
	}
	return result;
}


SensorMsg::PointXYZIT slideForGettingPoints(std::vector<SensorMsg::PointXYZIT> &points)
{
	int w_0 = 10, w_d = 10, i = 0, points_num = points.size();

	float xy_thresh = 0.1, z_thresh = 0.04;

	while((i + w_d) < points_num)
	{
		float z_max = points.[i].z(), z_min = points.[i].z(), z_dis = 0;
		int idx_ = 0;

		for (int i_ = 0; i_ < w_d; i_++)
		{
			float dis = fabs(points.[i+i_].z() - points.[i+i_+1].z());
			if (dis > z_dis) {z_dis = dis; idx_ = i+i_;}
			if (points[i+i_].z() < z_min){z_min = points[i+i_].z();}
			if (points[i+i_].z() > z_max){z_max = points[i+i_].z();}
		}

		if (fabs(z_max - z_min) >= z_thresh)
		{
			for (int i_ = 0; i_ < (w_d - 1); i_++)
			{
				float p_dist = sqrt(((points[i+i_].y() - points[i+1+i_].y())*(points[i+i_].y()-points[i+1+i_].y()))
				+ ((points[i+i_].x() - points[i+1+i_].x())*(points[i+i_].x() - points[i+1+i_].x())));
				if (p_dist >= xy_thresh)
				{
					return points[i_ + i];
				}
			}
			return points[idx_];
		}
		i += w_0;
	}
	SensorMsg::PointXYZIT tmp;
	tmp.set_intensity(-1);
	return tmp;
}

class LaneDetectBase
{
private:
	std::shared_ptr<nvinfer1::IPluginFactory> lane_plugin_factory_;
	std::shared_ptr<TensorNet> tensornet_;
	DimsCHW dims_input0_, dims_output0_, dims_output1_;
	float *output0_cpu_=nullptr, *output1_cpu_=nullptr, *input0_=nullptr, *output0_=nullptr, *output1_=nullptr;
	void *buffers_[3];
	std::vector<uint8_t> img_data;
	cv::Mat intrinsics, distortion;

	std::vector<cv::Point2f> pts_raw, pts_left, pts_right;

public:
	LaneDetectBase(std::string model_path, cv::Mat ins, cv::Mat dist)
	{
		intrinsics = ins, distortion = dist;
		lane_plugin_factory_ = std::make_shared<PluginFactoryLane>();
		tensornet_ = std::make_shared<TensorNet>(lane_plugin_factory_.get());

		const char *INPUT0_BLOB_NAME = "data", *OUTPUT0_BLOB_NAME = "label", *OUTPUT1_BLOB_NAME = "spatial";
		tensornet_->LoadNetwork(model_path.c_str());
		dims_input0_ = tensornet_->getTensorDims(INPUT0_BLOB_NAME);
		dims_output0_ = tensornet_->getTensorDims(OUTPUT0_BLOB_NAME);
		dims_output1_ = tensornet_->getTensorDims(OUTPUT1_BLOB_NAME);
		input0_ = tensornet_->allocateMemory(dims_input0_, (char *)"input0 blob");
		output0_ = tensornet_->allocateMemory(dims_output0_, (char *)"output0 blob");
		output1_ = tensornet_->allocateMemory(dims_output1_, (char *)"output1 blob");
		output0_cpu_ = new float[dims_output0_.c() * dims_output0_.h() * dims_output0_.w()];
		output1_cpu_ = new float[dims_output1_.c() * dims_output1_.h() * dims_output1_.w()];

		buffers_[0] = input0_;
		buffers_[1] = output0_;
		buffers_[2] = output1_;

		int size = FLAGS_LANE_IMG_W * FLAGS_LANE_IMG_H * sizeof(float3);
		img_data.resize(size);

		pts_raw.reserve(10000);
		pts_left.reserve(5000);
		pts_right.reserve(5000);
	}
	~LaneDetectBase()
	{
		delete output0_cpu_;
		delete output1_cpu_;
		cudaFree(input0_);
		cudaFree(output0_);
		cudaFree(output1_);
	}

	void detect(cv::Mat &frame, std::vector<cv::Point2f> &pts_rec, int &left_curbsize)
	{
		cv::Mat resize_frame;
		cv::Mat frame_roi = frame(cv::Rect(0, FLAGS_ROIY_START, frame.cols, frame.rows - FLAGS_ROIY_START));
		cv::resize(frame_roi, resize_frame, cv::Size(FLAGS_LANE_IMG_W, FLAGS_LANE_IMG_H),(0.0),(0.0), cv::INTER_LINEAR);
		utils::loadImg(resize_frame, (float *)(&img_data[0]),
					   make_float3(126.519748 / 255.0, 131.187286 / 255.0, 127.388270 / 255.0),
					   make_float3(11.248100 / 255.0, 11.453702 / 255.0, 11.2866413 / 255.0));
		cudaMemcpyAsync(input0_, &img_data[0], img_data.size(), cudaMemcpyHostToDevice);

		tensornet_->imageInference(buffers_, 3, 1);

		cudaDeviceSynchronize();
		cudaMemcpy(output0_cpu_, output0_,
			dims_output0_.c() * dims_output0_.h() * dims_output0_.w() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(output1_cpu_, output1_,
			dims_output1_.c() * dims_output1_.h() * dims_output1_.w() * sizeof(float), cudaMemcpyDeviceToHost);

		pts_raw.clear();
		pts_left.clear();
		pts_right.clear();
		left_curbsize = 0;
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

			if (max_index == 1 && output0_cpu_[max_index - 1] > FLAGS_LINE_EXIST_THRESHOLD)
			{
				float u = i % dims_output1_.w() / static_cast<double>(FLAGS_LANE_IMG_W) * frame_roi.cols;
				float v = i / dims_output1_.w() / static_cast<double>(FLAGS_LANE_IMG_H) * frame_roi.rows
					+ FLAGS_ROIY_START;
				pts_left.emplace_back(cv::Point2f(u, v));
			}
			if (max_index == 6 && output0_cpu_[max_index - 1] > FLAGS_LINE_EXIST_THRESHOLD)
			{
				float u = i % dims_output1_.w() / static_cast<double>(FLAGS_LANE_IMG_W) * frame_roi.cols;
				float v = i / dims_output1_.w() / static_cast<double>(FLAGS_LANE_IMG_H) * frame_roi.rows
					+ FLAGS_ROIY_START;
				pts_right.emplace_back(cv::Point2f(u, v));
			}
		}
		left_curbsize = pts_left.size();
		pts_raw.insert(pts_raw.end(), pts_left.begin(), pts_left.end());
		pts_raw.insert(pts_raw.end(), pts_right.begin(), pts_right.end());
		if(!pts_raw.empty())
			cv::undistortPoints(pts_raw, pts_rec, intrinsics, distortion, cv::noArray(), intrinsics);

		//for debug img
		// if(xValues_l.size() >= 2)
		// 	left_coeffi = polyfit_Eigen<float>(xValues_l, yValues_l, 3);
		// if(xValues_r.size() >= 2)
		// 	right_coeffi = polyfit_Eigen<float>(xValues_r, yValues_r, 2);
		// int min = *min_element(xValues_r.begin(), xValues_r.end());
		// int max = *max_element(xValues_r.begin(), xValues_r.end());
		// for(int i = min; i < max; i++)
		// {
		// 	int j = polyeval(right_coeffi, static_cast<float>(i));
		// 	if(j<1080)
		// 	{
		// 		frame.at<cv::Vec3b>(j,i)[0] = 0;
		// 		frame.at<cv::Vec3b>(j,i)[1] = 0;
		// 		frame.at<cv::Vec3b>(j,i)[2] = 255;
		// 	}
		// }
	}
};

class curbdetect: public Component<SensorMsg::Image, SensorMsg::PointCloud>
{
private:
	std::shared_ptr<LaneDetectBase> detect_;

	//publish for debug
	util::ObjectPool<SensorMsg::Image>::ObjectPoolPtr proto_img_pool_;
	util::ObjectPool<SensorMsg::PointCloud>::ObjectPoolPtr proto_pc_pool_;
	std::shared_ptr<SensorMsg::PointCloud> curb;

	std::shared_ptr<Writer<SensorMsg::Image> > img_pub_;
	std::shared_ptr<Writer<SensorMsg::PointCloud> > pc_pub_;

	std::vector<double> left_coeffi, right_coeffi;
	std::vector<cv::Point2f> pts_rec;
	int left_curbsize = 0;

	Transforms img_pose, pc_pose;
	Eigen::Affine3d transforms_sync, transform_img;
	Eigen::MatrixXd baselink2camera = Eigen::MatrixXd(3,4);

	//for line fitting
	std::vector<double> xValues_r, yValues_r, xValues_l, yValues_l;

public:
	bool Init() override
	{
		//get param
		auto param = std::make_shared<Parameter>(node_);
		auto tf = std::make_shared<math::Extrinsics>(param);
		if(tf->lookupTransfrom("/base_link", FLAGS_frame_id_img, &transform_img) == false)
			CRERROR << "tf fail camera" << "base link to :" << FLAGS_frame_id_img;
		std::string model_path = param->value<std::string>("model");
		std::string camera_type_ = param->value<std::string>("camera");

		//obj pool init
		//image
		std::function<void(SensorMsg::Image*)> init_img = [&](SensorMsg::Image *image){
			image->set_frame_id(camera_type_);
		};
		proto_img_pool_ = std::make_shared<util::ObjectPool<SensorMsg::Image> >(48, init_img);
		//point clouds
		std::function<void(SensorMsg::PointCloud*)> init_pc = [&](SensorMsg::PointCloud *pc){
			pc->set_frame_id(camera_type_);
		};
		proto_pc_pool_ = std::make_shared<util::ObjectPool<SensorMsg::PointCloud> >(48, init_pc);
		curb = proto_pc_pool_->GetObject();

		//get cam param and pose to init model class
		PoseProvider::Instance()->Init(node_);
		math::Intrinsics x(param, camera_type_);
		detect_ = std::make_shared<LaneDetectBase>(model_path, x.intrinsics, x.distortion);
		//initialize baselink to camera pixel transform
		Eigen::Matrix3d intrinsics_eigen;
		cv::cv2eigen(x.intrinsics, intrinsics_eigen);
		baselink2camera.block<3,3>(0,0) = intrinsics_eigen;
		baselink2camera.block<3,1>(0,3) = Eigen::Vector3d(0,0,0);
		baselink2camera = baselink2camera * transform_img.matrix().inverse();

		//writer init
		img_pub_ = node_->CreateWriter<SensorMsg::Image>("/image/debug");
		pc_pub_ = node_->CreateWriter<SensorMsg::PointCloud>("/curb/points");

		pts_rec.reserve(10000);
		xValues_r.reserve(10000);
		yValues_r.reserve(10000);
		xValues_l.reserve(10000);
		yValues_l.reserve(10000);
		return true;
	}

	bool Proc(const std::shared_ptr<SensorMsg::Image> &img_msg, const std::shared_ptr<SensorMsg::PointCloud> &pc_msg) override
	{
		cv::Mat frame(img_msg->height(), img_msg->width(), CV_8UC3);
		if(img_msg->encoding() == SensorMsg::TYPE_JPEG)
		{
			std::vector<uint8_t> msg_data(img_msg->data().begin(), img_msg->data().end());
			frame = cv::imdecode(msg_data, CV_LOAD_IMAGE_COLOR);
		}
		else if(img_msg->encoding() == SensorMsg::BGR8)
			frame.data = (uchar *)img_msg->data().data();

		left_coeffi.clear();
		right_coeffi.clear();
		xValues_r.clear();
		yValues_r.clear();
		if(PoseProvider::Instance()->GetState(img_msg->timestamp(), img_pose) >= 0 &&
			PoseProvider::Instance()->GetState(pc_msg->timestamp(), pc_pose) >= 0)
		{
			transforms_sync = pc_pose.affine3d().inverse() * img_pose.affine3d();
			detect_->detect(frame, pts_rec, left_curbsize);

			//calculate the baselink point at camera timestamp(assuming z=0)
			Eigen::Matrix3d fullrank_trans;
			fullrank_trans << baselink2camera.block<3,2>(0,0), baselink2camera.block<3,1>(0,3);
			curb->set_timestamp(pc_msg->timestamp());
			curb->set_frame_id(pc_msg->frame_id());
			curb->clear_point();
			for(int i = 0; i < pts_rec.size(); i++)
			{
				Eigen::Vector3d input(pts_rec[i].x, pts_rec[i].y, 1), output;
				try{
					output = fullrank_trans.inverse() * input;
				}
				catch(...){
					std::cout<<"reverse projection error, maybe sth wrong happens in the inverse()"<<std::endl;
				}

				//calculate the baselink point at lidar timestamp
				Eigen::Vector4d pt_cameratime(output(0) / output(2), output(1) / output(2), 0, 1);
				Eigen::Vector4d pt_lidartime;
				pt_lidartime = transforms_sync * pt_cameratime;
				if(i > left_curbsize)
				{
					xValues_r.emplace_back(pt_lidartime(0));
					yValues_r.emplace_back(pt_lidartime(1));
				}
			}

			if(xValues_r.size() >= 2)
				right_coeffi = polyfit_Eigen<double>(xValues_r, yValues_r, 2);
			// double min = *min_element(xValues_r.begin(), xValues_r.end());
			// double max = *max_element(xValues_r.begin(), xValues_r.end());
			// for(double i = min; i < max; i+=0.01)
			// {
			// 	double j = polyeval(right_coeffi, i);
			// 	// for debug pointcloud
			// 	auto tmp = curb->add_point();
			// 	tmp->set_x(i);
			// 	tmp->set_y(j);
			// 	tmp->set_z(0);
			// 	tmp->set_intensity(255);
			// }

			refine_curb(curb, pc_msg, left_coeffi, right_coeffi);
		}

		//for debug pointcloud
		pc_pub_->Write(curb);

		//for debug img
		// auto msg_out = proto_img_pool_->GetObject();
		// msg_out->set_timestamp(img_msg->timestamp());
		// msg_out->set_frame_id(img_msg->frame_id());
		// msg_out->set_width(frame.cols);
		// msg_out->set_height(frame.rows);
		// msg_out->set_data(frame.data, frame.cols * frame.rows * frame.channels());
		// msg_out->set_encoding(SensorMsg::BGR8);
		// img_pub_->Write(msg_out);
		return true;
	}

	void refine_curb(std::shared_ptr<SensorMsg::PointCloud> &curb, const std::shared_ptr<SensorMsg::PointCloud> &pc_msg,
					std::vector<double> &left_coeffi, std::vector<double> &right_coeffi)
	{
		std::vector<std::vector<SensorMsg::PointXYZIT> > lines(32);
		for(int i = 0; i < lines.size(); i++)
			lines[i].reserve(30);
		for(auto pt : pc_msg)
		{
			double rangel = polyeval(right_coeffi, pt.x()) - 1, ranger = polyeval(right_coeffi, pt.x()) + 1;
			if(pt.x() > 0 && pt.y() >= rangel && pt.y() <= ranger)
				lines[pt.ring()].emplace_back(pt);
		}
		for(int i = 0; i < lines.size(); i++)
		{
			std::sort(lines[i].begin(), lines[i].end(),
				[](SensorMsg::PointXYZIT &a, SensorMsg::PointXYZIT &b){return a.y() > b.y();});
			SensorMsg::PointXYZIT curb_pt = slideForGettingPoints(lines[i]);
			if(curb_pt.intensity() != -1)
			{
				auto tmp = curb->add_point();
				*tmp = curb_pt;
			}
		}
	}
};

NODE_REGISTER_COMPONENT(curbdetect);
