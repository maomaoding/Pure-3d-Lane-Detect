#include <iostream>
#include <algorithm>
#include <crmw/crmw.h>
#include <math/extrinsics.h>
#include <Eigen/Core>
#include <msgs/pointcloud.pb.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#define PROFILE_ENABLE
#include <crmw/profile.h>
#include <pose_provider.h>
#include <navsat_conversions.h>
#define ACCEPT_USE_OF_DEPRECATED_PROJ_API_H
#include <proj_api.h>

DEFINE_int32(alpha, 3, "param for Ostu loss");
DEFINE_string(frame_id, "velodyne", "lidar topic name");
DEFINE_int32(scanline_min, 61, "the first scan line index");
DEFINE_int32(scanline_max, 60, "the last scan line index");
DEFINE_int32(minthred, 30, "the minimum threshold");
//for realtime pointmap saving
DEFINE_string(out, std::string("lidarlane.pointmap"), "file path of output file, default is same as pcd path");
DEFINE_bool(embeded_offset, true, "wether the map offset embeded in the last two points");
DEFINE_double(offset_lat, (double)0, "offset x of points, make sense only when embeded_offset is false");
DEFINE_double(offset_lon, (double)0, "offset y of points, make sense only when embeded_offset is false");
DEFINE_double(offset_height, (double)0, "offset z of points, make sense only when embeded_offset is false");
DEFINE_int32(zone, 0, "utm zone for proj, 0 for autonomic calc");
DEFINE_int32(max_intensity, 40, "max intensity of pointcloud, with higher intensity for curb");
DEFINE_int32(min_intensity, 3, "min intensity of pointcloud, with lower intensity for ignore");

int Ostu(std::vector<int> &input, float alpha)
{
	//calculate histogram
	std::vector<int> hist(256, 0);
	for(auto i : input)
		hist[i]++;

	float sumB = 0, varMax = 0, sum = 0;
	int wB = 0, wF = 0, threshold = 0, total = input.size();

	for(int t = 0; t < 256; t++) sum += t * hist[t];

	for(int t = 0; t < 256; t++)
	{
		wB += hist[t];
		if(wB == 0) continue;

		wF = total - wB;
		if(wF == 0) break;

		sumB += static_cast<float>(t * hist[t]);

		float mB = sumB / wB, mF = (sum - sumB) / wF;

		//calculate between class variance
		float varBetween = static_cast<float>(wB) * static_cast<float>(wF) * (mB - mF) * (mB - mF);

		//calculate front class variance
		float varF = 0;
		for(int m = t+1; m < 256; m++)
			varF += hist[m] * (static_cast<float>(m) - mF) * (static_cast<float>(m) - mF);

		float loss = varBetween / (total * total) - alpha * varF / wF;

		varMax = loss >= varMax ? loss : varMax;
		threshold = loss >= varMax ? t : threshold;
	}
	//in case all points belong to the same class
	if(threshold == 0) threshold = 255;

	return threshold;
}

namespace COWA
{

class Lidarlane : public Component<SensorMsg::PointCloud>
{
private:
	std::shared_ptr<Writer<SensorMsg::PointCloud> > out_ptpub_;
	util::ObjectPool<SensorMsg::PointCloud>::ObjectPoolPtr proto_pt_pool_;
	std::string camera_type_;
	// Eigen::Affine3d transform_lidar;

	//pointmap saving param
	Transforms pose;
	Eigen::Affine3d transform_enu;
	bool init_pose_flag = false;
	Eigen::Vector3d ori;
	pcl::PointXYZI pstart_maj, pstart_min;
	SensorMsg::PointMap map;

	//algorithm param
	std::shared_ptr<SensorMsg::PointCloud> final_lane;
	std::vector<int> left_intensity, right_intensity;

public:
	int slideForGettingPoints(SensorMsg::PointCloud &points)
	{
		int w_0 = 10, w_d = 10, i = 0, points_num = points.point().size();

		float xy_thresh = 0.1, z_thresh = 0.08;

		while((i + w_d) < points_num)
		{
			float z_max = points.point(i).z(), z_min = points.point(i).z(), z_dis = 0;
			int idx_ = 0;

			for (int i_ = 0; i_ < w_d; i_++)
			{
				float dis = fabs(points.point(i+i_).z() - points.point(i+i_+1).z());
				if (dis > z_dis) {z_dis = dis; idx_ = i+i_;}
				if (points.point(i+i_).z() < z_min){z_min = points.point(i+i_).z();}
				if (points.point(i+i_).z() > z_max){z_max = points.point(i+i_).z();}
			}

			if (fabs(z_max - z_min) >= z_thresh)
			{
				for (int i_ = 0; i_ < (w_d - 1); i_++)
				{
					float p_dist = sqrt(((points.point(i+i_).y() - points.point(i+1+i_).y())*(points.point(i+i_).y()-points.point(i+1+i_).y()))
					+ ((points.point(i+i_).x() - points.point(i+1+i_).x())*(points.point(i+i_).x() - points.point(i+1+i_).x())));
					if (p_dist >= xy_thresh)
					{
						return i_ + i;							
					}
				}
				return idx_;
			}
			i += w_0;
		}
		return points.point().size()-1;
	}

	bool Init() override
	{
		auto param = std::make_shared<Parameter>(node_);
		auto tf = std::make_shared<math::Extrinsics>(param);
		// if(tf->lookupTransfrom("/base_link", FLAGS_frame_id, &transform_lidar) == false)
		// {
		// 	CRERROR << "tf fail camera"<<"base link to :"<<FLAGS_frame_id;
		// }
		PoseProvider::Instance()->Init(node_);

		camera_type_ = param->value<std::string>("camera");
		camera_type_ = "/hesai/lidar";
		out_ptpub_ = node_->CreateWriter<SensorMsg::PointCloud>("/lidar_lane");

		std::function<void(SensorMsg::PointCloud *)> init_pc = [this](SensorMsg::PointCloud *pc) -> void{
			pc->set_frame_id(camera_type_);
		};
		proto_pt_pool_ = std::make_shared<util::ObjectPool<SensorMsg::PointCloud> >(48, init_pc);

		final_lane = proto_pt_pool_->GetObject();

		left_intensity.reserve(300);
		right_intensity.reserve(300);
		map.mutable_points()->Reserve(700000);
		PROFILE_FILE_START();
		return true;
	}

	bool Proc(const std::shared_ptr<SensorMsg::PointCloud> &pointcloud) override
	{
		PROFILE_FUNCTION();

		//algorithm param
		final_lane->set_timestamp(pointcloud->timestamp());
		final_lane->set_frame_id(pointcloud->frame_id());
		final_lane->clear_point();

		std::vector<SensorMsg::PointCloud> pc_left(FLAGS_scanline_max - FLAGS_scanline_min + 1 + 32);
		std::vector<SensorMsg::PointCloud> pc_right(FLAGS_scanline_max - FLAGS_scanline_min + 1 + 32);

		//pose_provider
		if(PoseProvider::Instance()->GetState(pointcloud->timestamp(), pose) >= 0)
		{
			if(!init_pose_flag)
			{
				init_pose_flag = true;
				ori = pose.translation();
				double lat, Long;
				UTMtoLL(ori.y(), ori.x(), "51N", lat, Long);
				double maj_x = static_cast<int>(lat*1000) / 1000.0;
				double maj_y = static_cast<int>(Long*1000) / 1000.0;
				pstart_maj.x = maj_x;
				pstart_maj.y = maj_y;
				pstart_maj.z = 0;
				pstart_min.x = static_cast<double>(lat-maj_x);
				pstart_min.y = static_cast<double>(Long-maj_y);
				pstart_min.z = 0;
				if(FLAGS_embeded_offset)
				{
					auto conv = [](double x) -> double {
						double k = int(fabs(x) * 1000 + 0.1) * 0.001;
						if(x < 0)
							return -k;
						else
							return k;
					};
					FLAGS_offset_lat = conv(pstart_maj.x) + pstart_min.x;
					FLAGS_offset_lon = conv(pstart_maj.y) + pstart_min.y;
					FLAGS_offset_height = 0;
				}

				if(FLAGS_zone == 0)
					FLAGS_zone = static_cast<int>((FLAGS_offset_lon + 180)/6) + 1;

				char projection[256];
				sprintf(projection, "+proj=utm +zone=%d +ellps=WGS84", FLAGS_zone);
				projPJ pj_latlong, pj_utm;
				if (!(pj_latlong = pj_init_plus("+proj=longlat +datum=WGS84")) ){
					std::cout << ("pj_init_plus error: longlat\n") << std::endl;
					return -1;
				}
				if (!(pj_utm = pj_init_plus(projection)) ){
					std::cout << ("pj_init_plus error: utm\n") << std::endl;
					return -1;
				}

				double offset_x = FLAGS_offset_lon * DEG_TO_RAD;
				double offset_y = FLAGS_offset_lat * DEG_TO_RAD;
				int p = pj_transform(pj_latlong, pj_utm, 1, 1, &offset_x, &offset_y, NULL );
				if(p)
					std::cout << "Error code: " << p << ", Error message: " << pj_strerrno(p) << std::endl;

				map.set_project(projection);
				map.set_offset_x(offset_x);
				map.set_offset_y(offset_y);
			}
			auto func = [this](COWA::Transforms& pose){return COWA::Transforms(pose.translation()-ori, pose.rotation());};
			transform_enu = func(pose).affine3d();
			
			//drivable region
			for(auto pt : pointcloud->point())
			{
				// transform lidar coord to baselink coord(for non baselink coordinate frame)
				// Eigen::Vector4d baselink_pt(4), lidar_pt(4);
				// lidar_pt << pt.x(), pt.y(), pt.z(), 1;
				// baselink_pt = transform_lidar * lidar_pt;
				// pt.set_x(baselink_pt(0));
				// pt.set_y(baselink_pt(1));
				// pt.set_z(baselink_pt(2));

				//32 lidar
				if(pt.id() == 1 && pt.x() > 0 && pt.y() >= 1 && pt.y() <= 10)
				{
					auto tmp = pc_left[pt.ring()].add_point();
					*tmp = pt;
				}
				else if(pt.id() == 1 && pt.x() > 0 && pt.y() < -1 && pt.y() >= -10)
				{
					auto tmp = pc_right[pt.ring()].add_point();
					*tmp = pt;
				}
				//64 lidar
				// if(pt.ring()<=FLAGS_scanline_max && pt.ring()>=FLAGS_scanline_min && pt.x()>0 && pt.y()>0 && pt.id()==0)
				// {
				// 	auto tmp = pc_left[pt.ring()-FLAGS_scanline_min + 32].add_point();
				// 	*tmp = pt;
				// }
				// else if(pt.ring()<=FLAGS_scanline_max && pt.ring()>=FLAGS_scanline_min && pt.x()>0 && pt.y()<0 && pt.id()==0)
				// {
				// 	auto tmp = pc_right[pt.ring()-FLAGS_scanline_min + 32].add_point();
				// 	*tmp = pt;
				// }
			}

			for(int i = 0; i < pc_left.size(); i++)
			{
				if(pc_left[i].point().size() != 0)
				{
					std::sort(pc_left[i].mutable_point()->begin(), pc_left[i].mutable_point()->end(),
						[](SensorMsg::PointXYZIT &a, SensorMsg::PointXYZIT &b){return a.y() < b.y();});
					int leftindex = slideForGettingPoints(pc_left[i]);

					left_intensity.clear();

					for(int t = 0; t < leftindex; t++)
						left_intensity.push_back(pc_left[i].point(t).intensity());

					int threshold = Ostu(left_intensity, FLAGS_alpha);

					for(int t = 0; t < leftindex; t++)
					{
						if(pc_left[i].point(t).intensity() > std::max(threshold, FLAGS_minthred))
						{
							// auto tmp = final_lane->add_point();
							// *tmp = pc_left[i].point(t);

							// transform baselink coord to enu coord
							if(pc_left[i].point(t).intensity() < FLAGS_min_intensity) continue;

							Eigen::Vector4d baselink_pt(4), enu_pt(4);
							baselink_pt << pc_left[i].point(t).x(), pc_left[i].point(t).y(), pc_left[i].point(t).z(), 1;
							enu_pt = transform_enu * baselink_pt;
							auto q = map.add_points();
							q->set_x(enu_pt(0));
							q->set_y(enu_pt(1));
							q->set_z(enu_pt(2));
							q->set_intensity(std::min(static_cast<int>(pc_left[i].point(t).intensity()),
													static_cast<int>(FLAGS_max_intensity)));
							if(pc_left[i].point(t).intensity() > FLAGS_max_intensity)
								q->set_label(COWA::SensorMsg::PointLabel::CRUB);
						}
					}
				}

				if(pc_right[i].point().size() != 0)
				{
					std::sort(pc_right[i].mutable_point()->begin(), pc_right[i].mutable_point()->end(),
						[](SensorMsg::PointXYZIT &a, SensorMsg::PointXYZIT &b){return a.y() > b.y();});
					int rightindex = slideForGettingPoints(pc_right[i]);

					right_intensity.clear();

					for(int t = 0; t < rightindex; t++)
						right_intensity.push_back(pc_right[i].point(t).intensity());

					int threshold = Ostu(right_intensity, FLAGS_alpha);

					for(int t = 0; t < rightindex; t++)
					{
						if(pc_right[i].point(t).intensity() > std::max(threshold, FLAGS_minthred))
						{
							// auto tmp = final_lane->add_point();
							// *tmp = pc_right[i].point(t);

							// transform baselink coord to enu coord
							if(pc_right[i].point(t).intensity() < FLAGS_min_intensity) continue;

							Eigen::Vector4d baselink_pt(4), enu_pt(4);
							baselink_pt << pc_right[i].point(t).x(), pc_right[i].point(t).y(), pc_right[i].point(t).z(), 1;
							enu_pt = transform_enu * baselink_pt;
							auto q = map.add_points();
							q->set_x(enu_pt(0));
							q->set_y(enu_pt(1));
							q->set_z(enu_pt(2));
							q->set_intensity(std::min(static_cast<int>(pc_right[i].point(t).intensity()),
													static_cast<int>(FLAGS_max_intensity)));
							if(pc_right[i].point(t).intensity() > FLAGS_max_intensity)
								q->set_label(COWA::SensorMsg::PointLabel::CRUB);
						}
					}
				}
			}
		}


		// out_ptpub_->Write(final_lane);
		COWA::SetProtoToBinaryFile(map, FLAGS_out);

		return true;
	}
};
NODE_REGISTER_COMPONENT(Lidarlane);
}//namespace COWA