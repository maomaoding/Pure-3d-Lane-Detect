#include <crmw/crmw.h>
#include <msgs/image.pb.h>
#include <math/extrinsics.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <msgs/inner_camera.h>
//CamOuter2Inner,read msg from protobuf,write msg to innermsg
namespace COWA
{

class CamOuter2Inner : public Component<SensorMsg::Image>
{
private:
    std::shared_ptr<COWA::Writer<SensorMsg::InnerImage>> pub_;
public:
    CamOuter2Inner() {}
    ~CamOuter2Inner() = default;
    bool Init() override
    {
        pub_ = node_->CreateWriter<SensorMsg::InnerImage>("/camera/inner");
        return true;
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
        auto msg_out = std::make_shared<SensorMsg::InnerImage>();
        msg_out->image=frame;
        msg_out->timestamp=msg->timestamp();
        pub_->Write(msg_out);
        //cv::imwrite("1.jpg", img_detected);
    }
};
NODE_REGISTER_COMPONENT(CamOuter2Inner);


// class TestWriter : public Component<InnerMsg>
// {
// private:
//     std::shared_ptr<COWA::Writer<SensorMsg::Image>> pub_;
// public:
//     TestWriter() {}
//     ~TestWriter() = default;
//     bool Init() override
//     {
//         pub_ = node_->CreateWriter<SensorMsg::Image>("/camera/outer");
//         return true;
//     }
//     bool Proc(const std::shared_ptr<InnerMsg> &msg) override
//     {
//         auto msg_in = std::make_shared<InnerMsg>();
//         msg_in->img=msg->img;
//         msg_in->timestamp=msg->timestamp;
//         std::vector<uint8_t> send_buffer;
//         cv::imencode(".jpg", msg->img, send_buffer, std::vector<int>());
//         std::string str_encode(send_buffer.begin(), send_buffer.end());

//         auto msg_out = std::make_shared<SensorMsg::Image>();
//         msg_out->set_timestamp(msg->timestamp);
//         msg_out->set_width(msg->img.cols);
//         msg_out->set_height(msg->img.rows);
//         msg_out->set_data(str_encode);
//         msg_out->set_encoding(SensorMsg::TYPE_JPEG);
//         pub_->Write(msg_out);
//         //cv::imwrite("1.jpg", img_detected);
//     }
// };

// NODE_REGISTER_COMPONENT(TestWriter);
}