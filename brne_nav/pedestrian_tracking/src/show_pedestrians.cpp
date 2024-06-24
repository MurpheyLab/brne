#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "image_geometry/stereo_camera_model.h"
#include "sensor_msgs/msg/camera_info.hpp"
#include "crowd_nav_interfaces/msg/pedestrian.hpp"
#include "crowd_nav_interfaces/msg/pedestrian_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"

using namespace std::chrono_literals;

class ShowPedestrians : public rclcpp::Node
{
public:
  ShowPedestrians()
  : Node("show_pedestrians")
  {

    pedestrian_sub_ = create_subscription<crowd_nav_interfaces::msg::PedestrianArray>(
      "pedestrians", 10, std::bind(&ShowPedestrians::pedestrians_cb, this, std::placeholders::_1));
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("ped_markers", 10);
  }

private:
  rclcpp::Subscription<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr pedestrian_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  visualization_msgs::msg::MarkerArray ma;

  void pedestrians_cb(const crowd_nav_interfaces::msg::PedestrianArray & msg)
  {
    // issue: if the pedestrian disappears, the marker will stay.
    // This is solved by making the lifetime small.
    const auto current_time = this->get_clock()->now();
    // clear previous marker array
    ma.markers.clear();
    const int n_pedestrians = msg.pedestrians.size();
    const auto alpha = 1.0;
    const auto height = 0.01;
    const auto diameter = 0.1;
    const auto arrow_thickness = 0.05;
    const auto arrow_alpha = 0.4;
    const auto lifetime_sec = 1;
    for (int i = 0; i < n_pedestrians; i++) {
      const auto ped = msg.pedestrians.at(i);
      visualization_msgs::msg::Marker m;
      m.header.stamp = current_time;
      m.header.frame_id = "brne_odom";
      m.id = 2*i;   // ped.id;
      m.type = 3;         // cylinder
      m.action = 0;       // add/modify
      // Set color as orange
      m.color.a = alpha;
      m.color.r = 1.0;
      m.color.g = 0.67;
      m.color.b = 0.0;
      // Set Radius
      m.scale.x = diameter;
      m.scale.y = diameter;
      m.scale.z = height;
      m.pose.position.x = ped.pose.position.x;
      m.pose.position.y = ped.pose.position.y;
      m.pose.position.z = 0.5*height;
      m.lifetime.sec = lifetime_sec;
      // Add to marker array
      ma.markers.push_back(m);

      // add an arrow marker
      const auto vel_x = ped.velocity.linear.x;
      const auto vel_y = ped.velocity.linear.y;
      const auto vel_abs = sqrt(pow(vel_x, 2) + pow(vel_y, 2));
      visualization_msgs::msg::Marker arrow;
      arrow.header.frame_id = "brne_odom";
      arrow.header.stamp = current_time;
      arrow.id = 2*i + 1;
      arrow.type = 0;   // arrow
      arrow.action = 0;
      // pose is the end of the arrow
      arrow.pose.position.x = ped.pose.position.x;
      arrow.pose.position.y = ped.pose.position.y;
      arrow.pose.position.z = 0.01;
      // orientation should be the velocity direction
      const auto phi = atan2(vel_y, vel_x);
      tf2::Quaternion q;
      q.setRPY(0, 0, phi);
      arrow.pose.orientation.w = q.w();
      arrow.pose.orientation.x = q.x();
      arrow.pose.orientation.y = q.y();
      arrow.pose.orientation.z = q.z();
      // scale is the size of the arrow, proportional to velocity
      arrow.scale.x = vel_abs + 0.5*diameter;
      arrow.scale.y = arrow_thickness;
      arrow.scale.z = 0.01;
      arrow.color.a = arrow_alpha;
      arrow.color.r = 1.0;
      arrow.color.g = 0.67;
      arrow.color.b = 0.0;
      arrow.lifetime.sec = 1;
      ma.markers.push_back(arrow);
    }
    marker_pub_->publish(ma);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ShowPedestrians>());
  rclcpp::shutdown();
  return 0;
}
