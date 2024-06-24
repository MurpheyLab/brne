#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <armadillo>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "crowd_nav_interfaces/msg/pedestrian.hpp"
#include "crowd_nav_interfaces/msg/pedestrian_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "zed_interfaces/msg/objects_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"


using namespace std::chrono_literals;

class ConvertPeds : public rclcpp::Node
{
public:
  ConvertPeds()
  : Node("convert_zed"), n_prev_peds{0}
  {
    declare_parameter("vel_cutoff", 1.0); // m/s
    vel_cutoff = get_parameter("vel_cutoff").as_double();

    zed_sub_ = create_subscription<zed_interfaces::msg::ObjectsStamped>(
      "zed/zed_node/obj_det/objects", 10,
      std::bind(&ConvertPeds::zed_cb, this, std::placeholders::_1));
    pedestrian_pub_ =
      create_publisher<crowd_nav_interfaces::msg::PedestrianArray>("pedestrians", 10);

    // From: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/
    // Writing-A-Tf2-Broadcaster-Cpp.html
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    brne_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("brne_odom", 10);

    double rate_hz = 50;
    std::chrono::milliseconds rate = (std::chrono::milliseconds) ((int)(1000. / rate_hz));

    timer_ = create_wall_timer(rate, std::bind(&ConvertPeds::timer_callback, this));

    brne_odom.header.frame_id = "brne_odom";
    brne_odom.child_frame_id = "brne";
  }

private:
  double rate_hz;
  double vel_cutoff;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Subscription<zed_interfaces::msg::ObjectsStamped>::SharedPtr zed_sub_;
  rclcpp::Publisher<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr pedestrian_pub_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr brne_odom_pub_;
  nav_msgs::msg::Odometry brne_odom;

  arma::mat ped_array;
  arma::mat prev_ped_array;
  int n_prev_peds;
  int n_peds;

  rclcpp::Time prev_ped_stamp;
  rclcpp::Time ped_stamp;

  void zed_cb(const zed_interfaces::msg::ObjectsStamped & msg)
  {
    crowd_nav_interfaces::msg::PedestrianArray pa;
    ped_stamp = msg.header.stamp; // this->get_clock()->now();

    pa.header.stamp = ped_stamp;
    const int n_peds_msg = msg.objects.size();
    // RCLCPP_INFO_STREAM(get_logger(), "\n\nMESSAGE");
    for (int i = 0; i < n_peds_msg; i++) {
      const auto p = msg.objects.at(i);
      // These positions are in zed frame. Need to be converted into BRNE frame.
      // frame id is always zed_left_camera_frame
      const std::string ped_name = "ped_" + std::to_string(p.label_id);
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = msg.header.stamp;
      t.header.frame_id = "zed_left_camera_frame";
      t.child_frame_id = ped_name;
      t.transform.translation.x = p.position.at(0);
      t.transform.translation.y = p.position.at(1);
      t.transform.translation.z = p.position.at(2);
      // Send the transformation
      tf_broadcaster_->sendTransform(t);
      // immediately read this tf to get the transform. Seems a little janky but works for now.
      geometry_msgs::msg::TransformStamped T_brneodom_ped;
      try {
        T_brneodom_ped = tf_buffer_->lookupTransform(
          "brne_odom", ped_name,
          tf2::TimePointZero);
        crowd_nav_interfaces::msg::Pedestrian ped;
        ped.header.stamp = msg.header.stamp;
        ped.id = p.label_id;
        ped.pose.position.x = T_brneodom_ped.transform.translation.x;
        ped.pose.position.y = T_brneodom_ped.transform.translation.y;
        ped.pose.position.z = T_brneodom_ped.transform.translation.z;
        pa.pedestrians.push_back(ped);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO_STREAM(get_logger(), "Could not transform");
      }
    }

    n_peds = pa.pedestrians.size();
    ped_array = arma::mat(n_peds, 2, arma::fill::zeros);
    arma::mat for_compare(n_prev_peds, 2, arma::fill::zeros);
    for (int i=0; i<n_peds; i++){
      // do the frame to frame velocity transformation
      const auto ped = pa.pedestrians.at(i);
      ped_array.at(i, 0) = ped.pose.position.x;
      ped_array.at(i, 1) = ped.pose.position.y;
      arma::rowvec f2f_vel(2, arma::fill::zeros);
      if (n_prev_peds > 0){
        const arma::vec xs(n_prev_peds, arma::fill::value(ped.pose.position.x));
        const arma::vec ys(n_prev_peds, arma::fill::value(ped.pose.position.y));
        for_compare.col(0) = xs;
        for_compare.col(1) = ys;
        arma::mat difference = prev_ped_array - for_compare;
        auto norm = arma::vecnorm(difference, 2, 1);
        const auto prev_association_index = norm.index_min();
        f2f_vel = ped_array.row(i) - prev_ped_array.row(prev_association_index);
        const auto dt_ped = (ped_stamp - prev_ped_stamp).seconds();
        f2f_vel /= dt_ped;
        // truncate velocity if too large
        const auto abs_velocity = arma::norm(f2f_vel);
        if (abs_velocity > vel_cutoff){
          f2f_vel *= vel_cutoff/abs_velocity;
        }
      }
      // RCLCPP_INFO_STREAM(get_logger(), "Vel " << f2f_vel.at(0) << " " << f2f_vel.at(1));
      pa.pedestrians.at(i).velocity.linear.x = f2f_vel.at(0);
      pa.pedestrians.at(i).velocity.linear.y = f2f_vel.at(1);
    }
    pedestrian_pub_->publish(pa);

    // update previous values
    prev_ped_stamp = ped_stamp;
    prev_ped_array = ped_array;
    n_prev_peds = n_peds;
  }

  void timer_callback()
  {
    geometry_msgs::msg::TransformStamped T_bodom_brne;
    try {
      T_bodom_brne = tf_buffer_->lookupTransform(
        "brne_odom", "brne",
        tf2::TimePointZero);
      brne_odom.header.stamp = this->get_clock()->now();
      brne_odom.pose.pose.position.x = T_bodom_brne.transform.translation.x;
      brne_odom.pose.pose.position.y = T_bodom_brne.transform.translation.y;
      brne_odom.pose.pose.position.z = T_bodom_brne.transform.translation.z;
      brne_odom.pose.pose.orientation.w = T_bodom_brne.transform.rotation.w;
      brne_odom.pose.pose.orientation.x = T_bodom_brne.transform.rotation.x;
      brne_odom.pose.pose.orientation.y = T_bodom_brne.transform.rotation.y;
      brne_odom.pose.pose.orientation.z = T_bodom_brne.transform.rotation.z;
      brne_odom_pub_->publish(brne_odom);
    } catch (const tf2::TransformException & ex) {
      // RCLCPP_INFO_STREAM(get_logger(), "Could not transform brne_odom -> brne");
      return;
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ConvertPeds>());
  rclcpp::shutdown();
  return 0;
}
