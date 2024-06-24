/// Dead reckoning estimation
/// Subscriptions
///   - cmd_vel
/// Publishes
///   - odom
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

using namespace std::chrono_literals;

class SimRobot : public rclcpp::Node
{
public:
  SimRobot()
  : Node("sim_robot")
  {
    auto pd = rcl_interfaces::msg::ParameterDescriptor{};

    pd.description = "Timer rate (Hz)";
    declare_parameter("rate", 50., pd);

    rate_hz = get_parameter("rate").as_double();
    RCLCPP_INFO_STREAM(get_logger(), "Rate is " << ((int)(1000. / rate_hz)) << "ms");
    std::chrono::milliseconds rate = (std::chrono::milliseconds) ((int)(1000. / rate_hz));

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10, std::bind(&SimRobot::cmd_vel_cb, this, std::placeholders::_1));

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odom", 10);

    brne_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("brne_odom", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    timer_ = create_wall_timer(
      rate, std::bind(&SimRobot::timer_callback, this));

    odom.header.frame_id = "odom";
    odom.child_frame_id = "base_link";

    brne_odom.header.frame_id = "brne_odom";
    brne_odom.child_frame_id = "brne";

    first_time = true;
  }

private:
  double rate_hz;
  nav_msgs::msg::Odometry odom;
  nav_msgs::msg::Odometry brne_odom;
  geometry_msgs::msg::TransformStamped t;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr brne_odom_pub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Time last_time;
  rclcpp::Time current_time;
  bool first_time;
  double theta = 0;
  const double pi = 3.14159;

  void cmd_vel_cb(const geometry_msgs::msg::Twist & msg)
  {
    current_time = this->get_clock()->now();
    if (!first_time) {
      const auto dt = (current_time - last_time).nanoseconds();
      const auto vx = msg.linear.x;
      // const auto vy = msg.linear.y;
      const auto vw = msg.angular.z;
      // assume diff drive so no vy
      theta += vw * dt * 1e-9;
      // normalize theta
      if (theta > pi) {
        theta -= 2 * pi;
      } else if (theta <= -pi) {
        theta += 2 * pi;
      }
      tf2::Quaternion q;
      q.setRPY(0, 0, theta);
      // update odom message
      odom.header.stamp = current_time;
      odom.pose.pose.position.x += (vx * cos(theta)) * dt * 1e-9;
      odom.pose.pose.position.y += (vx * sin(theta)) * dt * 1e-9;
      odom.pose.pose.orientation.x = q.x();
      odom.pose.pose.orientation.y = q.y();
      odom.pose.pose.orientation.z = q.z();
      odom.pose.pose.orientation.w = q.w();
      // update brne_odom message
      brne_odom.header.stamp = current_time;
      brne_odom.pose.pose.position.x += (vx * cos(theta)) * dt * 1e-9;
      brne_odom.pose.pose.position.y += (vx * sin(theta)) * dt * 1e-9;
      brne_odom.pose.pose.orientation.x = q.x();
      brne_odom.pose.pose.orientation.y = q.y();
      brne_odom.pose.pose.orientation.z = q.z();
      brne_odom.pose.pose.orientation.w = q.w();
      // update brne_odom -> brne tf
      t.header.stamp = current_time;
      t.header.frame_id = "brne_odom";
      t.child_frame_id = "brne";
      t.transform.translation.x += (vx * cos(theta)) * dt * 1e-9;
      t.transform.translation.y += (vx * sin(theta)) * dt * 1e-9;
      t.transform.rotation.x = q.x();
      t.transform.rotation.y = q.y();
      t.transform.rotation.z = q.z();
      t.transform.rotation.w = q.w();
      tf_broadcaster_->sendTransform(t);
    } else {
      first_time = false;
    }
    last_time = current_time;
  }

  void timer_callback()
  {
    // odom_pub_->publish(odom);
    brne_odom_pub_->publish(brne_odom);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimRobot>());
  rclcpp::shutdown();
  return 0;
}
