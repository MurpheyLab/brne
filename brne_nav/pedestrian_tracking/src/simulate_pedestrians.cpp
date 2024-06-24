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
#include "std_srvs/srv/empty.hpp"

using namespace std::chrono_literals;

class SimulatePedestrians : public rclcpp::Node
{
public:
  SimulatePedestrians()
  : Node("simulate_pedestrians"), go{false}
  {
    declare_parameter("rate", 15.0);
    declare_parameter("n_peds", 1);
    declare_parameter("moving", false);
    declare_parameter("ped_start_x", 5.0); // m
    declare_parameter("ped_end_x", 0.0);   // m
    declare_parameter("ped_y", 0.05);       // m
    declare_parameter("ped_vel", -1.0);    // m/s

    declare_parameter("static_peds/x", std::vector<double>{1.0});
    static_ped_x = get_parameter("static_peds/x").as_double_array();
    declare_parameter("static_peds/y", std::vector<double>{0.0});
    static_ped_y = get_parameter("static_peds/y").as_double_array();

    rate_hz = get_parameter("rate").as_double();
    std::chrono::milliseconds rate = (std::chrono::milliseconds) ((int)(1000. / rate_hz));

    n_peds = get_parameter("n_peds").as_int();

    moving = get_parameter("moving").as_bool();

    pedestrian_pub_ =
      create_publisher<crowd_nav_interfaces::msg::PedestrianArray>("pedestrians", 10);

    timer_ = create_wall_timer(rate, std::bind(&SimulatePedestrians::timer_callback, this));

    // simulate a pedestrian moving at -1 m/s in x direction
    ped_start = get_parameter("ped_start_x").as_double();
    ped_end = get_parameter("ped_end_x").as_double();
    ped_rate = get_parameter("ped_vel").as_double(); 
    ped_y = get_parameter("ped_y").as_double(); 
    // time to go this far
    const auto ped_time = (ped_end-ped_start) / ped_rate; // s
    // divide this up based on rate hz to figure out how many steps needed
    n_steps = rate_hz * ped_time;
    ped_x = ped_start;

    move_srv_ = create_service<std_srvs::srv::Empty>(
      "move_ped",
      std::bind(&SimulatePedestrians::move_cb, this, std::placeholders::_1, std::placeholders::_2));
    reset_srv_ = create_service<std_srvs::srv::Empty>(
      "reset_ped",
      std::bind(&SimulatePedestrians::reset_cb, this, std::placeholders::_1, std::placeholders::_2));
  }

private:
  double rate_hz;
  int n_peds;
  bool moving;
  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp::Publisher<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr pedestrian_pub_;

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr move_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;

  double ped_start;
  double ped_end;
  double ped_x;
  double ped_y;
  double ped_rate;
  int idx = 0;
  int n_steps;
  bool go;

  std::vector<double> static_ped_x, static_ped_y;

  void move_cb(
    std::shared_ptr<std_srvs::srv::Empty::Request>,
    std::shared_ptr<std_srvs::srv::Empty::Response>)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Start Moving pedestrian");
    go = true;
  }

  void reset_cb(
    std::shared_ptr<std_srvs::srv::Empty::Request>,
    std::shared_ptr<std_srvs::srv::Empty::Response>)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Reset Pedestrian");
    idx = 0;
    ped_x = ped_start;
  }

  void timer_callback()
  {
    crowd_nav_interfaces::msg::PedestrianArray peds;
    const auto current_time = this->get_clock()->now();

    if (moving){

      crowd_nav_interfaces::msg::Pedestrian moving_ped;
      moving_ped.header.stamp = current_time;
      moving_ped.id = 1;
      moving_ped.pose.position.x = ped_x;
      moving_ped.pose.position.y = ped_y;
      moving_ped.velocity.linear.x = 0.0;
      moving_ped.velocity.linear.y = 0.0;

      if (go){
        if (idx < n_steps){
          ped_x += (ped_end - ped_start) / n_steps;
          idx ++;
          moving_ped.velocity.linear.x = ped_rate;
        } else {
          go = false;
        }
      }

      peds.pedestrians.push_back(moving_ped);

    } else {

      const auto max_peds = static_cast<int>(static_ped_x.size());

      for (int ped_counter = 0; ((ped_counter < n_peds) && (ped_counter < max_peds)); ped_counter++) {
        crowd_nav_interfaces::msg::Pedestrian pedi;
        pedi.header.stamp = current_time;
        pedi.id = ped_counter + 1;
        pedi.pose.position.x = static_ped_x.at(ped_counter);
        pedi.pose.position.y = static_ped_y.at(ped_counter);
        peds.pedestrians.push_back(pedi);
      }

    }

    pedestrian_pub_->publish(peds);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimulatePedestrians>());
  rclcpp::shutdown();
  return 0;
}
