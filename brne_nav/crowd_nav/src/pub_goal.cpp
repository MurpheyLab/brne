#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_srvs/srv/empty.hpp"
#include "crowd_nav_interfaces/srv/goal_req.hpp"

using namespace std::chrono_literals;

class PubGoal : public rclcpp::Node
{
public:
  PubGoal()
  : Node("pub_goal")
  {
    goal_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("goal_pose", 10);

    set_goal_pose_srv_ = create_service<crowd_nav_interfaces::srv::GoalReq>(
      "set_goal_pose",
      std::bind(&PubGoal::set_goal_pose_cb, this, std::placeholders::_1, std::placeholders::_2));
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_pub_;
  rclcpp::Service<crowd_nav_interfaces::srv::GoalReq>::SharedPtr set_goal_pose_srv_;

  void set_goal_pose_cb(
    std::shared_ptr<crowd_nav_interfaces::srv::GoalReq::Request> req,
    std::shared_ptr<crowd_nav_interfaces::srv::GoalReq::Response>)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Set Goal Pose");
    geometry_msgs::msg::PoseStamped gp;
    gp.header.stamp = this->get_clock()->now();
    gp.header.frame_id = "odom";
    gp.pose.position.x = req->x;
    gp.pose.position.y = req->y;
    goal_pose_pub_->publish(gp);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PubGoal>());
  rclcpp::shutdown();
  return 0;
}
