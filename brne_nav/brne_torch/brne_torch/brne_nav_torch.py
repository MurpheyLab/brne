#############################################
import numpy as np
from . import brne_torch as brne
import time

import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseArray
from nav_msgs.msg import Odometry, Path
from crowd_nav_interfaces.msg import Pedestrian, PedestrianArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int16, String
from action_msgs.msg import GoalStatus
from tf_transformations import euler_from_quaternion
from rclpy.time import Time
from rclpy.duration import Duration
import yaml


def pose2d_transform(msg):
    roll, pitch, yaw = euler_from_quaternion(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    return np.array([msg.position.x, msg.position.y, yaw])


# THIS IS THE NAVIGATION CLASS !!!
class BrneNavRos(Node):
    def __init__(self):
        super().__init__('brne_nav_torch')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.ped_info_pub = self.create_publisher(PoseArray, '/brne_peds', 1)
        self.opt_traj_pub = self.create_publisher(Path, '/optimal_path', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/brne_markers', 1)
        self.wall_pub = self.create_publisher(MarkerArray, '/walls', 1)
        self.num_peds_pub = self.create_publisher(Int16, '/brne/n_pedestrians', 1)
        self.result_pub = self.create_publisher(GoalStatus, '/navigation_result', 1)
        self.path_distr_pub = self.create_publisher(MarkerArray, "/brne_ped_path_distr", 1)

        latching_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.params_pub = self.create_publisher(String, '/brne/params', qos_profile=latching_qos)

        ####################################################################################
        # ALL PARAMETERS ARE DEFINED HERE!!!
        ####################################################################################
        self.declare_parameter('maximum_agents', 5)  # maximum number of agents BRNE will consider (including the robot)
        self.declare_parameter('num_samples', 196)  # number of samples assigned to each agent
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('plan_steps', 20)  # time steps of the planning horizon
        self.declare_parameter('max_lin_vel', 0.5)  # 0.6,0.8 maximum linear velocity allowed on the robot
        self.declare_parameter('max_ang_vel', 1.0)  # 1.0,1.2 maximum angular velocity allowed on the robot
        self.declare_parameter('nominal_vel', 0.4)  #0.4,0.5 nomimal (linear) velocity when plannig the initial trajectory
        self.declare_parameter('kernel_a1', 0.2)  # control the "straightness" of trajectory samples. Larger the value is, less straight the trajectory sample will be.
        self.declare_parameter('kernel_a2', 0.2)  # control the "width/spreadness" of trajectory samples. Larger the value is, more spread the trajectory samples are.
        self.declare_parameter('cost_a1', 4.0)  # control the safety zone, smaller the value is, more conversative the robot will be.
        self.declare_parameter('cost_a2', 1.0)  # control the safety zone, larger the value is, more conservative the robot will be.
        self.declare_parameter('cost_a3', 70.0)  #  control the safety penalty weight, larger the value is, more conservative the robot will be.
        self.declare_parameter('ped_sample_scale', 0.1)  # pedestrian's willingness for cooperation, default value is 1.0, the smaller it is, the less the robot would expect the pedestrians to make space for it
        self.declare_parameter('ad', -5.0)  # "aggressiveness" of the optimal controller, the more negative ad is, the more aggressive the robot will be on chasing the way points (might leads to jerky or aggressive behavior)
        self.declare_parameter('R_lin', 1.0)  # control penalty weight on linear velocity, the larger it is, the smoother the linear motion will be, but way point tracking accuracy might be compromised
        self.declare_parameter('R_ang', 2.0)  # control penalty weight on angular velocity, the larger it is, the smoother the rotation motion will be, but way point tracking accuracy might be compromised
        self.declare_parameter('replan_freq', 5.0)  # unit: Hz
        self.declare_parameter('people_timeout', 0.5)  # unit: seconds
        self.declare_parameter('corridor_y_min', -1.0)  # lower bound of y coordinate (one side of corridor)
        self.declare_parameter('corridor_y_max', 1.0)  # upper bound of y coordinate (the other side of corridor)
        self.declare_parameter('staircase_truncation', False)  # saturate F2F velocity in a staircase manner
        self.declare_parameter('people_timeout_off', True)
        self.declare_parameter('close_stop_threshold', 0.4)  # threshold for safety mask, leading to estop
        self.declare_parameter('open_space_velocity', 0.5)  # nominal velocity when the robot is in open space
        self.declare_parameter('brne_activate_threshold', 9.5)  # distance threshold from a pedestrian to enable BRNE
        self.declare_parameter('waypoint_dist', 4.0)  # distance for the next waypoint
        self.declare_parameter('speed_factor', 0.3)  # post-hoc factor for pedestrian path uncertainty
        ####################################################################################
        self.num_agents = self.get_parameter('maximum_agents').value
        self.num_samples = self.get_parameter('num_samples').value
        self.dt = self.get_parameter('dt').value
        self.plan_steps = self.get_parameter('plan_steps').value
        self.max_lin_vel = self.get_parameter('max_lin_vel').value
        self.max_ang_vel = self.get_parameter('max_ang_vel').value
        self.nominal_vel = self.get_parameter('nominal_vel').value
        self.kernel_a1 = self.get_parameter('kernel_a1').value
        self.kernel_a2 = self.get_parameter('kernel_a2').value
        self.cost_a1 = self.get_parameter('cost_a1').value
        self.cost_a2 = self.get_parameter('cost_a2').value
        self.cost_a3 = self.get_parameter('cost_a3').value
        self.ped_sample_scale = self.get_parameter('ped_sample_scale').value
        self.replan_freq = self.get_parameter('replan_freq').value
        self.people_timeout = Duration(seconds=self.get_parameter('people_timeout').value)
        self.corridor_y_min = self.get_parameter('corridor_y_min').value
        self.corridor_y_max = self.get_parameter('corridor_y_max').value
        self.staircase_truncation = self.get_parameter('staircase_truncation').value
        self.people_timeout_off = self.get_parameter('people_timeout_off').value
        self.close_stop_threshold = self.get_parameter('close_stop_threshold').value
        self.open_space_velocity = self.get_parameter('open_space_velocity').value
        self.brne_activate_threshold = self.get_parameter('brne_activate_threshold').value
        self.waypoint_dist = self.get_parameter('waypoint_dist').value
        self.speed_factor = self.get_parameter('speed_factor').value

        self.get_logger().debug(f'plan_steps: {self.plan_steps}')
        self.get_logger().debug(f'corridor_y_min: {self.corridor_y_min}')
        self.get_logger().debug(f'corridor_y_max: {self.corridor_y_max}')

        self.params_msg = String()
        param_dict = {}
        for param in self._parameters:
            param_dict[param] = self.get_parameter(param).value
        self.params_msg.data = yaml.safe_dump(param_dict)

        self.num_peds = 0  # number of current pedestrians that BRNE chooses to interact with

        parallel_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.ped_sub = self.create_subscription(PedestrianArray,
                                                '/pedestrians', self.ped_cb, 1,
                                                callback_group=parallel_cb_group)
        self.odom_sub = self.create_subscription(Odometry, "/brne_odom", self.odom_cb, 1,
                                                 callback_group=parallel_cb_group)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_cb, 1,
                                                 callback_group=parallel_cb_group)

        # this callback function publish control command at fixed frequency
        self.ped_msg_buffer = {}
        self.timer = self.create_timer(self.dt, self.timer_cb,
                                       callback_group=parallel_cb_group)
        self.brne_timer = self.create_timer(1.0/self.replan_freq, self.brne_cb,
                                            callback_group=parallel_cb_group)

        self.robot_pose = np.zeros(3)  # the robot's initial pose
        self.robot_goal = None   # the robot's goal
        self.robot_traj = []

        self.cmd_tracker = TrajTracker(dt=self.dt, max_lin_vel=self.max_lin_vel, max_ang_vel=self.max_ang_vel)   # the class that converts way points to control commands
        self.cmds = np.zeros((self.plan_steps, 2))   # the control command buffer
        self.cmd_counter = 0   # counter that goes with the control command buffer

        # the trajectory from the optimal control commands
        self.cmds_traj = np.tile(self.robot_pose, (self.plan_steps, 1))

        self.x_opt_trajs = np.zeros((self.num_agents, self.plan_steps))  # optimal trajectories from BRNE
        self.y_opt_trajs = np.zeros((self.num_agents, self.plan_steps))

        # initialize the BRNE covariance matrix here
        tlist = np.arange(self.plan_steps) * self.dt
        train_ts = np.array([tlist[0]])
        train_noise = np.array([1e-04])
        test_ts = tlist

        self.cov_Lmat, self.cov_mat = brne.get_Lmat(train_ts, test_ts, train_noise, self.kernel_a1, self.kernel_a2)

        ### F2F velocity estimation
        self.curr_ped_array = []
        self.prev_ped_array = []

        self.close_stop_flag = False

        self.brne_first_time = True

        # initialize agent trajectory samples
        self.xtraj_samples = np.zeros((
            self.num_agents * self.num_samples, self.plan_steps
        ))
        self.ytraj_samples = np.zeros((
            self.num_agents * self.num_samples, self.plan_steps
        ))
        for i in range(1, self.num_agents):
            self.xtraj_samples[i*self.num_samples : (i+1)*self.num_samples] = (i+1) * 1000.0
            self.ytraj_samples[i*self.num_samples : (i+1)*self.num_samples] = (i+1) * 1000.0

        self.publish_walls()

    def publish_walls(self):
        now = self.get_clock().now().to_msg()
        height = 1.0
        length = 10.0
        thickness = 0.01
        transparency = 0.2

        ma = MarkerArray()
        for i in range(2):
            wall = Marker()
            wall.header.frame_id = "brne_odom"
            wall.header.stamp = now
            wall.id = i
            wall.type = 1 # cube
            wall.action = 0
            wall.pose.position.x = 0.5*length - 1.0
            wall.pose.position.y = self.corridor_y_min
            wall.pose.position.z = 0.5*height
            wall.color.a = transparency
            wall.color.b = 1.0
            wall.scale.x = length
            wall.scale.y = thickness
            wall.scale.z = height
            ma.markers.append(wall)
        ma.markers[0].pose.position.y = self.corridor_y_min
        ma.markers[1].pose.position.y = self.corridor_y_max
        self.wall_pub.publish(ma)

    def brne_cb(self):
        brne_cb_tic = time.time()
        start_time = self.get_clock().now()

        ped_info_list = []
        dists2peds = []
        now = self.get_clock().now()
        if self.people_timeout_off == False:
            for ped_ident, (ped, stamp) in list(self.ped_msg_buffer.items()):
                if now - stamp > self.people_timeout:
                    del self.ped_msg_buffer[ped_ident]

        # we go through each perceived pedestrian and save the information
        for ped, stamp in self.ped_msg_buffer.values():
            dist2ped = np.sqrt((self.robot_pose[0]-ped.pose.position.x)**2 + (self.robot_pose[1]-ped.pose.position.y)**2)
            if dist2ped < self.brne_activate_threshold:  # only consider pedestrians within the activate threshold
                ped_info = np.array([
                    ped.pose.position.x, ped.pose.position.y, ped.velocity.linear.x, ped.velocity.linear.y
                ])
                ped_info_list.append(ped_info)

                dists2peds.append(dist2ped)

        ped_info_list = np.array(ped_info_list)
        self.num_peds = len(ped_info_list)

        dists2peds = np.array(dists2peds)

        # compute how many pedestrians we are actually interacting with
        num_agents = np.minimum(self.num_peds+1, self.num_agents)

        # Publish num agents
        num_peds_msg = Int16()
        num_peds_msg.data = int(num_agents-1)
        self.num_peds_pub.publish(num_peds_msg)

        # self.get_logger().debug(f'total # pedestrians: {self.num_peds}.')
        # self.get_logger().debug(f'brne # agents: {num_agents}.')

        # self.num_peds = 0
        if num_agents > 1:
            ped_indices = np.argsort(dists2peds)[:num_agents-1]  # we only pick the N closest pedestrian to interact with
            robot_state = self.robot_pose.copy()

            x_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)
            y_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)

            # self.get_logger().info(f'X and y pts shape {x_pts.shape} {y_pts.shape}')
            # ctrl space configuration here
            # xtraj_samples = np.zeros((
            #     self.num_agents * self.num_samples, self.plan_steps
            # ))
            # ytraj_samples = np.zeros((
            #     self.num_agents * self.num_samples, self.plan_steps
            # ))
            # for i in range(1, self.num_agents):
            #     xtraj_samples[i*self.num_samples : (i+1)*self.num_samples] = (i+1) * 1000.0
            #     ytraj_samples[i*self.num_samples : (i+1)*self.num_samples] = (i+1) * 1000.0


            closest_dist2ped = 100.0
            closest_ped_pos = np.zeros(2) + 100.0
            for i, ped_id in enumerate(ped_indices):
                ped_pos = ped_info_list[ped_id][:2]
                ped_vel = ped_info_list[ped_id][2:]
                # speed_factor = np.linalg.norm(ped_vel)
                # speed_factor = 0.2
                ped_xmean = ped_pos[0] + np.arange(self.plan_steps) * self.dt * ped_vel[0]
                ped_ymean = ped_pos[1] + np.arange(self.plan_steps) * self.dt * ped_vel[1]

                # self.get_logger().info(f'shape into xtraj samples {(xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples]).shape}')

                self.xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    x_pts[i*self.num_samples : (i+1)*self.num_samples] * self.speed_factor + ped_xmean
                self.ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    y_pts[i*self.num_samples : (i+1)*self.num_samples] * self.speed_factor + ped_ymean

                dist2ped = np.linalg.norm([
                    robot_state[:2] - ped_pos[:2]
                ])
                if dist2ped < closest_dist2ped:
                    closest_dist2ped = dist2ped
                    closest_ped_pos = ped_pos.copy()

            num_vis_samples = 10
            vis_samples_idx = (np.arange(num_vis_samples) + self.num_samples * np.arange(1, self.num_agents)[:,None]).ravel().astype(int)
            xtraj_samples_vis = self.xtraj_samples[vis_samples_idx, :]
            ytraj_samples_vis = self.ytraj_samples[vis_samples_idx, :]
            self.publish_path_distr(self.path_distr_pub, xtraj_samples_vis, ytraj_samples_vis)

            st = robot_state.copy()

            if self.robot_goal is None:
                goal = np.array([6.0, 0.0])
            else:
                state2goal = self.robot_goal[:2] - self.robot_pose[:2]
                dist2goal = np.linalg.norm(state2goal)
                state2goal /= dist2goal
                if dist2goal > self.waypoint_dist:
                    goal = self.robot_pose[:2] + self.waypoint_dist * state2goal
                else:
                    goal = self.robot_goal[:2]
                self.get_logger().info('goal: {}'.format(goal))

            if st[2] > 0.0:
                theta_a = st[2] - np.pi/2
            else:
                theta_a = st[2] + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])
            vec2goal = goal - st[:2]
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            radius = 0.5 * dist2goal / proj_len

            if st[2] > 0.0:
                ut = np.array([self.nominal_vel, -self.nominal_vel/radius])
            else:
                ut = np.array([self.nominal_vel, self.nominal_vel/radius])
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1))
            # self.get_logger().info(f"Nominal commands {nominal_cmds.shape}\n{nominal_cmds}")
            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, self.max_lin_vel, self.max_ang_vel, self.num_samples
            )
            # self.get_logger().info(f"ulist {ulist_essemble.shape}\n{ulist_essemble}")
            tiles = np.tile(robot_state, reps=(self.num_samples,1)).T
            # self.get_logger().info(f"Tiles {tiles.shape}\n{tiles}")
            traj_essemble = brne.traj_sim_essemble(
                np.tile(robot_state, reps=(self.num_samples,1)).T,
                ulist_essemble,
                self.dt
            )
            # self.get_logger().info(f"traj {traj_essemble.shape}\n{traj_essemble}")
            # self.get_logger().info(f"for xtraj samples {(traj_essemble[:,0,:].T).shape}\n{traj_essemble[:,0,:].T}")
            self.xtraj_samples[0:self.num_samples] = traj_essemble[:,0,:].T
            self.ytraj_samples[0:self.num_samples] = traj_essemble[:,1,:].T

            # generate sample weight mask for the closest pedestrian
            robot_xtrajs = traj_essemble[:,0,:].T
            robot_ytrajs = traj_essemble[:,1,:].T
            robot_samples2ped = (robot_xtrajs - closest_ped_pos[0])**2 + (robot_ytrajs - closest_ped_pos[1])**2
            robot_samples2ped = np.min(np.sqrt(robot_samples2ped), axis=1)
            safety_mask = (robot_samples2ped > self.close_stop_threshold).astype(float)
            # self.get_logger().info(f'safety mask\n{safety_mask}')
            safety_samples_percent = safety_mask.mean() * 100
            # self.get_logger().debug('percent of safe samples: {:.2f}%'.format(safety_samples_percent))
            # self.get_logger().debug('dist 2 ped: {:.2f} m'.format(closest_dist2ped))


            self.close_stop_flag = False
            if np.max(safety_mask) == 0.0:
                safety_mask = np.ones_like(safety_mask)
                self.close_stop_flag = True
            # self.get_logger().debug('safety mask: {}'.format(safety_mask))

            # BRNE OPTIMIZATION HERE !!!
            weights = brne.brne_nav(
                self.xtraj_samples, self.ytraj_samples,
                self.num_agents, self.plan_steps, self.num_samples,
                self.cost_a1, self.cost_a2, self.cost_a3, self.ped_sample_scale,
                self.corridor_y_min, self.corridor_y_max
            )

            # self.get_logger().info(f"Weights\n{weights}")

            if self.brne_first_time:
                self.get_logger().info("BRNE initialization complete!")
                self.brne_first_time = False

            if weights is None:
                self.get_logger().info("We are going out of bounds. Stop going to this goal")
                self.robot_goal = None
                g = GoalStatus()
                g.status = GoalStatus.STATUS_SUCCEEDED
                self.result_pub.publish(g)
                return

            # apply safety mask
            weights[0] *= safety_mask
            if (np.mean(weights[0]) != 0):
                weights[0] /= np.mean(weights[0])
            else:
                self.get_logger().info("Stopping because of safety mask")

            # generate optimal ctrl cmds and update buffer
            opt_cmds_1 = np.mean(ulist_essemble[:,:,0] * weights[0], axis=1)
            opt_cmds_2 = np.mean(ulist_essemble[:,:,1] * weights[0], axis=1)
            # self.get_logger().info(f"opt cmds 1 {opt_cmds_1}")
            self.cmds = np.array([opt_cmds_1, opt_cmds_2]).T
            self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)

            ped_trajs_x = np.zeros((num_agents-1, self.plan_steps))
            ped_trajs_y = np.zeros((num_agents-1, self.plan_steps))
            for i in range(num_agents-1):
                ped_trajs_x[i] = \
                    np.mean(self.xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
                ped_trajs_y[i] = \
                    np.mean(self.ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)

            self.publish_trajectory(self.opt_traj_pub, self.cmds_traj[:,0], self.cmds_traj[:,1])
            # self.publish_markers(ped_trajs_x, ped_trajs_y)

            if self.robot_goal is None or self.close_stop_flag == True:
                self.cmds = np.zeros((self.plan_steps, 2))
                self.cmds_traj = np.tile(robot_state, reps=(self.plan_steps,1))

            # for smoothness, we allow the robot to execute the first 5 time steps from the buffer
            if self.cmd_counter > 0:
                self.cmd_counter = 0

        else:  # if no pedestrian around, go straight to the goal
            self.close_stop_flag = False
            robot_state = self.robot_pose.copy()
            st = robot_state.copy()

            if self.robot_goal is None:
                goal = np.array([6.0, 0.0])
            else:
                state2goal = self.robot_goal[:2] - self.robot_pose[:2]
                dist2goal = np.linalg.norm(state2goal)
                state2goal /= dist2goal
                if dist2goal > self.waypoint_dist:
                    goal = self.robot_pose[:2] + self.waypoint_dist * state2goal
                else:
                    goal = self.robot_goal[:2]
                self.get_logger().info('goal: {}'.format(goal))

            if st[2] > 0.0:
                theta_a = st[2] - np.pi/2
            else:
                theta_a = st[2] + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])
            vec2goal = goal - st[:2]
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            radius = 0.5 * dist2goal / proj_len

            nominal_vel = self.open_space_velocity
            if st[2] > 0.0:
                ut = np.array([nominal_vel, -nominal_vel/radius])
            else:
                ut = np.array([nominal_vel,  nominal_vel/radius])
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1))

            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, nominal_vel+0.05, self.max_ang_vel, self.num_samples
            )
            traj_essemble = brne.traj_sim_essemble(
                np.tile(robot_state, reps=(self.num_samples,1)).T,
                ulist_essemble,
                self.dt
            )
            end_pose_essemble = traj_essemble[-1, 0:2, :].T
            dists2goal_essemble = np.linalg.norm(end_pose_essemble - goal, axis=1)
            opt_cmds = ulist_essemble[:, np.argmin(dists2goal_essemble), :]

            self.cmds = opt_cmds
            self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)

            if self.cmd_counter > 0:
                self.cmd_counter = 0

            self.publish_trajectory(self.opt_traj_pub, self.cmds_traj[:,0], self.cmds_traj[:,1])
            # self.publish_markers([], [])

            if self.robot_goal is None:
                self.cmds = np.zeros((self.plan_steps, 2))
                self.cmds_traj = np.tile(robot_state, reps=(self.plan_steps,1))

        end_time = self.get_clock().now()
        diff = (end_time - start_time).to_msg()
        diff_sec = diff.sec + diff.nanosec*1e-9
        self.get_logger().debug(f"Agents: {num_agents} Timer (torch): {diff_sec}")

        brne_cb_toc = time.time()
        brne_cb_elapsed_time = brne_cb_toc - brne_cb_tic
        self.get_logger().info('brne_cb elapsed time: {:.4f} s'.format(brne_cb_elapsed_time))

    def publish_trajectory(self, publisher, xs, ys):
        p = Path()
        p.header.frame_id = 'brne_odom'

        for x, y in zip(xs, ys):
            pose = PoseStamped()
            pose.header = p.header
            pose.pose.position.x = x
            pose.pose.position.y = y

            p.poses.append(pose)

        publisher.publish(p)

    def publish_path_distr(self, publisher, xtrajs, ytrajs):
        marker_array = MarkerArray()
        traj_count = 0

        for xtraj, ytraj in zip(xtrajs, ytrajs):
            marker = Marker()
            marker.id = traj_count
            marker.header.frame_id = "brne_odom"
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color = ColorRGBA(r=1.0, g=0.67, b=0.0, a=0.5)
            marker.pose.orientation.w = 1.0
            for x, y in zip(xtraj, ytraj):
                # print('x,y: ', float(x), float(y))
                p = Point(x=float(x), y=float(y), z=0.0)
                marker.points.append(p)
            marker_array.markers.append(marker)
            traj_count += 1

        publisher.publish(marker_array)

    def publish_markers(self, xsa, ysa):
        ma = MarkerArray()
        for xs, ys in zip(xsa, ysa):
            m = Marker()
            m.header.frame_id = 'brne_odom'
            m.ns = 'ped_traj'
            m.id = len(ma.markers)
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.1
            m.color.a = 1.0
            m.color.r = 0.96
            m.color.g = 0.50
            m.color.b = 0.19

            for x, y in zip(xs, ys):
                p = Point()
                p.x = x
                p.y = y
                m.points.append(p)

            ma.markers.append(m)
        self.marker_pub.publish(ma)

    def staircase_velocity(self, vel):
        """
        "Truncate" velocity like a staircase.
        """
        speed = np.linalg.norm(vel)

        if speed < 0.3:
            factor = 0.0
        elif speed < 0.6:
            factor = 0.3
        else:
            factor = speed

        new_vel = vel / speed * factor
        return new_vel


    def ped_cb(self, msg):
        """
        This is the pedestrian perception callback function.
        Everytime it receives new pedestrian information, it does the BRNE optimization to compute the optimal way
        points and convert the way points to optimal control commands, these commands will update the control buffer
        """

        # there should be an initialization flag, but in practice it does not really matter
        self.prev_ped_array = [ped for ped in self.curr_ped_array]  #  self.curr_ped_array.copy()
        # self.curr_ped_array = np.zeros((num_peds, 2))
        self.curr_ped_array = []

        if self.people_timeout_off:
            self.ped_msg_buffer = {}

        # force existing pedestrians' velocity to be zero in the timeout buffer
        for key in self.ped_msg_buffer:
            self.ped_msg_buffer[key][0].velocity.x = 0.0
            self.ped_msg_buffer[key][0].velocity.y = 0.0

        # stamp = Time.from_msg(msg.header.stamp)
        for ped in msg.pedestrians:
            ped_pose = ped.pose.position
            if np.isnan(ped_pose.x) or np.isnan(ped_pose.y):
                self.get_logger().debug(f'Detect NAN on {ped.id} !!!')
                continue  # skip the pedestrian is reading is nan

            ### F2F implementation
            f2f_vel = np.zeros(2)
            num_prev_peds = len(self.prev_ped_array)
            ped_position = np.array([ped_pose.x, ped_pose.y])
            # self.curr_ped_array[i] = ped_position.copy()
            self.curr_ped_array.append(ped_position.copy())

            # self.get_logger().info(f'Curr ped array {self.curr_ped_array}')

            if num_prev_peds > 0:
                # self.get_logger().info(f'prev ped array\n{self.prev_ped_array}')
                # self.get_logger().info(f'ped position\n{ped_position}')
                # self.get_logger().info(f'difference\n{self.prev_ped_array - ped_position}')
                # dists2prev = np.linalg.norm(self.prev_ped_array - ped_position, axis=1)
                dists2prev = [np.linalg.norm(ped_position - prev_ped_position) for prev_ped_position in self.prev_ped_array]
                # self.get_logger().info(f'dists2prev\n{dists2prev}')
                # self.get_logger().info(f'argmin\n{np.argmin(dists2prev)}')
                f2f_vel = ped_position - self.prev_ped_array[np.argmin(dists2prev)]
                f2f_vel /= 0.034  # assuming pedestrian information is published at 33 hz
                # self.get_logger().info(f"F2f vel {f2f_vel}")
                if self.staircase_truncation:
                    f2f_vel = self.staircase_velocity(f2f_vel)

            ped.velocity.linear.x = f2f_vel[0]
            ped.velocity.linear.y = f2f_vel[1]

            stamp = Time.from_msg(ped.header.stamp)

            # self.ped_msg_buffer.append(new_ped_msg)
            self.ped_msg_buffer[ped.id] = ped, stamp

        # self.curr_ped_array = np.array(self.curr_ped_array)

    def goal_cb(self, msg):
        position = msg.pose.position
        self.robot_goal = np.array([position.x, position.y])
        self.get_logger().info(f'Robot Goal Received: {position.x}, {position.y}')
        self.params_pub.publish(self.params_msg)

        self.check_goal()

    def odom_cb(self, msg):
        # the odometry callback function updates the robot's current pose
        self.robot_pose = pose2d_transform(msg.pose.pose)
        # self.get_logger().debug(f'odom pose: {self.robot_pose}')
        if self.robot_goal is None:
            return

        self.check_goal()

    def check_goal(self):
        dist2goal = np.sqrt((self.robot_pose[0]-self.robot_goal[0])**2 + (self.robot_pose[1]-self.robot_goal[1])**2)
        # self.get_logger().debug(f'dist2goal: {dist2goal}')
        if dist2goal < 0.5:
            self.robot_goal = None
            self.get_logger().info('Close to goal! Stopping.')
            # Minor hack: Instead of implementing an action server, we just publish a GoalStatus for bookkeeping
            g = GoalStatus()
            g.status = GoalStatus.STATUS_SUCCEEDED
            self.result_pub.publish(g)

    def timer_cb(self):
        """
        This the control callback function, it receives a fixed frequency timer signal,
        and publish the control command at the same frequency (10Hz here because dt = 0.1)
        """
        # self.get_logger().debug(f'cmd counter: {self.cmd_counter}')
        cmd = Twist()

        # here we get the command from the buffer, using self.cmd_counter to track the index
        #if self.cmd_counter >= self.plan_steps-1:
        if self.cmd_counter >= self.plan_steps-1 or self.robot_goal is None:
            cmd.linear.x = float(0.0)
            cmd.angular.z = float(0.0)
        else:
            cmd.linear.x = float(self.cmds[self.cmd_counter][0])
            cmd.angular.z = float(self.cmds[self.cmd_counter][1])
            self.cmd_counter += 1

        # manually correct drift of Unitree
        linear_vel = np.abs(cmd.linear.x)
        if linear_vel >= 0.1 and linear_vel <= 0.6:
            cmd.angular.z -= 0.040

        # self.get_logger().debug(f'current control: [{cmd.linear.x}, {cmd.angular.z}]')
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = BrneNavRos()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
