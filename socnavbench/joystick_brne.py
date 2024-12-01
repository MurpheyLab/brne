from typing import List, Optional

import numpy as np
from agents.agent import Agent
from dotmap import DotMap
from objectives.objective_function import ObjectiveFunction
from objectives.personal_space_cost import PersonalSpaceCost
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_agent_params
from socnav.socnav_renderer import SocNavRenderer
from trajectory.trajectory import SystemConfig, Trajectory
from utils.utils import euclidean_dist2

from joystick_py.joystick_base import JoystickBase

from . import brne


class JoystickBRNE(JoystickBase):
    def __init__(self):
        # planner variables
        # the list of commands sent to the robot to execute
        self.commands: List[str] = []
        self.simulator_joystick_update_ratio: int = 1
        # our 'positions' are modeled as (x, y, theta)
        self.robot_current: np.ndarray = None  # current position of the robot
        self.robot_v: float = 0  # not tracked in the base simulator
        self.robot_w: float = 0  # not tracked in the base simulator
        super().__init__("BRNE")  # parent class needs to know the algorithm

        print('use system dynamics: ', self.joystick_params.use_system_dynamics)
        assert not self.joystick_params.use_system_dynamics

        self.x_list: np.ndarray = None
        self.y_list: np.ndarray = None
        self.th_list: np.ndarray = None
        self.v_list: np.ndarray = None
    
    def from_conf(self, configs, idx):
        x = float(configs._position_nk2[0][idx][0])
        y = float(configs._position_nk2[0][idx][1])
        th = float(configs._heading_nk1[0][idx][0])
        v = float(configs._speed_nk1[0][idx][0])
        return (x, y, th, v)

    def init_obstacle_map(self, renderer: Optional[SocNavRenderer] = 0) -> SBPDMap:
        """ Initializes the sbpd map."""
        p: DotMap = self.agent_params.obstacle_map_params
        env = self.current_ep.get_environment()
        return p.obstacle_map(
            p,
            renderer,
            res=float(env["map_scale"]) * 100.0,
            map_trav=np.array(env["map_traversible"]),
        )

    def init_control_pipeline(self) -> None:
        # NOTE: this is like an init() run *after* obtaining episode metadata
        # robot start and goal to satisfy the old Agent.planner
        self.start_config: SystemConfig = SystemConfig.from_pos3(self.get_robot_start())
        self.goal_config: SystemConfig = SystemConfig.from_pos3(self.get_robot_goal())
        # rest of the 'Agent' params used for the joystick planner
        self.agent_params: DotMap = create_agent_params(
            with_planner=True, with_obstacle_map=True
        )
        # update generic 'Agent params' with joystick-specific params
        self.agent_params.episode_horizon_s = self.joystick_params.episode_horizon_s
        self.agent_params.control_horizon_s = self.joystick_params.control_horizon_s
        # init obstacle map
        self.obstacle_map: SBPDMap = self.init_obstacle_map()
        self.obj_fn: ObjectiveFunction = Agent._init_obj_fn(
            self, params=self.agent_params
        )
        psc_obj = PersonalSpaceCost(params=self.agent_params.personal_space_objective)
        self.obj_fn.add_objective(psc_obj)

        # Initialize Fast-Marching-Method map for agent's pathfinding
        Agent._init_fmm_map(self, params=self.agent_params)

        # Initialize system dynamics and planner fields
        self.planner = Agent._init_planner(self, params=self.agent_params)
        self.vehicle_data = self.planner.empty_data_dict()
        self.system_dynamics = Agent._init_system_dynamics(
            self, params=self.agent_params
        )
        # init robot current config from the starting position
        self.robot_current = self.current_ep.get_robot_start().copy()
        # init a list of commands that will be sent to the robot
        self.commands = None

        #################################################
        self.tsteps = 50
        self.num_peds = 5
        self.num_pts = 300
        self.num_steps = 10

        self.robot = self.get_robot_start()
        self.agents = {}
        agents_info = self.current_ep.get_agents()
        for key in list(agents_info.keys()):
            agent = agents_info[key]
            self.agents[key] = np.squeeze(
                agent.get_current_config().position_and_heading_nk3()
            )
        
        # sim_tlist = np.arange(self.tsteps) * self.sim_dt
        

    def joystick_sense(self):
        # ping's the robot to request a sim state
        self.send_to_robot("sense")

        # store previous pos3 of the robot (x, y, theta)
        robot_prev = self.robot_current.copy()  # copy since its just a list
        # listen to the robot's reply
        self.joystick_on = self.listen_once()

        # NOTE: at this point, self.sim_state_now is updated with the
        # most up-to-date simulation information

        # Update robot current position
        robot = list(self.sim_state_now.get_robots().values())[0]
        self.robot_current = robot.get_current_config().position_and_heading_nk3(
            squeeze=True
        )

        # Updating robot speeds (linear and angular) based off simulator data
        self.robot_v = euclidean_dist2(self.robot_current, robot_prev) / self.sim_dt
        self.robot_w = (self.robot_current[2] - robot_prev[2]) / self.sim_dt

        #################################
        robot_prev = self.robot.copy()
        agents_prev = {}
        for key in list(self.agents.keys()):
            agent = self.agents[key]
            agents_prev[key] = agent.copy()

        self.agents = {}
        self.agents_radius = {}
        agents_info = self.sim_state_now.get_all_agents()
        for key in list(agents_info.keys()):
            agent = agents_info[key]
            self.agents[key] = np.squeeze(
                agent.get_current_config().position_and_heading_nk3()
            )
            self.agents_radius[key] = agent.get_radius()
        robot_tmp = list(self.sim_state_now.get_robots().values())[0]
        self.robot = np.squeeze(
            robot_tmp.get_current_config().position_and_heading_nk3()
        )
        self.robot_radius = robot_tmp.get_radius()

        # self.robot_v = (self.robot - robot_prev) / self.sim_dt
        self.agents_v = {}
        for key in list(self.agents.keys()):
            if key in agents_prev:
                v = (self.agents[key] - agents_prev[key]) / self.sim_dt / 10
            else:
                v = np.array([0, 0, 0], dtype=np.float32)
            self.agents_v[key] = v

    def joystick_plan(self):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data
        - Access to sim_states from the self.current_world
        """
        # get information about robot by its "current position" which was updated in sense()
        [x, y, th] = self.robot_current
        v = self.robot_v
        # can also try:
        #     # assumes the robot has executed all the previous commands in self.commands
        #     (x, y, th, v) = self.from_conf(self.commands, -1)
        robot_config = SystemConfig.from_pos3(pos3=(x, y, th), v=0.3)
        self.planner_data = self.planner.optimize(
            robot_config, self.goal_config, sim_state_hist=self.sim_states
        )
        goal = self.get_robot_goal()
        dist2goal = np.sqrt((x-goal[0])**2 + (y-goal[1])**2)

        # TODO: make sure the planning control horizon is greater than the
        # simulator_joystick_update_ratio else it will not plan far enough

        tsteps = self.tsteps
        tlist = np.arange(tsteps) * self.sim_dt
        v_nominal = np.array([goal[0]-x, goal[1]-y])
        v_nominal = v_nominal / np.sqrt(v_nominal[0]**2+v_nominal[1]**2)
        v_nominal *= 1.0
        inc_list = tlist * v_nominal[:,np.newaxis]
        x_list = x + inc_list[0]
        y_list = y + inc_list[1]

        dist2obst = []
        for xt, yt in zip(x_list[:self.num_steps], y_list[:self.num_steps]):
            dist2obst.append(self.obstacle_map.dist_to_nearest_obs(np.array([[[xt,yt]]])))
        # print('dist to obst: ', np.min(dist2obst))
        min_dist2obst = np.min(dist2obst)

        meta_flag = False
        if dist2goal > 1.0 and min_dist2obst < 0.3:
        # if True:
            meta_flag = True
            self.commands = Trajectory.new_traj_clip_along_time_axis(
                self.planner_data["trajectory"],
                # self.agent_params.control_horizon,
                20,
                repeat_second_to_last_speed=True,
            )
            x_list = np.array(self.commands._position_nk2[0][:,0])
            y_list = np.array(self.commands._position_nk2[0][:,1])
            # print('verify x_list: ', len(x_list), end='  ')
            tsteps = 20
        else:
            tsteps = self.tsteps
        
        tlist = np.arange(tsteps) * self.sim_dt
        train_ts = np.array([tlist[0]])
        train_noise = np.array([1e-02])
        test_ts = tlist
        self.cov_Lmat, cov_mat = brne.get_Lmat_nb(train_ts, test_ts, train_noise)
        # print('cov diag: ', np.diagonal(cov_mat)[:10], end='  ')

        agent_dist_list = np.zeros(len(self.agents))
        for i, key in enumerate(list(self.agents.keys())):
            agent_dist_list[i] = np.sqrt((x-self.agents[key][0])**2 + (y-self.agents[key][1])**2)
        ped_keys = [list(self.agents.keys())[_i] for _i in np.argsort(agent_dist_list)[:self.num_peds]]
        num_brne_agents = len(ped_keys) + 1

        xmean_list = np.zeros((num_brne_agents, tsteps))
        ymean_list = np.zeros((num_brne_agents, tsteps))
        xmean_list[0] = x_list.copy()
        ymean_list[0] = y_list.copy()
        for i, key in enumerate(ped_keys):
            ped_v = np.array(self.agents_v[key][:2])
            # print('ped_v: ', ped_v, end='  ')
            # ped_v /= np.sqrt(ped_v[0]**2 + ped_v[1]**2)
            # ped_v *= 0.1
            xmean_list[i+1] = self.agents[key][0] + (tlist) * ped_v[0]
            ymean_list[i+1] = self.agents[key][1] + (tlist) * ped_v[1]
        
        x_opt_trajs = xmean_list.copy()
        y_opt_trajs = ymean_list.copy()

        if meta_flag == False:
        # if True:
            # if np.min(agent_dist_list) < 1.0:
            x_pts = brne.mvn_sample_normal(num_brne_agents * self.num_pts, tsteps, self.cov_Lmat)
            y_pts = brne.mvn_sample_normal(num_brne_agents * self.num_pts, tsteps, self.cov_Lmat)
            x_opt_trajs, y_opt_trajs, weights = brne.brne_nav(
                xmean_list, ymean_list, x_pts, y_pts,
                num_brne_agents, tsteps, self.num_pts
            )

            x_list = x_opt_trajs[0].copy()
            y_list = y_opt_trajs[0].copy()
            # print('weights: ', weights[0][::10], end='  ')
        else:
            x_list = x_opt_trajs[0].copy()
            y_list = y_opt_trajs[0].copy()

        v_list = np.sqrt((y_list[1:]-y_list[:-1])**2 + (x_list[1:]-x_list[:-1])**2)
        v_list = np.array([v, *v_list])
        th_list = np.arctan2(y_list[1:]-y_list[:-1], x_list[1:]-x_list[:-1])
        th_list = np.array([th, *th_list])

        self.x_list = x_list.copy()
        self.y_list = y_list.copy()
        self.th_list = th_list.copy()
        self.v_list = v_list.copy()

        # print('control_horizon: ', self.agent_params.control_horizon, end='  ')

    def joystick_act(self):
        if self.joystick_on:
            num_cmds_per_step = self.simulator_joystick_update_ratio
            # runs through the entire planned horizon just with a cmds_step of the above
            # num_steps = int(np.floor(self.commands.k / num_cmds_per_step))
            # num_steps = int(np.floor(self.agent_params.control_horizon / num_cmds_per_step))
            num_steps = self.num_steps
            for j in range(num_steps):
                xytv_cmds = []
                for i in range(num_cmds_per_step):
                    idx = j * num_cmds_per_step + i
                    # (x, y, th, v) = self.from_conf(self.commands, idx)
                    (x, y, th, v) = self.x_list[idx], self.y_list[idx], self.th_list[idx], self.v_list[idx]
                    xytv_cmds.append((x, y, th, v))
                self.send_cmds(xytv_cmds, send_vel_cmds=False)

                # break if the robot finished
                if not self.joystick_on:
                    break
            
            # print('idx: ', idx, end='  ')

    def update_loop(self):
        super().pre_update()  # pre-update initialization
        self.simulator_joystick_update_ratio = int(
            np.floor(self.sim_dt / self.agent_params.dt)
        )
        while self.joystick_on:
            # gather information about the world state based off the simulator
            self.joystick_sense()
            # create a plan for the next steps of the trajectory
            self.joystick_plan()
            # send a command to the robot
            self.joystick_act()
        # complete this episode, move on to the next if need be
        self.finish_episode()
