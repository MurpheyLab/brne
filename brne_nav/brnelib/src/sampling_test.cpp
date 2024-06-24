#include "brnelib/brne.hpp"

int main(){
  int n_steps =         25;
  int n_samples =       196;

  double kernel_a1 =    0.5;
  double kernel_a2 =    0.2;
  double cost_a1 =      8.0;
  double cost_a2 =      5.0;
  double cost_a3 =      20.0;
  double dt =           0.1;
  double y_min =        -0.5;
  double y_max =        0.5;

  int n_agents = 2;

  brne::BRNE brne_test{kernel_a1, kernel_a2,
                       cost_a1, cost_a2, cost_a3,
                       dt, n_steps, n_samples,
                       y_min, y_max};

  std::cout << brne_test.param_string() << std::endl;
  // brne_test.print_params();

  double max_lin_vel = 0.6;
  double max_ang_vel = 1.0;

  double nominal_lin_vel = 0.4;

  brne::TrajGen trajgen{max_lin_vel, max_ang_vel, n_samples, n_steps, dt};

  // Robot goes forward. starts at (0,0,0)
  arma::rowvec robot_pose(3, arma::fill::zeros);
  auto traj_samples = trajgen.traj_sample(nominal_lin_vel, 0.0, robot_pose);

  arma::mat xtraj_samples(n_agents*n_samples, n_steps, arma::fill::zeros);
  arma::mat ytraj_samples(n_agents*n_samples, n_steps, arma::fill::zeros);

  const auto robot_xtraj_samples = trajgen.get_xtraj_samples();
  const auto robot_ytraj_samples = trajgen.get_ytraj_samples();

  robot_xtraj_samples.save("robot_xtraj_samples.csv", arma::csv_ascii);
  robot_ytraj_samples.save("robot_ytraj_samples.csv", arma::csv_ascii);

  xtraj_samples.submat(0, 0, n_samples-1, n_steps-1) = robot_xtraj_samples;
  ytraj_samples.submat(0, 0, n_samples-1, n_steps-1) = robot_ytraj_samples;

  double ped_x = 2.0;
  double ped_y = 0.0;
  const auto ped_vx = -0.5;
  const auto ped_vy = 0.0;

  // nominal trajectories
  arma::mat x_nominal(n_agents, n_steps, arma::fill::zeros);
  arma::mat y_nominal(n_agents, n_steps, arma::fill::zeros);

  const auto x_pts = brne_test.mvn_sample_normal(n_agents - 1);
  const auto y_pts = brne_test.mvn_sample_normal(n_agents - 1);

  arma::vec ped_vel(std::vector<double>{ped_vx, ped_vy});
  auto speed_factor = arma::norm(ped_vel);
  arma::rowvec ped_xmean = arma::rowvec(n_steps, arma::fill::value(ped_x)) +
    arma::linspace<arma::rowvec>(0, (n_steps - 1), n_steps) * dt * ped_vx;
  arma::rowvec ped_ymean = arma::rowvec(n_steps, arma::fill::value(ped_y)) +
    arma::linspace<arma::rowvec>(0, (n_steps - 1), n_steps) * dt * ped_vy;
  arma::mat ped_xmean_mat(n_samples, n_steps, arma::fill::zeros);
  arma::mat ped_ymean_mat(n_samples, n_steps, arma::fill::zeros);
  ped_xmean_mat.each_row() = ped_xmean;
  ped_ymean_mat.each_row() = ped_ymean;
  const auto ped_xtraj_samples = arma::conv_to<arma::mat>::from(x_pts.submat(0, 0, n_samples - 1, n_steps - 1) * speed_factor + ped_xmean_mat);
  const auto ped_ytraj_samples = arma::conv_to<arma::mat>::from(y_pts.submat(0, 0, n_samples - 1, n_steps - 1) * speed_factor + ped_ymean_mat);
  ped_xtraj_samples.save("ped_xtraj_samples.csv", arma::csv_ascii);
  ped_ytraj_samples.save("ped_ytraj_samples.csv", arma::csv_ascii);

  xtraj_samples.submat(n_samples, 0, 2 * n_samples - 1, n_steps - 1) = ped_xtraj_samples;
  ytraj_samples.submat(n_samples, 0, 2 * n_samples - 1, n_steps - 1) = ped_ytraj_samples;

  auto weights = brne_test.brne_nav(xtraj_samples, ytraj_samples);

  const auto ulist = trajgen.get_ulist();
  const auto ulist_lin = arma::conv_to<arma::rowvec>::from(ulist.col(0));
  const auto ulist_ang = arma::conv_to<arma::rowvec>::from(ulist.col(1));
  const auto opt_cmds_lin = arma::mean(ulist_lin % weights.row(0));
  const auto opt_cmds_ang = arma::mean(ulist_ang % weights.row(0));

  arma::mat opt_cmds(n_steps, 2, arma::fill::zeros);
  opt_cmds.col(0) = arma::vec(n_steps, arma::fill::value(opt_cmds_lin));
  opt_cmds.col(1) = arma::vec(n_steps, arma::fill::value(opt_cmds_ang));

  const auto robot_opt_traj = trajgen.sim_traj(robot_pose, opt_cmds);
  robot_opt_traj.save("robot_optimal_traj.csv", arma::csv_ascii);

  // get the optimal path of the pedestrian
  const auto agent_weights = arma::conv_to<arma::vec>::from(weights.row(1));
  auto agent_x_samples = arma::conv_to<arma::mat>::from(x_pts.submat(0, 0, n_samples-1, n_steps-1));
  auto agent_y_samples = arma::conv_to<arma::mat>::from(y_pts.submat(0, 0, n_samples-1, n_steps-1));
  for (int i=0; i<n_samples; i++){
    agent_x_samples.row(i) *= agent_weights.at(i);
    agent_y_samples.row(i) *= agent_weights.at(i);
  }
  const auto ped_xtraj = ped_xmean + arma::mean(agent_x_samples,0);
  const auto ped_ytraj = ped_ymean + arma::mean(agent_y_samples,0);
  arma::mat ped_opt_traj(2, n_steps);
  ped_opt_traj.row(0) = ped_xtraj;
  ped_opt_traj.row(1) = ped_ytraj;
  ped_opt_traj = ped_opt_traj.t();
  ped_opt_traj.save("ped_optimal_traj.csv", arma::csv_ascii);

  return 0;
}