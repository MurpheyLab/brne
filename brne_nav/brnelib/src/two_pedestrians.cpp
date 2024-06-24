#include "brnelib/brne.hpp"

int main(){
  int n_steps =         20;
  int n_samples =       50;

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
  brne_test.print_params();

  // nominal trajectories
  arma::mat x_nominal(n_agents, n_steps, arma::fill::zeros);
  arma::mat y_nominal(n_agents, n_steps, arma::fill::zeros);

  // agent 0: x from -2 to 0, y at -0.1
  x_nominal.row(0) = arma::linspace<arma::rowvec>(-2, 0, n_steps);
  y_nominal.row(0) = arma::rowvec(n_steps, arma::fill::value(-0.1));
  // agent 1: from 0 to -2, y = -0.3
  x_nominal.row(1) = arma::linspace<arma::rowvec>(0, -2, n_steps);
  y_nominal.row(1) = arma::rowvec(n_steps, arma::fill::value(-0.3));

  std::cout << "X nominal\n" << x_nominal << std::endl;
  std::cout << "Y nominal\n" << y_nominal << std::endl;

  // random sampling
  auto x0_sample = brne_test.mvn_sample_normal();
  auto x1_sample = brne_test.mvn_sample_normal();
  auto y0_sample = brne_test.mvn_sample_normal();
  auto y1_sample = brne_test.mvn_sample_normal();

  std::cout << "sample x0 \n" << x0_sample << std::endl;
  // std::cout << "sample x1\n" << x1_sample << std::endl;
  // std::cout << "sample y0\n" << y0_sample << std::endl;
  // std::cout << "sample y1\n" << y1_sample << std::endl;

  // put the samples into one matrix
  arma::mat x_samples(n_agents*n_samples, n_steps, arma::fill::zeros);
  x_samples.submat(0, 0, n_samples-1, n_steps-1) = x0_sample;
  x_samples.submat(n_samples, 0, n_agents*n_samples-1, n_steps-1) = x1_sample;
  // std::cout << "X samples \n" << x_samples << std::endl;

  arma::mat y_samples(n_agents*n_samples, n_steps, arma::fill::zeros);
  y_samples.submat(0, 0, n_samples-1, n_steps-1) = y0_sample;
  y_samples.submat(n_samples, 0, n_agents*n_samples-1, n_steps-1) = y1_sample;
  // std::cout << "Y samples \n" << y_samples << std::endl;

  auto width_scale = 1.0/(y_samples.max() - y_samples.min());
  // std::cout << "width scale" << width_scale << std::endl;
  x_samples *= width_scale;
  y_samples *= width_scale;

  // save some stuff to txt file
  x_nominal.save("x_nominal.csv", arma::csv_ascii);
  y_nominal.save("y_nominal.csv", arma::csv_ascii);

  x_samples.save("x_samples.csv", arma::csv_ascii);
  y_samples.save("y_samples.csv", arma::csv_ascii);

  arma::mat xtraj_samples(n_agents*n_samples, n_steps, arma::fill::zeros);
  arma::mat ytraj_samples(n_agents*n_samples, n_steps, arma::fill::zeros);

  // setting submatrix
  // xtraj_samples.submat(n_samples, 0, n_agents*n_samples-1, n_steps-1) = x1_sample;
  for (auto i=0; i<n_samples; i++){
    xtraj_samples.row(i) = x_nominal.row(0) + x_samples.row(i);
    ytraj_samples.row(i) = y_nominal.row(0) + y_samples.row(i);
  }

  auto ped_sample_scale = 1.0;
  for (auto a=1; a<n_agents; a++){
    for (auto i=0; i<n_samples; i++){
      xtraj_samples.row(a*n_samples + i) = x_nominal.row(a) + x_samples.row(a*n_samples + i) * ped_sample_scale;
      ytraj_samples.row(a*n_samples + i) = y_nominal.row(a) + y_samples.row(a*n_samples + i) * ped_sample_scale;
    }
  }

  std::cout << "Xtraj samples \n" << xtraj_samples << std::endl;
  std::cout << "Ytraj samples \n" << ytraj_samples << std::endl;

  auto weights = brne_test.brne_nav(xtraj_samples, ytraj_samples);

  auto trajs = brne_test.compute_optimal_trajectory(x_nominal, y_nominal, x_samples, y_samples);
  for (int i = 0; i<n_agents; i++){
    std::cout << "Agent" << i << std::endl;
    auto t = trajs.at(i);
    std::cout << "X traj" << std::endl;
    for (double i:t.x){
      std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << "Y traj" << std::endl;
    for (double i:t.y){
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }

  std::cout << brne_test.param_string() << std::endl;


  arma::arma_version ver;
  std::cout << "ARMA version: "<< ver.as_string() << std::endl;


  double max_lin_vel = 0.6;
  double max_ang_vel = 1.0;
  double nominal_lin_vel = 0.4;
  double nominal_ang_vel = 0.1;

  arma::rowvec robot_state(std::vector<double>{0.0, 0.0, 0.0});

  arma::rowvec goal(std::vector<double>{6.0, 0.0});

  brne::TrajGen tg{max_lin_vel, max_ang_vel, n_samples, n_steps, dt};
  auto traj = tg.traj_sample(nominal_lin_vel, nominal_ang_vel, robot_state);

  auto opt_cmds = tg.opt_controls(goal);

  std::cout << "OPt cmds\n" << opt_cmds << std::endl;

  auto opt_traj = tg.sim_traj(robot_state, opt_cmds);


  auto mvnsamp = brne_test.mvn_sample_normal(3);

  std::cout << "MVN SAMP\n" << mvnsamp << std::endl;

  return 0;
}