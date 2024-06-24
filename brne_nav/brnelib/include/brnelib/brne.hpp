#include <iostream>
#include <cmath> 
#include <string>
#include <stdexcept>
#include <iosfwd>
#include <vector>
#include <armadillo>
#include <random>
#include <numeric>
#include <chrono>
#include <omp.h>

namespace brne
{
  /// @brief Describes a trajectory in 2D space.
  struct traj{
    /// @brief points forming the x trajectory
    std::vector<double> x;
    /// @brief points forming the y trajectory
    std::vector<double> y;
  };
  /// @brief Class for Bayes Rule Nash Equilibrium calculations
  class BRNE{
    private:
      double kernel_a1;
      double kernel_a2;
      double cost_a1;
      double cost_a2;
      double cost_a3;
      double dt;
      int n_steps;
      int n_samples;
      double y_min;
      double y_max;

      // these are constant for a single instance of this class
      arma::mat cov_Lmat;
      arma::mat cov_mat;      

      // these change depending on iteration
      int n_agents = 1;
      arma::mat index_table;
      arma::mat costs;
      arma::mat all_pts;
      arma::mat weights;
      arma::mat coll_mask;

      /// @brief Compute the kernel matrix
      /// @param t1 vector 1
      /// @param t2 vector 2
      /// @return kernel matrix
      arma::mat compute_kernel_mat(arma::vec t1, arma::vec t2);
      /// @brief compute the index table, as a function of the number of agents
      void compute_index_table();
      /// @brief Compute the cost matrix of a set of trajectories
      /// @param xtraj x trajectory samples
      /// @param ytraj y trajectory samples
      void compute_costs(arma::mat xtraj, arma::mat ytraj);
      /// @brief perform a collision check with the walls of the corridor
      /// @param ytraj y trajectory samples
      void collision_check(arma::mat ytraj);
      /// @brief Perform one iteration of bayesian updates on the weights matrix
      void update_weights();

    public:
      /// @brief empty constructor. Not actually useful, but might need for ros node.
      explicit BRNE();
      /// @brief Construct an instance of the BRNE class
      /// @param kernel_a1 control the "straightness" of trajectory samples. 
      /// The larger the value is, the less straight the trajectory sample will be.
      /// @param kernel_a2 control the "width/spreadness" of trajectory samples. 
      /// The larger the value is, more spread the trajectory samples are.
      /// @param cost_a1 control the safety zone
      /// The smaller the value is, the more conversative the robot will be.
      /// @param cost_a2 control the safety zone
      /// The larger the value is, the more conservative the robot will be.
      /// @param cost_a3  control the safety penalty weight
      /// The larger the value is, more conservative the robot will be.
      /// @param dt time between ticks (seconds)
      /// @param n_steps number of timesteps in the planning horizon
      /// @param n_samples number of samples each agent generates
      /// @param y_min minimum coordinate of the corridor
      /// @param y_max maximum coordinate of the corridor
      explicit BRNE(double kernel_a1, double kernel_a2,
                    double cost_a1, double cost_a2, double cost_a3,
                    double dt, int n_steps, int n_samples,
                    double y_min, double y_max);
      /// @brief Print out the constants of the BRNE class to cout
      void print_params();
      /// @brief Return parameters as a string
      std::string param_string();
      /// @brief Compute the covariance matrix
      void compute_Lmat();
      /// @brief sample the multivariate normal distribution
      /// @return sample of size n_steps x n_samples
      arma::mat mvn_sample_normal();
      /// @brief sample the multivariate normal distribution
      /// @return sample of size n_steps x (n_samples*n_peds)
      arma::mat mvn_sample_normal(int n_peds);
      /// @brief Compute BRNE weights for optimal trajectory
      /// @param xtraj_samples x trajectory samples, size (n_agents*n_samples) x n_steps
      /// @param ytraj_samples y trajectory samples, size (n_agents*n_samples) x n_steps
      /// @return optimal weights of size n_agents x n_samples. If row a is filled with zeros
      /// (ie, if weights.row(a).is_zero()), then agent a has no valid path.
      arma::mat brne_nav(arma::mat xtraj_samples, arma::mat ytraj_samples);
      /// @brief Compute optimal trajectories from the BRNE weights
      /// @param x_nominal nominal x trajectory
      /// @param y_nominal nominal y trajectory
      /// @param x_samples samples of x trajectories
      /// @param y_samples samples of y trajectories
      /// @return list of optimal trajectories, length is the number of agents.
      std::vector<traj> compute_optimal_trajectory(arma::mat x_nominal, arma::mat y_nominal,
                                                   arma::mat x_samples, arma::mat y_samples);
  };

  /// @brief Class for generating commands for a diff drive
  class TrajGen{
    private:
      double max_lin_vel;
      double max_ang_vel;
      int n_samples;
      int n_steps;
      double dt;

      arma::mat ulist;

      arma::mat x_nominal;
      arma::mat y_nominal;

      arma::mat xtraj_samples;
      arma::mat ytraj_samples;

      arma::mat end_pose;

      arma::mat dyn(arma::mat state, arma::mat controls);
      arma::rowvec dyn(arma::rowvec state, arma::rowvec controls);

      arma::mat dyn_step(arma::mat state, arma::mat controls);
      arma::rowvec dyn_step(arma::rowvec state, arma::rowvec controls);
    public:
      /// @brief empty constructor. Not actually useful, but might need for ros node.
      explicit TrajGen();
      /// @brief Construct a Trajectory Generator object
      /// @param max_lin_vel maximum linear velocity in m/s
      /// @param max_ang_vel maximum angular velocity in rad/s
      /// @param n_samples number of samples in the trajectory
      /// @param n_steps number of time steps we are predicting forward
      /// @param dt time between each timestep tick (s)
      explicit TrajGen(double max_lin_vel, double max_ang_vel, int n_samples, int n_steps, double dt);

      /// @brief Create a list of controls perturbed from nominal controls
      /// @param lin_vel nominal linear velocity
      /// @param ang_vel nominal angular velocity
      /// @param state vector of [x,y,theta] of the robot's position
      std::vector<arma::mat> traj_sample(double lin_vel, double ang_vel, arma::rowvec state);
      
      /// @brief get the trajectory sample x coordinate for the robot
      /// @return matrix of nsamples * nsteps
      arma::mat get_xtraj_samples();

      /// @brief get the trajectory sample y coordinate for the robot
      /// @return matrix of nsamples * nsteps
      arma::mat get_ytraj_samples();

      /// @brief Get the controls for the robot
      /// @return matrix of controls, size nsamples*2 (col 0 is linear and col 1 is angular control)
      arma::mat get_ulist();

      /// @brief Get the optimal controls for a given goal location
      /// @param goal goal vector of [x,y]
      /// @return matrix of optimal controls of nsteps * 2
      arma::mat opt_controls(arma::rowvec goal);

      /// @brief simulate a trajectory starting from state with controls
      /// @param state starting position of [x,y,theta]
      /// @param controls matrix of size nsteps*2 where each row is [lin vel, ang vel]
      /// @return matrix of size nsteps*3 where each row is the [x,y,theta] at that timestep.
      arma::mat sim_traj(arma::rowvec state, arma::mat controls);
  };
}