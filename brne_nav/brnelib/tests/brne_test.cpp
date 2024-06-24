#include "brnelib/brne.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <string>
#include <sstream>


namespace brne {

    TEST_CASE("one_static_ped", "brne"){

        // arma::arma_rng::set_seed_random();

        int n_steps =         5;
        int n_samples =       16;

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

        double max_lin_vel = 0.6;
        double max_ang_vel = 1.0;

        double nominal_lin_vel = 0.4;

        brne::TrajGen trajgen{max_lin_vel, max_ang_vel, n_samples, n_steps, dt};

        // nominal trajectories
        arma::mat x_nominal(n_agents, n_steps, arma::fill::zeros);
        arma::mat y_nominal(n_agents, n_steps, arma::fill::zeros);

        // Robot goes forward. starts at (0,0,0)
        arma::rowvec robot_pose(3, arma::fill::zeros);
        auto traj_samples = trajgen.traj_sample(nominal_lin_vel, 0.0, robot_pose);

        arma::mat xtraj_samples(n_agents*n_samples, n_steps, arma::fill::zeros);
        arma::mat ytraj_samples(n_agents*n_samples, n_steps, arma::fill::zeros);

        auto robot_xtraj_samples = trajgen.get_xtraj_samples();
        auto robot_ytraj_samples = trajgen.get_ytraj_samples();

        // these are hard coded from a test case that I know works
        arma::mat desired_robot_xtraj_samples(n_samples, n_steps, arma::fill::zeros);
        desired_robot_xtraj_samples.row(0) =  arma::rowvec(std::vector<double>{0.0200,0.0397,0.0591,0.0779,0.0959});
        desired_robot_xtraj_samples.row(1) =  arma::rowvec(std::vector<double>{0.0257,0.0511,0.0760,0.1001,0.1233});
        desired_robot_xtraj_samples.row(2) =  arma::rowvec(std::vector<double>{0.0314,0.0624,0.0929,0.1224,0.1507});
        desired_robot_xtraj_samples.row(3) =  arma::rowvec(std::vector<double>{0.0371,0.0738,0.1098,0.1446,0.1781});
        desired_robot_xtraj_samples.row(4) =  arma::rowvec(std::vector<double>{0.0428,0.0851,0.1267,0.1669,0.2055});
        desired_robot_xtraj_samples.row(5) =  arma::rowvec(std::vector<double>{0.0485,0.0965,0.1435,0.1891,0.2329});
        desired_robot_xtraj_samples.row(6) =  arma::rowvec(std::vector<double>{0.0542,0.1078,0.1604,0.2114,0.2603});
        desired_robot_xtraj_samples.row(7) =  arma::rowvec(std::vector<double>{0.0599,0.1192,0.1773,0.2337,0.2877});
        desired_robot_xtraj_samples.row(8) =  arma::rowvec(std::vector<double>{0.0200,0.0397,0.0591,0.0779,0.0959});
        desired_robot_xtraj_samples.row(9) =  arma::rowvec(std::vector<double>{0.0257,0.0511,0.0760,0.1001,0.1233});
        desired_robot_xtraj_samples.row(10) = arma::rowvec(std::vector<double>{0.0314,0.0624,0.0929,0.1224,0.1507});
        desired_robot_xtraj_samples.row(11) = arma::rowvec(std::vector<double>{0.0371,0.0738,0.1098,0.1446,0.1781});
        desired_robot_xtraj_samples.row(12) = arma::rowvec(std::vector<double>{0.0428,0.0851,0.1267,0.1669,0.2055});
        desired_robot_xtraj_samples.row(13) = arma::rowvec(std::vector<double>{0.0485,0.0965,0.1435,0.1891,0.2329});
        desired_robot_xtraj_samples.row(14) = arma::rowvec(std::vector<double>{0.0542,0.1078,0.1604,0.2114,0.2603});
        desired_robot_xtraj_samples.row(15) = arma::rowvec(std::vector<double>{0.0599,0.1192,0.1773,0.2337,0.2877});

        arma::mat desired_robot_ytraj_samples(n_samples, n_steps, arma::fill::zeros);
        desired_robot_ytraj_samples.row(0) =  arma::rowvec(std::vector<double>{-0.0010,-0.0040,-0.0089,-0.0158,-0.0245});
        desired_robot_ytraj_samples.row(1) =  arma::rowvec(std::vector<double>{-0.0013,-0.0051,-0.0115,-0.0203,-0.0315});
        desired_robot_ytraj_samples.row(2) =  arma::rowvec(std::vector<double>{-0.0016,-0.0063,-0.0140,-0.0248,-0.0385});
        desired_robot_ytraj_samples.row(3) =  arma::rowvec(std::vector<double>{-0.0019,-0.0074,-0.0166,-0.0293,-0.0455});
        desired_robot_ytraj_samples.row(4) =  arma::rowvec(std::vector<double>{-0.0021,-0.0085,-0.0191,-0.0338,-0.0525});
        desired_robot_ytraj_samples.row(5) =  arma::rowvec(std::vector<double>{-0.0024,-0.0097,-0.0217,-0.0383,-0.0595});
        desired_robot_ytraj_samples.row(6) =  arma::rowvec(std::vector<double>{-0.0027,-0.0108,-0.0242,-0.0429,-0.0665});
        desired_robot_ytraj_samples.row(7) =  arma::rowvec(std::vector<double>{-0.0030,-0.0120,-0.0268,-0.0474,-0.0735});
        desired_robot_ytraj_samples.row(8) =  arma::rowvec(std::vector<double>{ 0.0010, 0.0040, 0.0089, 0.0158, 0.0245});
        desired_robot_ytraj_samples.row(9) =  arma::rowvec(std::vector<double>{ 0.0013, 0.0051, 0.0115, 0.0203, 0.0315});
        desired_robot_ytraj_samples.row(10) = arma::rowvec(std::vector<double>{ 0.0016, 0.0063, 0.0140, 0.0248, 0.0385});
        desired_robot_ytraj_samples.row(11) = arma::rowvec(std::vector<double>{ 0.0019, 0.0074, 0.0166, 0.0293, 0.0455});
        desired_robot_ytraj_samples.row(12) = arma::rowvec(std::vector<double>{ 0.0021, 0.0085, 0.0191, 0.0338, 0.0525});
        desired_robot_ytraj_samples.row(13) = arma::rowvec(std::vector<double>{ 0.0024, 0.0097, 0.0217, 0.0383, 0.0595});
        desired_robot_ytraj_samples.row(14) = arma::rowvec(std::vector<double>{ 0.0027, 0.0108, 0.0242, 0.0429, 0.0665});
        desired_robot_ytraj_samples.row(15) = arma::rowvec(std::vector<double>{ 0.0030, 0.0120, 0.0268, 0.0474, 0.0735});

        REQUIRE(arma::approx_equal(robot_xtraj_samples, desired_robot_xtraj_samples, "absdiff", 0.0001));
        REQUIRE(arma::approx_equal(robot_ytraj_samples, desired_robot_ytraj_samples, "absdiff", 0.0001));

        xtraj_samples.submat(0, 0, n_samples-1, n_steps-1) = robot_xtraj_samples;
        ytraj_samples.submat(0, 0, n_samples-1, n_steps-1) = robot_ytraj_samples;

        // =========================================================================================

        // pedestrian static at (1.0, 0.01)
        double ped_x = 1.0;
        double ped_y = 0.01;
        xtraj_samples.submat((1)*n_samples, 0, (2)*n_samples-1, n_steps-1) = arma::mat(n_samples, n_steps, arma::fill::value(ped_x));
        ytraj_samples.submat((1)*n_samples, 0, (2)*n_samples-1, n_steps-1) = arma::mat(n_samples, n_steps, arma::fill::value(ped_y));

        auto static_weights = brne_test.brne_nav(xtraj_samples, ytraj_samples);

        std::cout << "Static Weights\n" << static_weights << std::endl;

        REQUIRE(static_cast<int>(static_weights.n_rows) == n_agents);
        REQUIRE(static_cast<int>(static_weights.n_cols) == n_samples);

        arma::mat desired_static_weights(n_agents, n_samples, arma::fill::zeros);
        // hard-coded from a version of the algorithm that I know works
        desired_static_weights.row(0) = arma::rowvec(std::vector<double>{7.1966e+00,8.8948e-01,6.0468e-02,3.5967e-03,2.7752e-04,3.3587e-05,6.5357e-06,1.9297e-06,6.9580e+00,8.3311e-01,5.4973e-02,3.2140e-03,2.4731e-04,3.0161e-05,5.9462e-06,1.7818e-06});
        desired_static_weights.row(1) = arma::rowvec(n_samples, arma::fill::ones);

        std::cout << "Desired Static Weights\n" << desired_static_weights << std::endl;

        REQUIRE(arma::approx_equal(static_weights, desired_static_weights, "absdiff", 0.0001));

        // =========================================================================================

        // pedestrian moving at -0.5 m/s in x direction
        double ped_vx = -0.5;
        double ped_vy = 0.0;

        arma::rowvec ped_xmean = arma::rowvec(n_steps, arma::fill::value(ped_x)) +
          arma::linspace<arma::rowvec>(0, (n_steps - 1), n_steps) * dt * ped_vx;
        arma::rowvec ped_ymean = arma::rowvec(n_steps, arma::fill::value(ped_y)) +
          arma::linspace<arma::rowvec>(0, (n_steps - 1), n_steps) * dt * ped_vy;

        arma::mat ped_xmean_mat(n_samples, n_steps, arma::fill::zeros);
        arma::mat ped_ymean_mat(n_samples, n_steps, arma::fill::zeros);
        ped_xmean_mat.each_row() = ped_xmean;
        ped_ymean_mat.each_row() = ped_ymean;

        const auto speed_factor = 1.0;

        const auto ped_xsamples = arma::conv_to<arma::mat>::from(ped_xmean_mat + brne_test.mvn_sample_normal() * speed_factor);
        const auto ped_ysamples = arma::conv_to<arma::mat>::from(ped_ymean_mat + brne_test.mvn_sample_normal() * speed_factor);

        xtraj_samples.submat((1)*n_samples, 0, (2)*n_samples-1, n_steps-1) = ped_xsamples;
        ytraj_samples.submat((1)*n_samples, 0, (2)*n_samples-1, n_steps-1) = ped_ysamples;

        const auto moving_weights = brne_test.brne_nav(xtraj_samples, ytraj_samples);

        std::cout << "Moving Weights\n" << moving_weights << std::endl;

        REQUIRE(static_cast<int>(moving_weights.n_rows) == n_agents);
        REQUIRE(static_cast<int>(moving_weights.n_cols) == n_samples);

        arma::mat desired_moving_weights(n_agents, n_samples, arma::fill::zeros);
        // hard-coded from a version of the algorithm that I know works
        desired_moving_weights.row(0) = arma::rowvec(std::vector<double>{7.4404e+00,1.9497e+00,2.5198e-01,2.0412e-02,1.5153e-03,1.4078e-04,1.9137e-05,3.9156e-06,5.3604e+00,9.0309e-01,6.8430e-02,3.6351e-03,2.2290e-04,2.1225e-05,3.3945e-06,8.7086e-07});
        desired_moving_weights.row(1) = arma::rowvec(std::vector<double>{1.1278e-02,1.3576e-07,1.7904e-07,1.8944e-07,1.8020e-07,5.1752e-03,1.4078e-07,1.4274e-07,1.1432e-05,1.2219e-06,1.4527e-07,2.3051e-07,2.3235e-01,1.3523e-07,1.5751e+01,3.4930e-05});

        std::cout << "Desired Moving Weights\n" << desired_moving_weights << std::endl;

        REQUIRE(arma::approx_equal(moving_weights, desired_moving_weights, "absdiff", 0.001));
    }

}