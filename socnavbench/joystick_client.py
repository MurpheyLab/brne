import argparse

from params.central_params import create_joystick_params

from joystick_py.joystick_base import JoystickBase
from joystick_py.joystick_planner import JoystickWithPlanner, JoystickWithPlannerPosns
from joystick_py.joystick_random import JoystickRandom

from joystick_py.joystick_brne import JoystickBRNE
from joystick_py.joystick_cvm import JoystickCVM

from RVO2.joystick_RVO import JoystickRVO
from RVO2.joystick_RVO_with_checkpoints import JoystickRVOwCkpt
from social_force.joystick_social_force import JoystickSocialForce
from sacadrl.joystick_sacadrl import JoystickSACADRL
from sacadrl.joystick_sacadrl_with_checkpoints import JoystickSACADRLwCkpt


def run_joystick(J: JoystickBase) -> None:
    """run the joystick process"""
    # connect to the robot
    J.init_send_conn()
    J.init_recv_conn()
    # first listen() for the episode names
    assert J.get_all_episode_names()
    episodes = J.get_episodes()

    # we want to run on at least one episode
    assert len(episodes) > 0
    for ep_title in episodes:
        print("Waiting for episode: {}...".format(ep_title))

        # second listen() for the specific episode details
        J.get_episode_metadata()
        assert J.current_ep and J.current_ep.get_name() == ep_title

        J.init_control_pipeline()

        J.update_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joystick planning algorithm param")
    parser.add_argument(
        "--algo",
        type=str.lower,
        default="rvowckpt",
        choices=[
            "sampling",
            "random",
            "randomcpp",
            "rvo",
            "rvowckpt",
            "sacadrl",
            "sacadrlwckpt",
            "socialforce",
            "brne",
            "cvm"
        ],
        help="Choose the specific joystick algorithm to run in the simulation",
    )
    args = parser.parse_args()
    joystick_params = create_joystick_params()
    if args.algo.lower() == "sampling":
        if joystick_params.use_system_dynamics:
            # uses the joystick that sends velocity commands instead of positional
            J = JoystickWithPlanner()
        else:
            # uses the joystick that sends positional commands instead of velocity
            J = JoystickWithPlannerPosns()
    elif args.algo.lower() == "random":
        J = JoystickRandom()
    elif args.algo.lower() == "randomcpp":
        raise NotImplementedError  # run subprocess of cpp binary
    elif args.algo.lower() == "rvo":
        J = JoystickRVO()
    elif args.algo.lower() == "rvowckpt":
        J = JoystickRVOwCkpt()
    elif args.algo.lower() == "socialforce":
        J = JoystickSocialForce()
    elif args.algo.lower() == "sacadrl":
        J = JoystickSACADRL()
    elif args.algo.lower() == "sacadrlwckpt":
        J = JoystickSACADRLwCkpt()
    
    elif args.algo.lower() == "brne":
        J = JoystickBRNE()
    elif args.algo.lower() == "cvm":
        J = JoystickCVM()
    
    else:
        raise NotImplementedError

    run_joystick(J)
