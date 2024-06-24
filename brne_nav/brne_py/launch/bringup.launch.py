from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(Node(
        package='brne_py',
        executable='brne_nav',
        output='screen',
    ))

    return ld
