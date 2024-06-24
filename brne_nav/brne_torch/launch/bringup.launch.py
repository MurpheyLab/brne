from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(Node(
        package='brne_torch',
        executable='brne_nav_torch',
        output='screen',
    ))

    return ld
