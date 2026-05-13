from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_voice_system',
            executable='voice_client',
            name='nuc_node',
            output='screen',
        ),
    ])
