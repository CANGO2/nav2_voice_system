from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'nav2_voice_system'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'),
         glob('nav2_voice_system/maps/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kth',
    description='Voice navigation system with LLM API',
    entry_points={
        'console_scripts': [
            # 노트북 B에서 실행
            'navigation_server = nav2_voice_system.node_b_server:main',
            # 노트북 A에서 실행
            'voice_client = nav2_voice_system.node_a_voice:main',
        ],
    },
)
