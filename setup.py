import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'qcar2_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eduardex',
    maintainer_email='eduardex@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_motorcommands = qcar2_teleop.teleop_motorcommands:main',
            'color_segmentation_node = qcar2_teleop.color_segmentation_node:main',
            'yellow_line_position_node = qcar2_teleop.yellow_line_position_node:main',
            'yellow_line_follower_controller = qcar2_teleop.yellow_line_follower_controller:main',
            'pid_tuner_gui_node = qcar2_teleop.pid_tuner_gui_node:main',
            'controller_plotter_node = qcar2_teleop.controller_plotter_node:main',
        ],
    },
)
