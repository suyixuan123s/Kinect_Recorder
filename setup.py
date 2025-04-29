import os
from setuptools import setup

def get_requirements():
    requirement_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirement_path, 'r') as f:
        requirements = f.readlines()
        return requirements

setup(
    name='rack-pipe-detector',
    version='20240711',
    description='AprilTag-based detector for racks and pipes',
    author='Yixuan Su',
    autohr_email='suyixuan19981215@gmail.com',
    packages=['rack_pipe_detector'],
    install_requires=get_requirements()
)