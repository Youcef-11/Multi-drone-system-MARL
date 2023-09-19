# Synchronization of a Multi-Drone System with Reinforcement Learning (MARL)

## Introduction

Synchronizing drones to form a fleet is a critical challenge in various domains, including both civilian and military applications. Our project falls within this thematic scope, with a focus on the synchronization of two drones.

## Content

This branch contains two training algorithms and the associated simulation:

1. **Proximal Policy Optimization (PPO)**:
   - Launch PPO training using the following command (don't forget to source the ROS workspace):

     ```bash
     roslaunch bebop2_train bebop2_double_train.launch
     ```

2. **Soft Actor Critic (SAC)**:
   - Start SAC training with the following command:

     ```bash
     roslaunch bebop2_train SAC_train.launch
     ```

The package also includes a teleoperation module that allows control in both real and simulated environments. You can initiate the simulation with:

- For a simulation with 2 drones:

     ```bash
     roslaunch rotors_gazebo mav_2_bebop.launch
     ```

- For a simulation with a single drone:

     ```bash
     roslaunch rotors_gazebo mav_1_bebop.launch
     ```

## Installation

To run this project, you will need to have the following installed:

- [ROS (Robot Operating System)](http://wiki.ros.org/ROS/Installation)
- [OpenAI ROS](http://wiki.ros.org/openai_ros)
- [iROS Drone](https://github.com/arnaldojr/iROS_drone/tree/noetic)
- [Rotors Simulator](https://github.com/ethz-asl/rotors_simulator)

Ensure that you have properly configured your ROS environment and sourced the ROS workspace before running the provided commands.

## References

- [OpenAI ROS](http://wiki.ros.org/openai_ros)
- [iROS Drone](https://github.com/arnaldojr/iROS_drone/tree/noetic)
- [Rotors Simulator](https://github.com/ethz-asl/rotors_simulator)
- [SAC Algorithm](https://spinningup.openai.com/en/latest/algorithms/sac.html#)
- [PPO Algorithm](https://pylessons.com/BipedalWalker-v3-PPO)

Feel free to explore the provided algorithms, simulations, and references to further understand and contribute to the synchronization of multi-drone systems using reinforcement learning.


SAC Algorithm  : https://spinningup.openai.com/en/latest/algorithms/sac.html#

PPO Algorithm : https://pylessons.com/BipedalWalker-v3-PPO
