# Synchronization-of-a-multi-drone-system-with-reinforcement-learning-MARL

## Introduction

La synchronisation de drones pour former une flotte est un enjeu majeur dans plusieurs
domaines. On y trouve plusieurs applications civiles ou militaires. Notre projet s’inscrit dans
cette thématique en nous limitant à la synchronisation de deux drones.

## Contenu

Cette branch contient deux algorithme d'entrainement ainsi que la simulation qui va avec :

### Le PPO (proximal policy optimization) : 
L'entrainemnet par PPO peut etre lancé avec la commande (sans oublier de sourcer le workspace ros)
```bash
roslaunch bebop2_train bebop2_double_train.launch
```
### Le SAC (soft actor critic)
```bash
roslaunch bebop2_train SAC_train.launch
```

Le package contient également un teleop qui peremt de controler autant réel que simulé.
Vous pouvez lancer la simulation avec : 

```bash
roslaunch rotors_gazebo mav_2_bebop.launch
```
Pour lancer la simulation avec 2 drones
ou 

```bash
roslaunch rotors_gazebo mav_1_bebop.launch
```
pour un seul drones

## Références :

Openai_ros : http://wiki.ros.org/openai_ros

Iros drone : https://github.com/arnaldojr/iROS_drone/tree/noetic

Rotors_simulator : https://github.com/ethz-asl/rotors_simulator

SAC Algorithm  : https://spinningup.openai.com/en/latest/algorithms/sac.html#

PPO Algorithm : https://pylessons.com/BipedalWalker-v3-PPO
