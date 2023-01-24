import numpy as np
# from onpolicy.envs.mpe.core import World, Agent, Landmark # gazebo environment (??)
from onpolicy.envs.mpe.scenario import BaseScenario

class Agent(object):
    def __init__(self, pos):
        self.pos = pos #x ,y, z, Y
        self.action = None


class World(object):
    def __init__(self, n_agents, d, d_min):
        self.world_step = 0
        self.n_agents = n_agents
        self.d = d
        self.d_min = d_min
        self.agents = self.init_ag(self.n_agents, self.d)
        self.world_length = 25
        self.dim_p = 3
        # simulation timestep
        self.dt = 0.1


    def init_ag(self, n_agents, d):
        agents = []
        for i in range(n_agents):
            agents.append(Agent(np.array([0, 0 + i*d, 0, 0]).reshape(-1, 1)))
        return agents

    def step(self):
        self.world_step += 1
        for agent in self.agents:
            agent.pos



class Scenario(BaseScenario):
    def make_world(self, args, d=1.5, d_min=.5):
        world = World(args.num_agents, d, d_min)

        return world

    def reset_world(self, world):
        world.init_ag(world.n_agents, world.d)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2, d_min):
        dist = np.linalg.norm((agent1.pos[:-1]-agent2.pos[:-1]))
        return True if dist < d_min else False

    def reward(self, world):
        rew = world.world_step/world.world_length

        if self.is_collision(world.agents[0], world.agents[1], world.d_min):
            rew = -1000
        
        return rew

    def observation(self, world):
        return np.concatenate((world.agents[0].pos, world.agents[1].pos))

    def done(self, world):
        return is_collision(world.agents[0], world.agents[1], world.d_min)
