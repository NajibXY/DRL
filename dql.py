import argparse

import copy
import random

import torch

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt


class Interaction:

    def __init__(self,e,a,s,r,f):
        self.e = e
        self.a = a
        self.s = s
        self.r = r
        self.f = f


class Buffer:

    def __init__(self, size):
        self.size = size
        self.buffr = []

    def append(self, interaction):
        if len(self.buffr) >= self.size:
            self.buffr.pop(0)
        self.buffr.append(interaction)

    def sample(self, n):
        if n > len(self.buffr):
            n = len(self.buffr)
        return random.sample(self.buffr, n)

    def length(self):
        return len(self.buffr)
    
    
class QModel(torch.nn.Module):
    
    def __init__(self, e_size, a_size):
        super(QModel, self).__init__()
        self.couche1 = torch.nn.Linear(e_size, 32)
        self.couche2 = torch.nn.Linear(32, 32)
        self.couche3 = torch.nn.Linear(32, 32)
        self.couche4 = torch.nn.Linear(32, a_size)

    def forward(self, i):
        o = self.couche1(i)
        o = torch.relu(o)
        o = self.couche2(o)
        o = torch.relu(o)
        o = self.couche3(o)
        o = torch.relu(o)
        o = self.couche4(o)

        return o

class DQN_Agent(object):
    
    def __init__(self, env, buffr_size):
        self.gamma = 0.95
        self.freq_copy = 1000
        self.tau = 1
        self.tau_decay = 0.999
        self.min_tau = 0.2
        self.sigma = 1e-3
        self.alpha = 0.01

        # self.exploration = "greedy"
        self.exploration = "boltzmann"
        
        input_size = env.observation_space.shape[0]
        actions_size = env.action_space.n
        
        # Réseau de neurones
        self.network = QModel(input_size, actions_size)
        self.target = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.sigma)

        # Buffer
        self.buffr = Buffer(buffr_size)
        self.env = env
        self.reward_sum = 0

        self.cpt_app = 0

    # Stratégie d'exploration (e-greedy/boltzmann) de l'agent
    # input:
    #   * Q : Q-valeurs pour chaque action sous la forme d'une liste
    # output:
    #   * Action décidée
    def act(self, Q):
        tirage = random.random()
        self.tau = self.min_tau + (self.tau - self.min_tau) * self.tau_decay

        if (self.exploration == "greedy"):
            if (tirage >= self.tau):
                return Q.max(0)[1].item()
            else:
                return self.env.action_space.sample()

        elif (self.exploration == "boltzmann"):
            sum_ = 0
            den_ = torch.exp(Q / self.tau).sum()
            for i in range(Q.shape[0]):
                sum_ += torch.exp(Q[i] / self.tau) / den_
                if (sum_ >= tirage):
                    break
            return i

        else:
            raise Exception('Stratégie \'{}\' inconnue.'.format(self.exploration))

    # Pas de l'agent
    def step(self):
        etat_ = self.ob
        x = torch.Tensor(etat_)
        y = self.network.forward(x)
        action = self.act(y)
        self.ob, reward, self.done, _ = self.env.step(action)

        # Stockage de l'interaction
        interaction = Interaction(etat_, action, self.ob, reward, self.done)
        self.buffr.append(interaction)

        # Stockage de la récompense cumulée
        self.reward_sum += reward

    def step_opti(self):
        etat_ = self.ob
        x = torch.Tensor(etat_)
        y = self.network.forward(x)
        action = y.max(0)[1].item()
        self.ob, reward, self.done, _ = self.env.step(action)
        self.reward_sum += reward

    # Reset de l'agent (à chaque nouvel épisode)
    def reset(self):
        self.reward_sum = 0
        self.ob = self.env.reset()

    # Fonction d'apprentissage de l'agent
    def learn(self):
        batch = self.buffr.sample(20)
        for exp in batch:
            self.cpt_app += 1

            for target_param, param in zip(self.target.parameters(), self.network.parameters()):
                target_param.data.copy_(self.alpha * param + (1 - self.alpha) * target_param)
            self.optimizer.zero_grad()

            mse = torch.nn.MSELoss()
            if (not (exp.f)):
                e = torch.Tensor(exp.e)
                s = torch.Tensor(exp.s)

                Q = self.network.forward(e)[exp.a]

                Q2c = self.target.forward(s)

                loss = mse(Q, (exp.r + self.gamma * Q2c.max()))
                loss.backward()
                self.optimizer.step()

            else:
                expe = torch.Tensor(exp.e)
                Q = self.network.forward(expe)[exp.a]
                loss = (Q - exp.r).pow(2)
                loss.backward()
                self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'results/dql_agent'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent2 = DQN_Agent(env, 1000000)

    episode_count = 500
    epoch = 10

    reward = 0
    rewards = []

    done = False

    for i in range(episode_count):
        agent2.reset()
        while True:
            agent2.step()
            if agent2.done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        agent2.learn()
        rewards.append(agent2.reward_sum)

    for i in range(episode_count):
        agent2.reset()
        while True:
            agent2.step_opti()
            if agent2.done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        rewards.append(agent2.reward_sum)

    epochs_rec = []
    for i in range(0, len(rewards), epoch):
        m = rewards[i:i + epoch]
        epochs_rec.append(sum(m) / len(m))

    # Affichage de l'évolution de la récompense
    fig = plt.figure()
    fig.suptitle('Récompense cumumée par épisode', fontsize=12)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Total de la récompense cumulée', fontsize=10)
    plt.plot(rewards)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()