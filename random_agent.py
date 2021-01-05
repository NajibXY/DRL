import argparse
import torch
import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

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
    outdir = 'results/random_agent'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    rewards = []

    for i in range(episode_count):
        ob = env.reset()
        step = 0
        reward_tot = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            reward_tot = reward_tot + reward
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
            rewards.append(reward_tot)

    # Affichage de l'évolution de la récompense
    fig = plt.figure()
    fig.suptitle('Récompense cumumée par épisode', fontsize=12)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Total de la récompense cumulée', fontsize=10)
    plt.plot(rewards)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()