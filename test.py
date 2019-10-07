from DDPG import filter_env
from DDPG.ddpg import *
import gc
from envs.make_env import *
import numpy as np
gc.enable()

ENV_NAME = 'simple'
EPISODES = 100000
TEST = 5
MAX_EPISODE_STEPS = 20


def main():
    #env = filter_env.makeFilteredEnv(make_env(ENV_NAME))
    env = make_env(ENV_NAME)
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in range(MAX_EPISODE_STEPS):
            # actions = np.array([env.action_space[e].sample() for e in range(env.n)])
            action = agent.noise_action(state[0])
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state[0], action[0], reward[0], next_state[0], done[0])
            state = next_state
            if done[0]:
                break
        # Testing:
        if episode % 50 == 0 and episode > 50:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(MAX_EPISODE_STEPS):
                    # env.render()
                    # actions = np.array([env.action_space[e].sample() for e in range(env.n)])
                    action = agent.noise_action(state[0])
                    next_state, reward, done, _ = env.step(action)
                    agent.perceive(state[0], action[0], reward[0], next_state[0], done[0])
                    state = next_state
                    total_reward += reward[0]
                    if done[0]:
                        break
                env.close()
            ave_reward = total_reward/TEST
            print('episode: {0} Evaluation Average Reward: {1}'.format(episode, ave_reward))
    env.monitor.close()


if __name__ == '__main__':
    main()
