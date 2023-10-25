import os
import time
import argparse
import numpy as np
from simple_model import MAModel
from simple_agent import MAAgent

import paddle
import parl
from parl.algorithms import MADDPG
from env import MAenv
from parl.utils import logger, summary

#import pdb

CRITIC_LR = 0.001
ACTOR_LR = 0.0005
GAMMA = 0.9
TAU = 0.001
BATCH_SIZE = 128
MAX_EPISODES = 10000
MAX_STEP_PER_EPISODE = 59  
STAT_RATE = 10

def random_shape(action_n):
    for i in range(len(action_n)):
        action_n[i]=np.random.uniform(-1, 1, size=len(action_n[i]))
    return action_n


def run_episode(t, env, agents):
    obs_n = env.reset()
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while True:

        steps += 1

        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        print('obs_n',obs_n)
        if agents[0].rpm.size() <= agents[0].min_memory_size:
            action_n=random_shape(action_n)
        Sbus, _, _ = env.getbus(t)
        next_obs_n, reward_n, done_n, _ = env.step(Sbus, action_n)
        done = all(done_n)
        terminal = (steps >= MAX_STEP_PER_EPISODE)

        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        if done or terminal:
            break

        if args.restore and args.show:
            continue

        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)
            if critic_loss != 0.0:
                summary.add_scalar('critic_loss_%d' % i, critic_loss,
                                   agent.global_train_step)

    return total_reward, agents_reward, steps


def train_agent():
    env = MAenv(args.env)

    critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)
    logger.info('critic_in_dim: {}'.format(critic_in_dim))

    agents = []
    for i in range(env.n):
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim)
        algorithm = MADDPG(
            model,
            agent_index=i,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE,
            speedup=(not args.restore))
        agents.append(agent)
    total_steps = 0
    total_episodes = 0

    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n)]

    if args.restore:
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    t_start = time.time()
    logger.info('Starting...')
    t=0;
    while total_episodes <= MAX_EPISODES:
        t += 180;
        if t>=10080:
            t=t%10080
        ep_reward, ep_agent_rewards, steps = run_episode(t, env, agents)
        summary.add_scalar('train_reward/episode', ep_reward, total_episodes)
        summary.add_scalar('train_reward/step', ep_reward, total_steps)
        if args.show:
            print('episode {}, reward {}, `s rewards {}, steps {}'.format(
                total_episodes, ep_reward, ep_agent_rewards, steps))

        total_steps += steps
        total_episodes += 1
        episode_rewards.append(ep_reward)
        for i in range(env.n):
            agent_rewards[i].append(ep_agent_rewards[i])

        if total_episodes % STAT_RATE == 0:
            mean_episode_reward = round(
                np.mean(episode_rewards[-STAT_RATE:]), 3)
            final_ep_ag_rewards = []
            for rew in agent_rewards:
                final_ep_ag_rewards.append(round(np.mean(rew[-STAT_RATE:]), 2))
            use_time = round(time.time() - t_start, 3)
            logger.info(
                'Steps: {}, Episodes: {}, Mean episode reward: {}, mean agents rewards {}, Time: {}'
                .format(total_steps, total_episodes, mean_episode_reward,
                        final_ep_ag_rewards, use_time))
            t_start = time.time()
            summary.add_scalar('mean_episode_reward/episode',
                               mean_episode_reward, total_episodes)
            summary.add_scalar('mean_episode_reward/step', mean_episode_reward,
                               total_steps)
            summary.add_scalar('use_time/1000episode', use_time,
                               total_episodes)

            if not args.restore:
                model_dir = args.model_dir
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)
                    agents[i].save(model_dir + model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MADDPG_VVC')
    
    parser.add_argument('--env', type=str, default='simple_speaker_listener',
                       help='scenario of MultiAgentEnv')
    parser.add_argument('--show', action='store_true', default=False,
                        help='display or not')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='restore or not, must have model_dir')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='directory for saving model')
    parser.add_argument('--exp_name', type=str, default='inverter',
                        help='save name')
    parser.add_argument('--network_name', type=str, default='ieee33',
                        help='')

    args = parser.parse_args()
    logger.set_dir('./train_log/' + str(args.env))

    train_agent()
