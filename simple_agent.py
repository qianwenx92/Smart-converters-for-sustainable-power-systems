import parl
import paddle
import numpy as np
from parl.utils import ReplayMemory
from parl.utils import machine_info, get_gpu_count


class MAAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None,
                 speedup=False, expl_noise=0.1):
        assert isinstance(agent_index, int)
        assert isinstance(obs_dim_n, list)
        assert isinstance(act_dim_n, list)
        assert isinstance(batch_size, int)
        assert isinstance(speedup, bool)
        self.agent_index = agent_index
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = batch_size
        self.speedup = speedup
        self.n = len(act_dim_n)

        self.memory_size = int(1e5)
        self.min_memory_size = batch_size * 25
        self.rpm = ReplayMemory(
            max_size=self.memory_size,
            obs_dim=self.obs_dim_n[agent_index],
            act_dim=self.act_dim_n[agent_index])
        self.global_train_step = 0

        super(MAAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)
        self.expl_noise = expl_noise

    def sample(self, obs, use_target_model=False):
        """ predict action by model or target_model
        """
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim_n[self.agent_index])
        action = (action_numpy + action_noise).clip(-1, 1)
        return action


    def predict(self, obs, use_target_model=False):
        """ predict action by model or target_model
        """
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        act = self.alg.predict(obs, use_target_model=use_target_model)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def learn(self, agents):
        """ sample batch, compute q_target and train
        """
        self.global_train_step += 1

        if self.global_train_step % 5 != 0:
            return 0.0

        if self.rpm.size() <= self.min_memory_size:
            return 0.0

        batch_obs_n = []
        batch_act_n = []
        batch_obs_next_n = []

        rpm_sample_index = self.rpm.make_index(self.batch_size)
        for i in range(self.n):
            batch_obs, batch_act, _, batch_obs_next, _ \
                = agents[i].rpm.sample_batch_by_index(rpm_sample_index)
            batch_obs_n.append(batch_obs)
            batch_act_n.append(batch_act)
            batch_obs_next_n.append(batch_obs_next)
        _, _, batch_rew, _, batch_isOver = self.rpm.sample_batch_by_index(
            rpm_sample_index)
        batch_obs_n = [
            paddle.to_tensor(obs, dtype='float32') for obs in batch_obs_n
        ]
        batch_act_n = [
            paddle.to_tensor(act, dtype='float32') for act in batch_act_n
        ]
        batch_rew = paddle.to_tensor(batch_rew, dtype='float32')
        batch_isOver = paddle.to_tensor(batch_isOver, dtype='float32')

        target_act_next_n = []
        batch_obs_next_n = [
            paddle.to_tensor(obs, dtype='float32') for obs in batch_obs_next_n
        ]
        for i in range(self.n):
            target_act_next = agents[i].alg.predict(
                batch_obs_next_n[i], use_target_model=True)
            target_act_next = target_act_next.detach()
            target_act_next_n.append(target_act_next)
        target_q_next = self.alg.Q(
            batch_obs_next_n, target_act_next_n, use_target_model=True)
        target_q = batch_rew + self.alg.gamma * (
            1.0 - batch_isOver) * target_q_next.detach()

        # learn
        critic_cost = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        critic_cost = critic_cost.cpu().detach().numpy()[0]

        return critic_cost

    def add_experience(self, obs, act, reward, next_obs, terminal):
        self.rpm.append(obs, act, reward, next_obs, terminal)
