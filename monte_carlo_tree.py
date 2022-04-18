import numpy as np
from model.job import States
from copy import deepcopy
from schedule_env import ScheduleEnv
from schedule_kpi import schedule_kpi


class MonteCarloTree:
    def __init__(self, test_env: ScheduleEnv, actions, policy_net):
        self.policy_net = policy_net
        self.test_env = deepcopy(test_env)
        self.actions = actions

    def _expansion(self, job_info, action):
        _ = self.test_env.reset()
        self.test_env.job_info = job_info
        state, _, done = self.test_env.step(action)
        action_queue = [action]

        while not done:
            action_idx = set([i for i in range(len(self.actions))])
            remain_actions = list(action_idx - set(action_queue))
            output = self.policy_net(state)
            action = self._select_action(self.test_env.job_info, output, remain_actions=remain_actions)[0]
            action_queue.append(action)
            state, reward, done = self.test_env.step(action)
        kpi_score = schedule_kpi(self.test_env.job_info.observation, self.test_env.equipments, False)
        return kpi_score, action_queue

    def _action_filter(self, state, select_actions):
        result_actions = []
        for action_idx in select_actions:
            action = self.actions[action_idx]
            eqp_name = action[2]
            op_order = action[1]
            job_name = action[0]
            current_op_order = state.get_current_op_order(job_name)
            select_job = state.observation[(job_name, op_order)]
            if select_job.op_order == current_op_order and not select_job.is_done:
                result_actions.append(action_idx)
        return result_actions

    def _select_action(self, state: States, output, remain_actions=None):
        action_indexes = remain_actions if remain_actions else [i for i in range(len(self.actions))]
        filted_actions = self._action_filter(state, action_indexes)
        sorted_action = [idx for idx in sorted(filted_actions, key=lambda item: output[-1][item].item(), reverse=True)]
        filter_action_value = {i: output[-1][i].item() for i in sorted_action}
        threshold = np.quantile(list(filter_action_value.values()), .75)
        filter_action_value = {action: value for action, value in filter_action_value.items() if value >= threshold}
        return list(filter_action_value.keys())

    def tree_search(self, state, output):
        state_tmp = deepcopy(state)
        self.test_env.job_info = state_tmp
        actions = self._select_action(state_tmp, output)
        action_score = {action: self._expansion(state_tmp, action) for action in actions}
        chosed_action = max(action_score, key=action_score.get)
        return chosed_action

    def light_tree_search(self, state, output):
        state_tmp = deepcopy(state)
        self.test_env.job_info = state_tmp
        actions = self._select_action(state_tmp, output)[:3]
        action_score = {action: self._expansion(state_tmp, action) for action in actions}
        chosed_action = max(action_score, key=action_score.get)
        return action_score[chosed_action][1]
