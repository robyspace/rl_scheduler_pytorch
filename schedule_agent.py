import config
from model_logger import gantt_result, clear_result
import torch
import random
import math
from model.replay_memory import ReplayMemory, Transition
import torch.nn as nn
from model_logger import clear_result
from copy import deepcopy
from model.job import States
from monte_carlo_tree import MonteCarloTree
from schedule_kpi import schedule_kpi


class ScheduleAgent:
    def __init__(self, policy_net, device, job_count, attribute_count, action_count, optimizer, test_env, job_actions,
                 actions):
        self.best_observation_array = None
        self.device = device
        self.policy_net = policy_net
        self.target_net = deepcopy(self.policy_net)
        self.policy_net.eval()
        self.target_net.eval()
        self.policy_net.train()
        self.target_net.train()
        self.memory = ReplayMemory(config.REPLAY_BUFFER_CAPACITY)
        self.job_count = job_count
        self.attribute_count = attribute_count
        self.action_count = action_count
        self.optimizer = optimizer
        self.steps_done = 0
        self.best_observation = None
        self.best_kpi_score = 0
        # self.monte_carlo_tree = MonteCarloTree(test_env, actions, policy_net)
        # TODO : modify monte carlo
        self.test_env = deepcopy(test_env)
        self.job_actions = deepcopy(job_actions)
        self.job_actions_back = job_actions
        self.actions = actions

    def test_agent(self, job_count: int, equipment_list, display_status: bool = True, num_episodes=5,
                   tree_search: bool = False):
        for i in range(num_episodes):
            state = self.test_env.reset()
            if display_status:
                print("num_episodes--->", i)
            step = 0
            done = False
            total_return = 0
            episode_return = 0
            if display_status:
                while not done:
                    print("step {}:".format(step))
                    print(gantt_result(self.test_env.job_info, equipment_list))
                    # print(clear_result(state, job_count))
                    action = self.select_action(state, self.test_env.job_info, is_train=False,
                                                tree_search=tree_search)
                    state, reward, done = self.test_env.step(action)
                    if done:
                        clear_result(state, job_count, len(equipment_list))
                    step += 1
            else:
                while not done:
                    action = self.select_action(state, self.test_env.job_info, is_train=False,
                                                tree_search=tree_search)
                    state, reward, done = self.test_env.step(action)
                    if done:
                        episode_return = reward
                total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return, self.test_env.job_info.observation

    def select_action(self, state, job_info: States, is_train=True, tree_search: bool = False):
        sample = random.random()
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                        math.exp(-1. * self.steps_done / config.EPS_DECAY)
        self.steps_done += 1
        if is_train:
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    q_values = self.policy_net(state)
                    sorted_q_value = q_values.sort()[1][-1]
                    is_enable = False
                    idx = -1
                    while not is_enable:
                        action_id = sorted_q_value[idx]
                        is_enable = self.check_action_is_legal(action_id, job_info)
                        idx -= 1

                    return action_id.view(1, 1).to(self.device)
            else:
                actions = [i for i in range(self.action_count)]
                is_enable = False
                while not is_enable:
                    action_id = torch.tensor([[random.choice(actions)]], device=self.device, dtype=torch.long)
                    is_enable = self.check_action_is_legal(action_id, job_info)
                    index = actions.index(action_id)
                    actions.pop(index)
                return action_id.view(1, 1).to(self.device)
        else:
            if tree_search:
                # output = self.policy_net(state)
                # action_index = self.monte_carlo_tree.tree_search(job_info, output)
                # return action_index
                # TODO : fix monte carlo
                q_values = self.policy_net(state)
                sorted_q_value = q_values.sort()[1][-1]
                is_enable = False
                idx = -1
                while not is_enable:
                    action_id = sorted_q_value[idx]
                    is_enable = self.check_action_is_legal(action_id, job_info)
                    idx -= 1

                return action_id.view(1, 1).to(self.device)
                # return self.policy_net(state).max(-1)[1].view(1, 1)
            else:
                q_values = self.policy_net(state)
                sorted_q_value = q_values.sort()[1][-1]
                is_enable = False
                idx = -1
                while not is_enable:
                    action_id = sorted_q_value[idx]
                    is_enable = self.check_action_is_legal(action_id, job_info)
                    idx -= 1

                return action_id.view(1, 1).to(self.device)

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        transitions = self.memory.sample(config.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(config.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * config.DISCOUNT) + reward_batch

        # Compute Huber loss
        # mse = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = mse(state_action_values, expected_state_action_values.unsqueeze(1).float())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_best_observation(self, equipments_list, job_info_list, is_inference: bool = False):
        state = self.test_env.reset()
        done = False
        best_observation_step = []
        while not done:
            best_observation_step.append(state.numpy())
            action = self.select_action(state, self.test_env.job_info, is_train=False, tree_search=is_inference)
            state, reward, done = self.test_env.step(action)
        best_observation_step.append(state.numpy())
        print("best result--> ")
        clear_result(best_observation_step[-1], len(job_info_list), len(equipments_list))
        self.best_observation_array = best_observation_step
        self.best_observation = self.test_env.job_info.observation

    def light_monte_carlo(self):
        state = self.test_env.reset()
        output = self.policy_net(state)
        action_queue = self.monte_carlo_tree.light_tree_search(self.test_env.job_info, output)
        for action in action_queue:
            state, reward, done = self.test_env.step(action)
        self.best_observation = self.test_env.job_info.observation

    def check_action_is_legal(self, action_id, job_info):
        job_action, equipment_id = self.actions[action_id]
        select_job_group = int(job_action / 2)
        select_job_head_tail = 0 if int(job_action % 2) == 0 else -1
        job_name = job_info.job_actions[select_job_group][select_job_head_tail]
        op_order = job_info.get_current_op_order(job_name)
        is_enable = job_info.check_eqp_can_run_job(job_name, op_order, equipment_id)
        return is_enable
