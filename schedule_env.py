import numpy as np
import torch
from model.job import States
from schedule_reward import ScheduleReward
from typing import Tuple, List
from copy import deepcopy


class ScheduleEnv:
    def __init__(self, job_info: States, equipments, job_actions: List, max_step, avg_op_order, actions):
        self.job_info = job_info
        self.deadline = job_info.deadline
        self.obervation_spec = self.job_info.get_observation_spec()
        self.equipments = list(equipments)
        # action, array for jobs*equipments
        self.job_actions = job_info.job_actions
        self.actions = actions
        self.avg_op_order = avg_op_order
        self.schedule_reward = ScheduleReward(self.job_info, self.equipments, self.actions, self.deadline, avg_op_order,
                                              job_actions)
        self.eqps_count = len(self.equipments)
        self.jobs_count = len(self.job_info.jobs_name)
        self._state = self.job_info.get_observation()
        self._episode_ended = False
        self.max_step = max_step
        self.step_count = 0

    def reset(self):
        self.job_info.reset()
        self.job_actions = self.job_info.job_actions
        self._state = self.job_info.get_observation()
        self._episode_ended = False
        self.step_count = 0
        self.schedule_reward = ScheduleReward(self.job_info, self.equipments, self.actions, self.deadline,
                                              self.avg_op_order, self.job_actions)
        return torch.tensor(self._state)

    def refresh(self, job_info: States):
        self.job_info = job_info
        self.schedule_reward = ScheduleReward(self.job_info, self.equipments, self.actions, self.deadline,
                                              self.avg_op_order, self.job_actions)
        self._state = self.job_info.get_observation()

    # input eqp_id
    def update_state(self, eqp_id, job_name, op_order):
        ob = self.job_info.observation[(job_name, op_order)]
        if ob.op_order == self.job_info.get_current_op_order(ob.job_name) and not ob.is_done:
            last_job = self.job_info.get_eqp_last_job(eqp_id)
            eqp_name = self.equipments[eqp_id]
            ob.running_eqp[eqp_id] = 1
            setup_time = self.job_info.get_setup_time(last_job, job_name, eqp_name) if last_job else 0
            # setup_time = self.setup_time.loc[last_job, job_name] if last_job else 0
            pre_op_order_end_time = self.job_info.get_pre_op_order_end_time(job_name,
                                                                            op_order)  # if op order = 1 return -1
            eqp_end_time = self.job_info.get_eqp_end_time(eqp_id)
            checkin_time = max(pre_op_order_end_time, eqp_end_time)  # get latest end time
            start_time = checkin_time + setup_time
            run_time = self.job_info.get_job_run_time(job_name, op_order, eqp_name)
            end_time = start_time + run_time
            ob.is_done = 1
            ob.start_time = start_time
            ob.end_time = end_time
            ob.setup_time = setup_time
            ob.run_time = run_time
        self.update_actions(job_name)
        self._state = self.job_info.get_observation()

    def update_actions(self, job):
        if self.job_info.get_current_op_order(job) == -1:
            for job_action_group in self.job_actions:
                job_action_group.remove(job)

    def step(self, action) -> Tuple[np.array, float, bool]:
        # return observation, reward, done
        # if self._episode_ended:
        #      self.reset()
        self.step_count += 1
        job_action, equipment_action_id = self.actions[action]
        # job_action and equipment_actions are network output
        select_job_group = int(job_action / 2)
        select_job_head_tail = 0 if int(job_action % 2) == 0 else -1
        job_name = self.job_actions[select_job_group][select_job_head_tail]
        equipment_name = self.equipments[equipment_action_id]

        op_order = self.job_info.get_current_op_order(job_name)
        self.job_info.jobs_info[(job_name, op_order)].can_run_eqps.index(1)

        # action_index = action
        # Make sure episodes don't go on forever.
        # action [job,eqp] this job do in this epq
        eqp_index = self.equipments.index(equipment_name)
        # eqp_index = self.equipments.index(eqp_name)
        not_done_job = [v for v in self.job_info.observation.values() if not v.is_done]  # all job is done
        if ((len(not_done_job) == 1 and not_done_job[0].job_name == job_name and not_done_job[
            0].op_order == op_order) or (self.step_count == self.max_step)):
            self._episode_ended = True

        # Agent take infinite step to take action, refine it in 100 steps
        if self._episode_ended:
            self.update_state(eqp_index, job_name, op_order)
            reward = self.schedule_reward.get_episode_ended_reward(self.job_info, self.step_count)
            return torch.tensor(self._state), reward, self._episode_ended
        else:
            reward = self.schedule_reward.get_episode_not_ended_reward(self.job_info, action)
            self.update_state(eqp_index, job_name, op_order)
            return torch.tensor(self._state), reward, self._episode_ended

    def get_current_state(self):
        if self._episode_ended == True:
            return None
        else:
            return torch.tensor(self._state)
