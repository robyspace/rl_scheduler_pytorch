import statistics

from schedule_kpi import schedule_kpi
from typing import Dict, List
from Interface.reward_interface import Reward
from model.job import States
from collections import defaultdict


class ScheduleReward(Reward):
    def __init__(self, jobs_info: States, equipments_info: List[str], actions, deadline, avg_op_orders: int,
                 job_actions: List):
        self.jobs_info = jobs_info
        self.jobs_run_time = jobs_info.job_run_time
        self.eqps = equipments_info
        self.jobs_count = len(self.jobs_info.jobs_name)
        self.eqps_count = len(self.eqps)
        self.actions = actions
        self.deadline = deadline
        self.avg_op_orders = avg_op_orders
        self.job_actions = job_actions

    def get_episode_ended_reward(self, state, step_count) -> float:
        # return self.get_utilization_rate_reward(state) + self.get_deadline_reward(
        #     state) + self.get_wait_time_reward(state) + self.get_cycle_time_reward(state)
        return self.get_kpi_reward(state)
        # Positive Reward

    def get_more_eqp_reward(self, state):
        eqp_set = set()
        for ob in state.observation.values():
            eqp_id = ob.running_eqp.tolist().index(1)
            eqp_set.add(eqp_id)
        return len(eqp_set) * 10

    def get_utilization_rate_reward(self, state) -> float:
        return state.get_utilization_rate() * 50

    # Negative Reward

    def get_cycle_time_reward(self, state):
        # 計算每個op order 平均完成時間  job/天
        jobs_start_time = {}
        jobs_end_time = {}
        for job_info in state.observation.values():
            if job_info.is_done == True:
                if job_info.job_name in jobs_start_time:
                    if jobs_start_time[job_info.job_name] > job_info.start_time:
                        jobs_start_time[job_info.job_name] = job_info.start_time
                    if jobs_end_time[job_info.job_name] < job_info.end_time:
                        jobs_end_time[job_info.job_name] = job_info.end_time
                else:
                    jobs_start_time[job_info.job_name] = job_info.start_time
                    jobs_end_time[job_info.job_name] = job_info.end_time

        cycle_time = statistics.mean(
            [jobs_end_time[job_name] - jobs_start_time[job_name] for job_name in jobs_start_time.keys()]) / 60 / 24
        cycle_time_func = lambda x: -100 * x + 400
        return cycle_time_func(cycle_time)

    def get_episode_not_ended_reward(self, current_state: States, action):
        reward = 0
        """
        action index
        0 -> ['J0' 'E0']
        1 -> ['J0' 'E1']
        2 -> ['J1' 'E0']
        3 -> ['J1' 'E1']
        4 -> ['J2' 'E0']
        5 -> ['J2' 'E1']
        6 -> ['J3' 'E0']
        7 -> ['J3' 'E1']
        8 -> ['J4' 'E0']
        9 -> ['J4' 'E1']
        """
        job_actions = current_state.job_actions

        job_action, equipment_actions = self.actions[action]
        select_job_group = int(job_action / 2)
        select_job_head_tail = 0 if int(job_action % 2) == 0 else -1
        job_name = job_actions[select_job_group][select_job_head_tail]
        op_order = current_state.get_current_op_order(job_name)
        reward += self.get_kpi_reward(current_state)
        reward += self.get_job_end_reward(current_state, job_name, op_order, equipment_actions)
        # reward += self.get_less_op_order_reward(current_state, job_name)

        return reward

    def get_job_end_reward(self, state, job_name, op_order, equipment_id):
        if (job_name, op_order + 1) not in state.observation:
            deadline = state.observation[(job_name, op_order)].deadline
            last_job = state.get_eqp_last_job(equipment_id)
            last_job_end_time = state.get_pre_op_order_end_time(job_name, op_order)
            eqp_end_time = state.get_eqp_end_time(equipment_id)
            checkin_time = max(last_job_end_time, eqp_end_time)
            setup_time = self.jobs_info.get_setup_time(last_job, job_name, self.eqps[equipment_id]) if last_job else 0
            # setup_time = self.setup_time.loc[last_job, job_name] if last_job else 0
            start_time = checkin_time + setup_time
            run_time = state.get_job_run_time(job_name, op_order, self.eqps[equipment_id])
            end_time = start_time + run_time
            if end_time >= deadline:
                delta = end_time - deadline
                return delta * -0.5
            else:
                return 100
        return 0

    def get_deadline_reward(self, state):
        # Calculate the jobs end_time, if the end_time > deadline, will cost * (-2)
        over_deadline_time = 0
        over_deadline_jobs = set()
        jobs_end_time = {}
        for job_info in state.observation.values():
            if job_info.end_time > self.deadline[job_info.job_name]:
                over_deadline_time += (job_info.end_time - self.deadline[job_info.job_name])
            if job_info.job_name in jobs_end_time:
                if jobs_end_time[job_info.job_name] < job_info.end_time:
                    jobs_end_time[job_info.job_name] = job_info.end_time
            else:
                jobs_end_time[job_info.job_name] = job_info.end_time

        in_deadline_jobs = [job_name for job_name, end_time in jobs_end_time.items()
                            if end_time < self.deadline[job_name]]
        return over_deadline_time * -0.5 + len(in_deadline_jobs) * 5

    def get_wait_time_reward(self, state):
        # Agent choose the smallest setup time
        # eg.
        # J0 -> J1: 12
        # J0 -> J4: 20
        # Agent should choose the J0 -> J1
        # This function will collect all the setup time, and * (-2)
        wait_time = 0
        # eqp_run_time = {}
        eqp_run_time = defaultdict(int)
        eqp_end_time = defaultdict(int)
        # eqp_wait_time = defaultdict(int)
        for ob in state.observation.values():
            if ob.is_done:
                # get eqp wait time
                eqp_id = list(ob.running_eqp).index(1)
                eqp_run_time[eqp_id] += ob.run_time
                if ob.end_time > eqp_end_time[eqp_id]:
                    eqp_end_time[eqp_id] = ob.end_time
        for eqp_id, run_time in eqp_run_time.items():
            wait_time += eqp_end_time[eqp_id] - run_time
        return wait_time * -0.05

    def get_kpi_reward(self, state):
        kpi = schedule_kpi(state.observation, self.eqps, is_display=False)
        return kpi * 10

    def get_less_op_order_reward(self, state: States, job_name):
        is_empty = False
        op_order = 1
        last_operation = 0
        while not is_empty:
            if (job_name, op_order) in state.observation:
                if not state.observation[(job_name, op_order)].is_done:
                    last_operation += 1
            else:
                is_empty = True
            op_order += 1

        return last_operation * -50 / 9 + (50 / 9 + 50)
