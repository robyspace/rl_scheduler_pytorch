import numpy as np
from typing import Dict, List, NamedTuple, Tuple
from copy import deepcopy


class JobInfo:
    def __init__(self, job_name: str, product: str, run_time: List[float], deadline: float, can_run_eqps, op_order: int,
                 quantity: int, operation: str, group_id):
        self.job_name = job_name
        self.product = product
        self.run_time = run_time
        self.deadline = deadline
        self.can_run_eqps = can_run_eqps
        self.op_order = op_order
        self.quantity = quantity
        self.operation = operation
        self.group_id = group_id


class States:
    def __init__(self, jobs_info: Dict[Tuple, JobInfo], setup_time, equipment_list, job_actions):
        self.jobs_info = jobs_info
        self.equipment_list = list(equipment_list)
        self.jobs_name = [name[0] for name in jobs_info.keys()]
        self.op_orders = [name[1] for name in jobs_info.keys()]
        self.job_quantity = [job_info.quantity for job_info in jobs_info.values()]
        self.setup_time_map = setup_time
        self.job_run_time = [job_info.run_time for job_info in jobs_info.values()]
        self.eqp_count = len(equipment_list)
        self.deadline = {job_info.job_name: job_info.deadline for job_info in jobs_info.values()}
        self.observation = {}
        self.job_actions = job_actions
        self.job_actions_back = deepcopy(job_actions)
        for job_info in jobs_info.values():
            self.observation[(job_info.job_name, job_info.op_order)] = JobObservation(job_info, self.eqp_count)
            # self.observation.append(JobObservation(job_info, self.eqp_count))

    def reset(self):
        self.observation.clear()
        self.job_actions = deepcopy(self.job_actions_back)
        for job_info in self.jobs_info.values():
            self.observation[(job_info.job_name, job_info.op_order)] = JobObservation(job_info, self.eqp_count)
            # self.observation.append(JobObservation(job_info, self.eqp_count))

    # input eqp_id
    # get this equipment last job
    # return last job name
    def get_eqp_last_job(self, eqp_id):
        run_eqp = [0 if i != eqp_id else 1 for i in range(self.eqp_count)]
        last_end_time = max([ob.end_time for ob in self.observation.values()
                             if list(ob.running_eqp) == run_eqp], default=-1)

        if last_end_time != -1:
            last_job = [ob.job_name for ob in self.observation.values() if
                        list(ob.running_eqp) == run_eqp and ob.end_time == last_end_time][0]
        else:
            last_job = None
        return last_job

    def get_observation_spec(self):
        # return np.array([n.get_one_observation() for n in self.observation]).shape
        return np.transpose(np.array([n.get_one_observation() for n in self.observation.values()])).shape

    def get_observation(self):
        # return np.array([n.get_one_observation() for n in self.observation])
        attribute_count = list(self.observation.values())[0].get_one_observation().size

        state = np.transpose(np.array([n.get_one_observation() for n in self.observation.values()])).reshape(1,
                                                                                                             attribute_count,
                                                                                                             len(self.jobs_info))
        # need Standardization index [1,2,3,5,6,7]
        need_standardization = [1, 2, 3, 5, 6, 7]
        for i, row in enumerate(state[0]):
            if i in need_standardization:
                std = np.std(row)
                avg = np.mean(row)
                state[0][i] = np.apply_along_axis(lambda x: (x - avg) / (std + 0.00001), 0, arr=row)
        return state

    def get_eqp_end_time(self, eqp_id):
        eqp_state = [0 for _ in range(self.eqp_count)]
        eqp_state[eqp_id] = 1
        eqp_end_time = max([i.end_time for i in self.observation.values() if list(i.running_eqp) == eqp_state],
                           default=0)
        return eqp_end_time

    def get_utilization_rate(self):
        # eqp_pair collect all Jobs on equipments, if not put the job on the machine, it will get (0, 0)
        # output -> [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
        eqp_pair = []
        for ob in self.observation.values():
            if list(ob.running_eqp) not in eqp_pair:
                eqp_pair.append(list(ob.running_eqp))

        # eqp_schedule_time -> the max cost time in all equipments, eg. (75, 100) -> 100
        eqp_schedule_time = max([ob.end_time for ob in self.observation.values()])

        # eqp_operation_time -> all equiments cost time including setup time, eg. [75, 120]
        eqp_operation_time = []
        for pair in eqp_pair:
            total_cost_time_for_one_equiment = 0
            for ob in self.observation.values():
                if ob.is_done and pair == list(ob.running_eqp):
                    total_cost_time_for_one_equiment += ob.run_time
            eqp_operation_time.append(total_cost_time_for_one_equiment)

        # utilization -> total time exclude setup time / max cost time
        utilize = []
        for i in range(len(eqp_pair)):
            utilize_value = 0 if eqp_schedule_time == 0 else eqp_operation_time[i] / eqp_schedule_time
            utilize.append(utilize_value)
        utilize_rate = sum(utilize) / self.eqp_count
        return utilize_rate

    def get_job_run_time(self, job_name, op_order, equipment_name):
        eqp_idx = self.equipment_list.index(equipment_name)
        ob = self.jobs_info[(job_name, op_order)]

        # run_time = [job_info.run_time[eqp_idx] for job_info in self.jobs_info
        #             if job_info.job_name == job_name and job_info.op_order == op_order][0]
        return ob.run_time[eqp_idx]

    def get_current_op_order(self, job_name):
        last_op_order = max([ob.op_order for ob in self.observation.values() if ob.job_name == job_name and ob.is_done],
                            default=0)  # 0 means no job is done
        current_job = self.observation[(job_name, last_op_order + 1)] if (job_name,
                                                                          last_op_order + 1) in self.observation else None
        # current_job = [ob for ob in self.observation if ob.job_name == job_name and ob.op_order == last_op_order + 1]
        if current_job:
            return current_job.op_order
        else:
            return -1  # -1 means the jobs is finished

    def get_pre_op_order_end_time(self, job_name, op_order):
        end_time = -1
        if op_order > 1:
            ob = self.observation[(job_name, op_order)]
            end_time = ob.end_time
            # end_time = [ob.end_time for ob in self.observation
            #             if job_name == ob.job_name
            #             and ob.op_order == op_order - 1][0]
        return end_time

    def get_job_can_run_eqp(self, job_name):
        current_op_order = self.get_current_op_order(job_name)
        can_run_eqps = [self.equipment_list[i] for i, item in
                        enumerate(self.jobs_info[(job_name, current_op_order)].can_run_eqps)
                        if item == 1]

        return can_run_eqps

    def check_eqp_can_run_job(self, job_name, op_order, equipment_id):
        can_run_eqps = self.jobs_info[(job_name, op_order)].can_run_eqps
        # equipment_name = self.equipment_list[equipment_id]
        if can_run_eqps[equipment_id] == 1:
            return True
        else:
            return False

    def get_running_eqps(self):
        running_eqps = set([ob.running_eqp.tolist().index(1) for ob in self.observation.values() if ob.is_done])
        return running_eqps

    def get_setup_time(self, last_job, current_job, eqp_name):
        # get setup time by job and equipment name
        # not stage
        last_group = self.jobs_info[(last_job, 1)].group_id
        current_group = self.jobs_info[(current_job, 1)].group_id
        time = self.setup_time_map[(self.setup_time_map['UpperGroup'] == last_group) &
                                   (self.setup_time_map['LowerGroup'] == current_group) &
                                   (self.setup_time_map['Equipment'] == eqp_name)]['Time'].values[0]
        return time


class JobObservation:
    def __init__(self, job_info, equipment_count):
        self.op_order = job_info.op_order
        self.eqp_count = equipment_count
        self.is_done: float = 0.0
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.run_time = 0
        self.setup_time: float = 0.0
        self.can_run_eqp = job_info.can_run_eqps
        self.running_eqp = np.zeros((self.eqp_count,))
        self.job_name = job_info.job_name
        self.quantity = job_info.quantity
        self.deadline = job_info.deadline
        self.eqp_run_time = job_info.run_time

    def get_one_observation(self):
        observation = [self.is_done,
                       self.start_time,
                       self.end_time,
                       self.setup_time, self.op_order, self.run_time, self.quantity,
                       self.deadline] + list(self.eqp_run_time) + list(self.can_run_eqp) + list(self.running_eqp)
        return np.array(observation, dtype=float)

# [is_done,start_time,end_time,setup_time,op order,run_time,quantity,deadline,can_run_eqp,running_eqp]
