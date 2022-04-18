import sys
import time
import pandas as pd
import config
from model.job import States, JobInfo
from itertools import product
from datetime import datetime, timedelta
import statistics

import numpy as np
from time_util import datetime_to_timedelta, timedelta_to_datetime


def datetime_to_minute(start_time, target_date):
    time_delta = target_date - start_time
    return time_delta.total_seconds() / 60


class PreProcessor:
    def __init__(self):
        self.action_map = []
        self.action_count = 0
        self.avg_op_order = 0  # 避免多工序，尾數工序排不到，新增reawrd 如果取得大於avg op order 就多給分
        self.jobs_info = None
        self.job_info_dict = []
        self.setup_time_map = pd.DataFrame()
        self.job_info: States
        self.equipments = []
        self.max_step = None
        self.equipments_list = []
        self.equipment_actions = []
        self.job_actions = []

    def job_classify(self):
        job_max_op_order = {k: v for k, v in self.job_info_dict.keys()}
        op_order_count_jobs = list(
            {k: v for k, v in sorted(job_max_op_order.items(), key=lambda item: item[1])}.keys())
        deadline_jobs = list(
            {k[0]: v for k, v in sorted(self.job_info_dict.items(), key=lambda item: item[1].deadline)}.keys())
        avg_run_time_jobs = list(
            {k[0]: v for k, v in
             sorted(self.job_info_dict.items(), key=lambda item: statistics.mean(item[1].run_time))}.keys())
        return [op_order_count_jobs, deadline_jobs, avg_run_time_jobs]

    def job_preprocess(self, orders: pd.DataFrame(), process_time: pd.DataFrame(), group: pd.DataFrame(),
                       equipment_list, start_time):
        # use order and process_time to init jobs and job observation
        job_info_dict = {}
        job_total_info = orders.merge(process_time, on=['Product']).reset_index(drop=True)
        job_oporder_pair = job_total_info.loc[:,
                           ['Job', 'OPOrder', 'Product', 'Deadline', 'Quantity',
                            'Operation']].drop_duplicates().reset_index(drop=True)
        job_oporder_pair['Deadline'] = pd.to_datetime(job_oporder_pair['Deadline'])
        job_oporder_pair['Deadline'] = job_oporder_pair['Deadline'].apply(lambda x: datetime_to_minute(start_time, x))

        for index in job_oporder_pair.index:
            can_run_eqps_time = job_total_info.query(
                'Job =="{0}" and OPOrder =={1}'.format(job_oporder_pair.loc[index, 'Job'],
                                                       job_oporder_pair.loc[index, 'OPOrder'])).loc[:,
                                ['Equipment', 'Time']]
            can_run_eqps_time_dict = can_run_eqps_time.set_index('Equipment').T.to_dict('list')

            can_run_eqps_encode = [1.0 if eqp in can_run_eqps_time_dict else 0.0 for eqp in
                                   equipment_list]

            run_time = [can_run_eqps_time_dict[eqp][0] * job_oporder_pair.loc[index, 'Quantity'] / 60
                        if eqp in can_run_eqps_time_dict else 0.0
                        for eqp in equipment_list]
            product = job_oporder_pair.loc[index, 'Product']
            group_id = group[group['Product'] == product]['Group'].values[0]
            job_name = str(job_oporder_pair.loc[index, 'Job'])

            job_info_dict[(job_name, job_oporder_pair.loc[index, 'OPOrder'])] = JobInfo(job_name=job_name,
                                                                                        product=product,
                                                                                        run_time=run_time,
                                                                                        deadline=job_oporder_pair.loc[
                                                                                            index, 'Deadline'],
                                                                                        can_run_eqps=can_run_eqps_encode,
                                                                                        op_order=job_oporder_pair.loc[
                                                                                            index, 'OPOrder'],
                                                                                        quantity=job_oporder_pair.loc[
                                                                                            index, 'Quantity'],
                                                                                        operation=job_oporder_pair.loc[
                                                                                            index, 'Operation'],
                                                                                        group_id=group_id)
        return job_info_dict

    def setup_time_preprocess(self, setup_time: pd.DataFrame, equipments: pd.DataFrame, orders: pd.DataFrame):
        # 改為 group1 ~ group2 在 某個 eqp 上的轉換時間
        setup_time_eqp = setup_time.merge(equipments, on=['Stage']).drop(
            columns=['Stage', 'Operation', 'Position']).drop_duplicates()
        return setup_time_eqp
        # jobs_combination = pd.DataFrame(list(product(orders['Job'].values, orders['Job'].values)),
        #                                 columns=['UpperJob', 'LowerJob'])
        # jobs_combination['UpperProduct'] = jobs_combination['UpperJob'].apply(
        #     lambda job: orders[orders['Job'] == job]['Product'].values[0])
        # jobs_combination['LowerProduct'] = jobs_combination['LowerJob'].apply(
        #     lambda job: orders[orders['Job'] == job]['Product'].values[0])
        # jobs_combination = jobs_combination.merge(group, left_on=['UpperProduct'], right_on=['Product']).rename(
        #     columns={'Group': 'UpperGroup'}).drop(columns=['Product'])
        # jobs_combination = jobs_combination.merge(group, left_on=['LowerProduct'], right_on=['Product']).rename(
        #     columns={'Group': 'LowerGroup'}).drop(columns=['Product'])
        # jobs_combination = jobs_combination.merge(setup_time, on=['UpperGroup', 'LowerGroup'])
        # jobs_combination_setup_time = jobs_combination.loc[:, ['UpperJob', 'LowerJob', 'Time']]
        # setup_time_df = pd.pivot_table(jobs_combination_setup_time, values='Time', index=['UpperJob'],
        #                                columns=['LowerJob'])

        # return setup_time_df
        pass

    def equipment_preprocess(self, orders, process_time):
        job_total_info = orders.merge(process_time, on=['Product']).reset_index(drop=True)
        eqp_list = job_total_info['Equipment'].unique()
        return eqp_list

    def process(self):
        try:
            setup_time_df = pd.read_csv(config.SETUP_TIME_PATH)
            orders_df = pd.read_csv(config.ORDER_PATH)
            process_time_df = pd.read_csv(config.PROCESS_TIME_PATH)
            group_df = pd.read_csv(config.GROUP_PATH)
            equipments_df = pd.read_csv(config.EQUIPMENTS_PATH)
            start_time = datetime.strptime(config.START_TIME, "%Y-%m-%d")

        except Exception as e:
            raise e
        self.setup_time_map = self.setup_time_preprocess(setup_time_df, equipments_df, orders_df)

        # self.setup_time_map = setup_time_df
        self.equipments_list = self.equipment_preprocess(orders_df, process_time_df)
        self.job_info_dict = self.job_preprocess(orders_df, process_time_df, group_df, self.equipments_list, start_time)
        self.avg_op_order = int(max([j.op_order for j in self.job_info_dict.values()]) / 2)
        self.job_actions = self.job_classify()

        self.jobs_info = States(self.job_info_dict, self.setup_time_map, self.equipments_list, self.job_actions)
        self.equipment_actions = [i for i in range(len(self.equipments_list))]
        self.action_map = list(product([i for i in range(len(self.job_actions) * 2)],
                                       [i for i in range(len(self.equipment_actions))]))
        self.max_step = len(self.job_info_dict)
