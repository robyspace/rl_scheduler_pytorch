import operator
import pandas as pd
import numpy as np
from model.job import States


def clear_result(observation_result, job_counts: int, equipment_count):
    observation_result = observation_result
    ob_shapes = observation_result.shape
    observation_result = np.transpose(observation_result).reshape((1, ob_shapes[2], ob_shapes[1]))
    # 只取前面資訊，後半段can_run_eqp和run_eqp不要
    observation_result_former = observation_result[:, :, :-equipment_count * 2]
    # 只取最後的run_eqp
    # 取消one-hot-encoding -> index number
    # 並回去原本結果
    observation_result_latter = observation_result[:, :, -equipment_count:]
    eqp_index = [np.argmax(i) for i in observation_result_latter[0]]
    eqp_index_list = []
    for index in eqp_index:
        if index != 0:
            eqp_index_list.append(index + 1)
        else:
            eqp_index_list.append(0)
    observation_result_latter = np.array(eqp_index_list).reshape((1, job_counts, 1))
    result_list = np.round(np.concatenate([observation_result_former, observation_result_latter], axis=-1), 2).tolist()
    for line in result_list[0]:
        print(line)
    # return np.concatenate([observation_result_former, observation_result_latter], axis=-1).tolist()


def gantt_result(job_state: States, equipment_list):
    is_done_jobs = [ob for ob in job_state.observation.values() if ob.is_done]
    is_done_jobs = sorted(is_done_jobs, key=operator.attrgetter('start_time'))
    gantt_df = pd.DataFrame(index=equipment_list, columns=[i for i in range(len(is_done_jobs))])
    for jobs in is_done_jobs:
        eqp_id = list(jobs.running_eqp).index(1)
        eqp_name = equipment_list[eqp_id]
        column_num = gantt_df.loc[eqp_name, :].dropna().shape[0]
        gantt_df.loc[eqp_name, column_num] = jobs.job_name + "_" + str(jobs.op_order)
    print(gantt_df.dropna(how='all'))
