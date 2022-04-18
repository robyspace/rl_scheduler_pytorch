from pandas import array
from model import job
import config
from preprocessor import PreProcessor
import numpy as np
import statistics
from typing import List, Dict, Tuple
from model.job import JobObservation


def schedule_kpi(schedule_result: Dict[Tuple, JobObservation], eqp_list, is_display: bool = True):
    equipments_num = len(eqp_list)
    schedule_date = config.SCHEDULE_DAYS * 24 * 60
    utilization = get_utilization(schedule_result, equipments_num, schedule_date)
    ofr = get_ofr(schedule_result, schedule_date)
    capacity = get_capacity(schedule_result, schedule_date)
    cycle_time = get_mean_cycle_time(schedule_result)
    scale_kpi, normal_kpi = get_score(utilization, ofr, capacity, cycle_time, config.KPI_SCORE_WEIGHT.copy())
    if is_display:
        print("達交率 ofr: {}".format(ofr))
        print("平均生產週期 cycle_time： {}".format(cycle_time))
        print("平均稼動率 utilization: {}".format(utilization))
        print("產能 capacity: {}".format(capacity))
        print("KPI score: {}".format(normal_kpi))

    return scale_kpi


def get_utilization(schedule_result, equipments_num, schedule_date):
    # 稼動率
    # 機台run的時間/schedule_date
    # 取平均
    eqp_machine_run_total_time = [0 for i in range(equipments_num)]
    for job_info in schedule_result.values():
        if job_info.is_done:
            eqp_id = job_info.running_eqp.tolist().index(1)
            eqp_machine_run_total_time[eqp_id] += job_info.run_time

    # Average the utilization to all eqp
    # np.delete(arr, np.where(arr == 2))
    eqp_machine_run_total_time = [time for time in eqp_machine_run_total_time if time != 0]
    mean_utilization = statistics.mean(
        list(np.array(eqp_machine_run_total_time) / schedule_date)) if eqp_machine_run_total_time else 0
    return mean_utilization


def get_ofr(schedule_result, schedule_date):
    # 達交率 overfield rate
    # 以工單為單位
    # 完成工單量 / 總工單量
    # job_finish_on_time = 0
    job_finish_dict = {}
    for job_info in schedule_result.values():
        if job_info.job_name in job_finish_dict:
            if job_info.end_time <= job_info.deadline and job_info.end_time <= schedule_date:
                job_finish_dict[job_info.job_name].append(job_info.is_done)
            else:
                job_finish_dict[job_info.job_name].append(0)
        else:
            job_finish_dict[job_info.job_name] = [job_info.is_done]
    return len([1 for is_done_list in job_finish_dict.values() if all(is_done_list) == True]) / len(job_finish_dict)


def get_capacity(schedule_result, schedule_date):
    # 總生產量 / 需生產總數量 (有在進行或者完成都包含在內)
    # 誤！！！！ 待改
    job_finish_on_time_quantity = 0
    total_quantity = {}
    job_quantity = {}
    for job_info in schedule_result.values():
        total_quantity[job_info.job_name] = job_info.quantity
        if job_info.job_name in job_quantity:
            if job_info.is_done != 1:
                job_quantity[job_info.job_name] = 0
        else:
            if job_info.end_time <= schedule_date and job_info.is_done == 1:
                job_quantity[job_info.job_name] = job_info.quantity
    return sum(job_quantity.values()) / sum(total_quantity.values())


def get_mean_cycle_time(schedule_result):
    # 以天為單位，每張工單的平均完成時間，目前是以單工序計算
    # 未來計算公式：取所有工單的 (該工單最晚結束時間 - 該工單最早起始時間) 平均
    jobs_start_time = {}
    jobs_end_time = {}
    for job_info in schedule_result.values():
        if job_info.is_done == True:
            if job_info.job_name in jobs_start_time:
                if jobs_start_time[job_info.job_name] > job_info.start_time:
                    jobs_start_time[job_info.job_name] = job_info.start_time
                if jobs_end_time[job_info.job_name] < job_info.end_time:
                    jobs_end_time[job_info.job_name] = job_info.end_time
            else:
                jobs_start_time[job_info.job_name] = job_info.start_time
                jobs_end_time[job_info.job_name] = job_info.end_time
        else:
            if job_info.job_name in jobs_start_time:
                del jobs_start_time[job_info.job_name]
                del jobs_end_time[job_info.job_name]

    return statistics.mean(
        [jobs_end_time[job_name] - jobs_start_time[job_name] for job_name in
         jobs_start_time.keys()]) / 60 / 24 if jobs_start_time else 0

    # return statistics.mean(
    #     [statistics.mean(op_order_time_list) for op_order_time_list in jobs_cycle_time.values()]) / 60 / 24


def get_score(utilization, ofr, capacity, cycle_time, reward_weight: dict):
    '''
    example:
    reward_weight = {ofr: 1, cycle_time: 2, utilization: 3, capacity: 4} => 這個是 user 設定的優先級 優先級 * 0.1 = 權重
    cycle_time要是負的
    '''
    score_mapping = {1: 100, 2: 70, 3: 35, 4: 10}
    kpi_weight_map = {kpi_name: score_mapping[priority] for kpi_name, priority in reward_weight.items()}
    for key, value in reward_weight.items():
        reward_weight[key] = (5 - value) * 0.1
    x = [utilization, ofr, capacity, cycle_time]
    max_x = max(x)
    min_x = min(x)
    if max_x - min_x != 0:
        scale_x = [(x_i - min_x + 0.01) / (max_x - min_x) for x_i in x]
    else:
        scale_x = [0, 0, 0, 0]
    normal_score = reward_weight["ofr"] * ofr - \
                   reward_weight["cycle_time"] * cycle_time + \
                   reward_weight["utilization"] * utilization + \
                   reward_weight["capacity"] * capacity
    scale_score = kpi_weight_map["ofr"] * scale_x[1] - \
                  kpi_weight_map["cycle_time"] * scale_x[3] + \
                  kpi_weight_map["utilization"] * scale_x[0] + \
                  kpi_weight_map["capacity"] * scale_x[2]
    return scale_score, normal_score
