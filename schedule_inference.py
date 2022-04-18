import sys
import time
import pandas as pd
import config
from Module.dueling_dqn import DuelingDQN
from Module.dqn import DQN
from Module.drqn import DRQN
from Module.dueling_drqn import DuelingDRQN
import torch
from schedule_env import ScheduleEnv
from schedule_agent import ScheduleAgent
import torch.optim as optim
from schedule_kpi import schedule_kpi
from collections import defaultdict
from time_util import timedelta_to_datetime
from pathlib import Path


def load_model(preprocessor, attribute_count: int, action_count: int, device) -> torch.nn.Module:
    checkpoint = torch.load(config.MODEL_FILE, map_location=device)
    policy_net = implement_network(checkpoint, len(preprocessor.job_info_dict), action_count, attribute_count, device)
    print("Testing...")
    start_time = time.time()
    load_model_testing(policy_net, preprocessor)
    print("Total Testing Time --> {} seconds".format(str(time.time() - start_time)))
    if not config.IS_RETRAIN:
        sys.exit()
    return policy_net


def load_model_testing(policy_net, preprocessor):
    schedule_env = ScheduleEnv(preprocessor.jobs_info, preprocessor.equipments_list, preprocessor.job_actions,
                               preprocessor.max_step, preprocessor.avg_op_order, preprocessor.action_map)
    schedule_env.reset()
    attribute_count = list(preprocessor.jobs_info.observation.values())[0].get_one_observation().size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.RMSprop(policy_net.parameters(), lr=config.LEARNING_RATE)
    schedule_agent = ScheduleAgent(policy_net, device, len(preprocessor.job_info_dict), attribute_count,
                                   len(schedule_env.actions), optimizer, schedule_env, preprocessor.job_actions,
                                   preprocessor.action_map)
    if config.LIGHT_SEARCH:
        # TODO : fix monte carlo
        # schedule_agent.light_monte_carlo()
        schedule_agent.save_best_observation(preprocessor.equipments_list, preprocessor.job_info_dict,
                                             is_inference=True)
    else:
        schedule_agent.save_best_observation(preprocessor.equipments_list, preprocessor.job_info_dict,
                                             is_inference=True)
    schedule_kpi(schedule_agent.best_observation, preprocessor.equipments_list)
    get_detail_seq(schedule_agent.best_observation, preprocessor.equipments_list)


def implement_network(checkpoint, job_count, action_count, attribute_count, device):
    if checkpoint['class_name'].upper() == 'DQN':
        model = DQN(job_count, attribute_count, action_count, device).to(device)
    elif checkpoint['class_name'].upper() == 'DRQN':
        model = DRQN(job_count, attribute_count, action_count, device).to(device)
    elif checkpoint['class_name'].upper() == 'DUELINGDQN':
        model = DuelingDQN(job_count, attribute_count, action_count, device).to(device)
    elif checkpoint['class_name'].upper() == 'DUELINGDRQN':
        model = DuelingDRQN(job_count, attribute_count, action_count, device).to(device)
    else:
        raise "model not exist"
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        raise "Module shape not match"
    return model


def get_detail_seq(job_info, eqp_list):
    Path(config.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    result = defaultdict(list)
    for ob in job_info.values():
        result['job_name'].append(ob.job_name)
        result['op_order'].append(ob.op_order)
        result['quantity'].append(ob.quantity)
        result['setup_time'].append(ob.setup_time)
        result['start_time'].append(timedelta_to_datetime(ob.start_time))
        result['end_time'].append(timedelta_to_datetime(ob.end_time))
        result['deadline'].append(timedelta_to_datetime(ob.deadline))
        eqp_id = ob.running_eqp.tolist().index(1)
        eqp_name = eqp_list[eqp_id]
        result['equipment'].append(eqp_name)
        result['is_done'].append(ob.is_done)
    df = pd.DataFrame.from_dict(result)
    df.to_csv(Path(config.OUTPUT_PATH)/"rl_detail_sequence.csv")
