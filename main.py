import time
from schedule_env import ScheduleEnv
import torch
from preprocessor import PreProcessor
from Module.dqn import DQN
from Module.drqn import DRQN
from Module.dueling_drqn import DuelingDRQN
import torch.optim as optim
from itertools import count
import config
import matplotlib.pyplot as plt
from schedule_kpi import schedule_kpi
from schedule_inference import load_model
from model_logger import clear_result, gantt_result
from schedule_agent import ScheduleAgent
import warnings
warnings.filterwarnings("ignore")  # ignore for server 2
# Cuda detect
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU State:', device)
s_time = time.time()
print('Start preprocess ...')
preprocessor = PreProcessor()
preprocessor.process()
print('preprocess End  {} seconds'.format(str(time.time() - s_time)))
schedule_env = ScheduleEnv(preprocessor.jobs_info, preprocessor.equipments_list, preprocessor.job_actions,
                           preprocessor.max_step, preprocessor.avg_op_order, preprocessor.action_map)
test_env = ScheduleEnv(preprocessor.jobs_info, preprocessor.equipments_list, preprocessor.job_actions,
                       preprocessor.max_step, preprocessor.avg_op_order, preprocessor.action_map)
schedule_env.reset()
attribute_count = list(preprocessor.jobs_info.observation.values())[0].get_one_observation().size

if config.LOAD_MODEL:
    policy_net = load_model(preprocessor, attribute_count, len(schedule_env.actions), device)
else:
    policy_net = DuelingDRQN(len(preprocessor.job_info_dict), attribute_count, len(preprocessor.action_map), device).to(device)

optimizer = optim.RMSprop(policy_net.parameters(), lr=config.LEARNING_RATE)
schedule_agent = ScheduleAgent(policy_net, device, len(preprocessor.job_info_dict), attribute_count,
                               len(schedule_env.actions), optimizer, test_env, preprocessor.job_actions,
                               preprocessor.action_map)

returns = []
best_return = config.BEST_RETURN
best_observation = {}
tolerance_step = config.TOLERANCE_STEP
tolerance_count = 0
total_step = 0
for i_episode in range(config.TRAINING_EPISODE):
    # Initialize the environment and state
    state = schedule_env.reset()
    for t in count():
        # Select and perform an action
        action = schedule_agent.select_action(state, schedule_env.job_info)
        total_step += 1
        _, reward, done = schedule_env.step(action.item())
        reward = torch.tensor([reward], device=device)

        next_state = schedule_env.get_current_state()
        schedule_agent.memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = schedule_agent.optimize_model()
        if total_step % config.LOG_INTERVAL == 0:
            loss = loss.item() if loss else 'nan'
            print("step {0} --> loss : {1} ".format(str(total_step), str(loss)))
        if total_step % config.EVAL_INTERVAL == 0:
            avg_return, observation = schedule_agent.test_agent(len(preprocessor.job_info_dict),
                                                                preprocessor.equipments_list, False, num_episodes=2)
            schedule_kpi(observation, preprocessor.equipments_list)
            if best_return < avg_return:
                best_return = avg_return
                tolerance_count = 0
                # Store good results
                schedule_agent.save_best_observation(preprocessor.equipments_list, preprocessor.job_info_dict)
                schedule_kpi(schedule_agent.best_observation, preprocessor.equipments_list)
                # Save model
                torch.save({'model_state_dict': schedule_agent.policy_net.state_dict(),
                            'class_name': schedule_agent.policy_net.__class__.__name__
                            }, config.MODEL_FILE, _use_new_zipfile_serialization=False)

            else:
                tolerance_count += 1
                if tolerance_count == tolerance_step:
                    break

            returns.append(avg_return)
            print("step {0} --> avg_return : {1}, best_return --> {2}".format(str(total_step), str(avg_return),
                                                                              str(best_return)))
        if total_step % 5000 == 0:
            # test_agent(agent.policy, test_env)
            schedule_agent.test_agent(len(preprocessor.job_info_dict), preprocessor.equipments_list,
                                      display_status=False, num_episodes=3)

            # Plot
            plt.plot(returns)
            plt.savefig('return.png')
            plt.close()
        if done:
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % config.TARGET_UPDATE_PERIOD == 0:
        schedule_agent.update_target_net()

print('Complete')

finish_schedule_result = schedule_agent.best_observation_array[-1]

# Print KPI score
print("\n\nFinish Schedule Result:")
clear_result(finish_schedule_result, len(preprocessor.job_info_dict), len(preprocessor.equipments_list))
schedule_kpi(schedule_agent.best_observation, preprocessor.equipments_list)
