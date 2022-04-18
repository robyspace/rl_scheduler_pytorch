# User Input
START_TIME = "2020-05-14"
SCHEDULE_DAYS = 30
KPI_SCORE_WEIGHT = {"capacity": 3, "ofr": 2, "cycle_time": 1, "utilization": 4}

# RL data parameter
SETUP_TIME_PATH = "data/setup_time.csv"
ORDER_PATH = "data/orders.csv"
PROCESS_TIME_PATH = "data/process_time.csv"
GROUP_PATH = "data/group.csv"
EQUIPMENTS_PATH = "data/equipment.csv"
WORKING_TIME_PATH = "data/working_time.csv"

# RL module parameter
REPLAY_BUFFER_CAPACITY = 20000
# EPSILON_GREEDY
EPS_END = 0.05
EPS_START = 0.9
EPS_DECAY = 50

REWARD_SCALE_FACTOR = 1.0  # default
# GRADIENT_CLIPPING = 1.0
# TARGET_UPDATE_TAU = 0.1
TARGET_UPDATE_PERIOD = 7
DISCOUNT = 0.95

# DL parameter
LEARNING_RATE = 0.0006
BATCH_SIZE = 64
TRAINING_EPISODE = 20000
LOG_INTERVAL = 200
NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 1000
MODEL_FILE = './policy_set/policy'
# Early stopping
BEST_RETURN = -1e10
TOLERANCE_STEP = 10

LOAD_MODEL = False
IS_RETRAIN = False
LIGHT_SEARCH = True
OUTPUT_PATH = "output"
# working_time data
WORKING_TIME_DATA = {
    "workingTimes": {
        "Sunday": [
            {
                "startTime": "00:00",
                "endTime": "23:59"
            }
        ],
        "Monday": [
            {
                "startTime": "00:00",
                "endTime": "23:59"
            }
        ],
        "Tuesday": [
            {
                "startTime": "00:00",
                "endTime": "23:59"
            }
        ],
        "Wednesday": [
            {
                "startTime": "00:00",
                "endTime": "23:59"
            }
        ],
        "Thursday": [
            {
                "startTime": "00:00",
                "endTime": "23:59"
            }
        ],
        "Friday": [
            {
                "startTime": "00:00",
                "endTime": "23:59"
            }
        ],
        "Saturday": [
            {
                "startTime": "00:00",
                "endTime": "23:59"
            }
        ]
    },
    "holidays": [
        "2021-07-10",
        "2021-07-11"
    ]
}
