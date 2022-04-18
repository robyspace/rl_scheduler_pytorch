from datetime import datetime, timedelta
from time import strptime
import config

def datetime_to_timedelta(timestamp):
    time_dif = datetime.strptime(timestamp, "%Y-%m-%d %H:%M") - datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"], "%Y-%m-%d")
    last_free_time = None
    free_time_total = 0
    for i in range(time_dif.days + 1):
        date = datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"], "%Y-%m-%d") + timedelta(days=i)
        weekday = date.strftime("%A")
        working_times = config.WORKING_TIME_DATA["workingTimes"][weekday]
        for working_time in working_times:
            # 每天的工作時間，有可能會被拆成兩段
            start_time, end_time = working_time["startTime"], working_time["endTime"]
            start_working_time = datetime.strptime(date.strftime("%Y-%m-%d") + " " + start_time, "%Y-%m-%d %H:%M")
            if last_free_time:
                # 過程中的 free time 在這計算
                free_time_total += (start_working_time - last_free_time).total_seconds() / 60
                # last_free_time = pd.to_datetime(date + pd.Timedelta(hours=int(end_time.split(":")[0]), minutes=int(end_time.split(":")[1])))
                last_free_time = date + timedelta(datetime.strptime(end_time, "%H:%M").hour) + timedelta(datetime.strptime(end_time, "%H:%M").second)
            else:
                # 第一天的 free time 在這裡計算
                free_time_total += (start_working_time - datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"], "%Y-%m-%d")).total_seconds() / 60
                last_free_time = datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"] + " " + end_time, "%Y-%m-%d %H:%M")
            if datetime.strptime(timestamp, "%Y-%m-%d %H:%M") < last_free_time:
                # 時間已經到了start time 和 end time ，代表 free time 計算完成準備轉換
                break
    minutestamp = (time_dif.total_seconds() / 60 - free_time_total)
    return minutestamp


def timedelta_to_datetime(minute_stamp):
    last_free_time = None
    for i in range(config.WORKING_TIME_DATA["scheduleDays"]):
        date = datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"], "%Y-%m-%d") + timedelta(days=i)
        weekday = date.strftime("%A")
        working_times = config.WORKING_TIME_DATA["workingTimes"][weekday]
        for working_time in working_times:
            start_time, end_time = working_time["startTime"], working_time["endTime"]
            start_working_time = datetime.strptime(date.strftime("%Y-%m-%d") + " " + start_time, "%Y-%m-%d %H:%M")
            if last_free_time:
                # 有經過休息時間就需要加上去
                minute_stamp += ((start_working_time - last_free_time).total_seconds() / 60)
                last_free_time = date + timedelta(datetime.strptime(end_time, "%H:%M").hour) + timedelta(datetime.strptime(end_time, "%H:%M").second)
            else:
                # 有經過休息時間就需要加上去
                minute_stamp += ((start_working_time - datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"], "%Y-%m-%d")).total_seconds() / 60)
                last_free_time = datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"] + " " + end_time, "%Y-%m-%d %H:%M")
            # 確認該時間是否已經達到目標日期
            current_timestamp = datetime.strptime(config.WORKING_TIME_DATA["scheduleStartDate"], "%Y-%m-%d") + timedelta(minutes=minute_stamp)
            if current_timestamp <= last_free_time:
                return current_timestamp
