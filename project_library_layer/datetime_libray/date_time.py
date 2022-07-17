from datetime import datetime
from dateutil.parser import parse

def get_time():
    """

    :return current time:
    """
    return datetime.now().strftime("%H:%M:%S").__str__()

def get_date():
    """

    :return current date:
    """
    return datetime.now().date().__str__()

def get_difference_in_second(future_date_time:str,past_date_time:str):
    """

    :param future_date:
    :param past_date:
    :return:
    """
    future_date = parse(future_date_time)
    past_date = parse(past_date_time)
    difference = (future_date - past_date)
    total_seconds = difference.total_seconds()
    return total_seconds

def is_future_date(date_time:str,current_date:str):
    date_time = parse(date_time)
    current_date = parse(current_date)
    difference = (date_time - current_date)
    total_seconds = difference.total_seconds()
    if total_seconds>0:
        return True
    else:
        return False


def get_difference_in_milisecond(future_date_time:str,past_date_time:str):
    """

    :param future_date:
    :param past_date:
    :return:
    """
    total_seconds = get_difference_in_second(future_date_time,past_date_time)
    total_milisecond=total_seconds*1000
    return total_milisecond

