import time

global _start_time

def tic():
    global _start_time
    _start_time = time.time()

def tac():
    global _start_time
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))

def timer():
    global _start_time
    t_sec = round(time.time() - _start_time)
    return t_sec
