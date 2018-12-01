import time

def display_formatted_time(elapsed_time,  msg=""):
    minutes,  seconds = map(int,  divmod(elapsed_time,  60))
    print("Elapsed time - {0}: {1}min {2}s".format(msg,  minutes,  seconds))

def measure(func):
    """
    @measure
    def test():
        for i in range(1000000000):
            pass

    if __name__ == '__main__':
        print("start")
        test()
        print("end")
    """

    def wrapper(*args, **kwargs):
        def display_formatted_time(elapsed_time, msg=""):
            minutes, seconds = map(int, divmod(elapsed_time, 60));
            print("Elapsed time - {0}: {1}min {2}s".format(msg, minutes, seconds))

        since = time.time()
        func(*args, **kwargs)
        display_formatted_time(time.time() - since, func.__name__)

    return wrapper
