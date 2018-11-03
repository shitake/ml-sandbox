
def display_formatted_time(elapsed_time,  msg=""):
    minutes,  seconds = map(int,  divmod(elapsed_time,  60))
    print("Elapsed time - {0}: {1}min {2}s".format(msg,  minutes,  seconds))
