import os
import time


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


def create_dir(dir_path):
    try:
        os.mkdir(dir_path)
        print(f"Directory {dir_path} created ")
    except FileExistsError:
        print(f"Directory {dir_path} already exists")