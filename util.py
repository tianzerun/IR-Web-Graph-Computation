import os
import time
import uuid
import pickle


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


def create_dir(dir_path):
    try:
        os.mkdir(dir_path)
        print(f"Directory {dir_path} created ")
    except FileExistsError:
        print(f"Directory {dir_path} already exists")


def hash_id(url):
    return str(uuid.uuid3(uuid.NAMESPACE_URL, url))


def write_pickle(path, data):
    with path.open("wb") as fp:
        pickle.dump(data, fp)


def load_pickle(path):
    with path.open("rb") as fp:
        return pickle.load(fp)
