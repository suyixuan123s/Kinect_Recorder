import os


def get_default_data_folder() -> str:
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data')
