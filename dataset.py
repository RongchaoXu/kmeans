import numpy as np


def get_dataset(file_path):
    return np.loadtxt(file_path)


if __name__ == '__main__':
    d = get_dataset('./Data for Problem 2/seeds.txt')
    print(np.isinf(d).any())