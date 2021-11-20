from dataset import get_dataset
from model import kmeans


if __name__ == '__main__':
    dataset = get_dataset('Data for Problem 2/seeds.txt')
    ks = [3, 5, 7]
    for k in ks:
        sses = []
        for i in range(10):
            sse = kmeans(dataset, k)
            sses.append(sse)
        print(sum(sses)/10)