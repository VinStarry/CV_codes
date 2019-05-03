import numpy as np
import matplotlib.pyplot as plt

x_array = np.zeros(30, dtype = float)
y_array = np.zeros(30, dtype = float)
color_array = np.zeros(30, dtype = int)

def parse_data_set(filepath):
    fp = open(filepath, 'r')
    index = 0

    for lines in fp:
        data = lines.strip().strip('\n').split(',')
        x_array[index] = data[0]
        y_array[index] = data[1]
        if (x_array[index] + y_array[index] < 1):
            color_array[index] = 0
        else:
            color_array[index] = 1
        index += 1

def visualization():
    x = np.linspace(0, 1)
    y = -x + 1
    plt.plot(x, y, '-r', label = 'y=-x+1')
    plt.scatter(x_array, y_array, c = color_array)
    plt.savefig('perceptron_sample' + '.png')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    parse_data_set(filepath = './raw_data.txt')
    visualization()
