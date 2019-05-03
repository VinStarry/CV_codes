import numpy as np
import matplotlib.pyplot as plt

x_array = np.zeros(30, dtype = float)
y_array = np.zeros(30, dtype = float)
color_array = np.zeros(30, dtype = int)

'''
@:function parse_data_set
@:param filepath: the path of data_set.txt
  parse data from filepath, and add information to global variables
  like x_array, y_array and color_array in order to visualization
@:return datalist: the data parsed from data_set.txt, the data list
  is used for training later
'''
def parse_data_set(filepath):
    data_list = []
    fp = open(filepath, 'r')
    index = 0

    for lines in fp:
        data = lines.strip().strip('\n').split(',')
        tag = 0
        if int(data[-1]) == 0:
            tag = -1
        else:
            tag = 1
        temp_tup = (np.array([float(data[0]), float(data[1])]), tag)
        x_array[index] = data[0]
        y_array[index] = data[1]
        if (x_array[index] + y_array[index] < 1):
            color_array[index] = 0
        else:
            color_array[index] = 1
        data_list.append(temp_tup)
        index += 1

    fp.close()
    return data_list

'''
@:function train_perceptron
@:param data_list: the data parsed from input file
@:eta: learning rate
  Use the algorithm of perceptron to calculate the boundary
@:returns steps: how many steps to calculate the boundary
          bias: the bias of the boundary
          weight: the weight of the boundary
'''
def train_perceptron(data_list, eta = 1.0):
    weight = np.array([0, 0])
    bias = 0.0
    steps = 0

    flag = True
    while flag:
        misclassified_count = 0
        for data in data_list:
            x = data[0]
            y = data[1]
            if ((np.dot(x, weight)+ bias)  * y <= 0.0):
                weight = weight + eta * x * y
                bias = bias + eta * y
                misclassified_count += 1
                steps += 1
                visualize_data(data_list, weight[0], weight[1], bias, steps)
        if misclassified_count == 0:
            flag = False
    return steps, bias, weight

'''
@:function visualize_data
@:param data_list: the data parsed from input file
@:param a: a in 'ax + by + c = 0'
@:param b: b in 'ax + by + c = 0'
@:param c: c in 'ax + by + c = 0'
@:param image_no: the number of thses images
  visualize the data and calculation results, including
  points in the data set and their labels, the current
  boundary calculated, AKA 'ax + by + c = 0', and save the picture
@:return none 
'''
def visualize_data(data_list, a, b, c, image_no = 0):
    x = np.linspace(0, 1)
    if (b == 0):
        return
    y = (-a / b) * x - (c / b)
    plt.plot(x, y, '-r', label = 'boundary')
    plt.scatter(x_array, y_array, c = color_array)
    plt.savefig('perceptron_result_im_' + str(image_no) + '.png')
    plt.show()
    return


if __name__ == '__main__':
    dlist = parse_data_set(filepath = './raw_data.txt')
    steps, bias, weight = train_perceptron(dlist, eta = 1.0)

    print('steps = ' + str(steps))
    print('weight: ', end=' ')
    print(weight)
    print('bias: ', end=' ')
    print(bias)
