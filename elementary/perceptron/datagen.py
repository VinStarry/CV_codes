import numpy as np
import matplotlib.pyplot as plt

'''
@:function data_gen
@:param number : how many samples to create
  create data set (2D-points set) that each for each point P_i(xi, yi),
  0 <= xi <= 1, 0 <= yi <= 1
  if xi + yi < 1 then the tag of P_i is 0
  otherwise is 1 
@:return: none
@:raise
'''
def data_gen(number):
    # generate random arrays, each element is a float number between 0 and 1
    x_array = np.random.rand(number)
    y_array = np.random.rand(number)

    color_array = np.zeros(number, dtype = np.int)

    # tag the points (use x + y = 1 as the boundary)
    for i in range(number):
        if (x_array[i] + y_array[i] < 1):
            color_array[i] = 0
        else:
            color_array[i] = 1

    # write data into a file
    fp = open("raw_data.txt", "w")
    for i in range(number):
        fp.write(str(round(x_array[i], 2)) + "," + str(round(y_array[i], 2)) + "," + str(color_array[i]) + "\n")
    fp.close()

    # visualize the data
    plt.scatter(x_array, y_array, c = color_array)
    plt.show()

    return

if __name__ == '__main__':
    data_gen(30)