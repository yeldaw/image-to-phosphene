import numpy as np
import matplotlib.pyplot as plt


def create_heatmap(x_loc, y_loc, x_size, y_size, num=5):
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(0, num, x_size), np.linspace(-num, num, y_size))

    x = x + (x_size/2-x_loc)/(x_size/2) * num
    y = y + (y_size/2-y_loc)/(y_size/2) * num


    # print(-(x_size-x_loc)/x_size)
    # print(-(y_size-y_loc)/y_size)

    dst = np.sqrt(x * x + y * y)
    # Intializing sigma and muu
    sigma = 1
    muu = 0.000

    # Calculating Gaussian array
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
    plt.imshow(gauss)
    plt.show()
