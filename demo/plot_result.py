import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5)
plt.xticks(x, ['1/4', '1/2', 1, 2, 4])

time = [84.32, 87.45, 88.38, 88.89]
plt.plot([2, 3, 3.5, 4], time, 'g^-', label='time step')

width = [76.62, 81.43, 84.32, 86.22, 88.47]
plt.plot([0, 1, 2, 3, 4], width, 'y>-', label='width')

depth = [82.75, 84.32, 89.23]
plt.plot([1, 2, 3], depth, 'bs-', label='depth')

plt.xlabel('computation consumption')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('./computation-consumption.svg')
plt.show()
