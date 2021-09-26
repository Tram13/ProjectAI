from functools import reduce
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

size = 0.3

# Data to plot
inner_labels = 'Bestuursrecht', 'Ambtenarenrecht', 'Belastingrecht', 'Omgevingsrecht', 'Socialezekerheidsrecht',\
               'Vreemdelingenrecht', 'Civiel recht', 'Arbeidsrecht', 'Burgerlijk procesrecht', 'Insolventierecht',\
               'Ondernemingsrecht', 'Personen- en familierecht', 'Verbintenissenrecht', 'Strafrecht'

outer_labels = 'Bestuursrecht', 'Civiel Recht', 'Strafrecht'

sizes = np.array([[72993, 4432, 28482, 6421, 26360, 14732], [72185, 3297, 1413, 2356, 1105, 14167, 3609], [59749]],
                 dtype=object)

outer_colors = ['#8c8ab7', '#e67e00', '#81B814', ]
inner_colors = ['#dfdef5', '#cfcdef', '#bfbdea', '#b7b4e8', '#aface5', '#9e9bce',
                '#ff9819', '#ffa333', '#ffaf4d', '#ffba66', '#ffc680', '#ffd199', '#ffddb3',
                '#81B814']
sum_sizes = np.array([sum(sub) for sub in sizes])
flat_sizes = reduce(lambda x, y: x + y, sizes)

ax.pie(sum_sizes, radius=1, colors=outer_colors, labels=outer_labels,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(flat_sizes, radius=1 - size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.set(aspect="equal")
# plt.show()

plt.savefig('distribution.png')
print("saved distribution.png")
