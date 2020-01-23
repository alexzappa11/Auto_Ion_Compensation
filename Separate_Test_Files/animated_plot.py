import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time as time

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(n):
    x = []
    y = []
    for i in range(0, 100):
        x.append(i)
        y.append(i ^ 2)

    ax1.clear()
    ax1.plot(x, y)


ani = animation.FuncAnimation(fig, animate, interval=500)
plt.show()
