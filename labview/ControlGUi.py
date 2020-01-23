import tkinter as tk
import numpy as np

master = tk.Tk()
master.geometry("1000x500+300+300")

position = 500

example_voltage = np.array([0, 0, 0, 0, -4.89323688, 4.87633803, 4.21009563, 0.54137305, -2.56616087, -5.90619072, 3.383148, -0.24544995, 2.68213093, -1.14521202, 2.26031778, -1.56279098, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055,  1.92823141, -1.91687055, 1.92823141, -1.91687055,
                            1.92823141, -1.91687055, 1.92823141, -1.91687055,  1.92823141, -1.91687055])

tk.Label(master, text="Current Position: ").grid(row=1)
tk.Label(master, text="Shuttle ion to this position: ").grid(row=0)
tk.Label(master, text="Currently Supplied Voltages: ").grid(row=2)

button_goto = tk.Button(master, text="Go", command=None)
button_goto.grid(row=0, column=2, padx=20)


shuttle_position = tk.Entry(master)
current_position = tk.Label(master)

current_position.grid(row=1, column=1, pady=20)
shuttle_position.grid(row=0, column=1)

current_position.config(text=position, font=24)
shuttle_position.config(font=24)

Channels = tk.Label()
Channels.config(text=example_voltage)
Channels.grid(row=3, column=0)


master.mainloop()
