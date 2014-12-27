import numpy as np
from simulation import Simulation

"""
Stable Orbit about Earth (alpha, vkick)
=======================================
(x, 0)

Moon Hits
=========
(235, 3.023), (30, 3.003)

Interesting Paths
=================
(240, 3.0263)
"""

# Initial angular position of the lander in orbit about Earth
alpha = np.radians(30.)
# Initial velocity boost to send the lander toward the Moon.
# Units are in terms of the Earth-Moon distance.
vkick = 3.003
MAX_ITERATIONS = int(1E+5)

s = Simulation(alpha, vkick, MAX_ITERATIONS, "lander_sim")
s.run()
s.write_data("lander_sim_data.txt")
s.plot_system()
s.plot_lander_moon_distance()

