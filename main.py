import numpy as np

from simulation import Simulation

# Initial angular position of the lander in orbit about Earth
ALPHA = np.radians(235.)
# Initial velocity boos to send the lander toward the Moon
V_KICK = 3.023
MAX_ITERATIONS = int(1E+5)

s = Simulation(ALPHA, V_KICK, MAX_ITERATIONS, "lander_sim")
s.run()
s.write_data("lander_sim_data.txt")
s.plot_system()
s.plot_lander_moon_distance()

