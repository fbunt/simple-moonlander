import numpy as np
import matplotlib.pyplot as plt

from localsystem import LocalSystem
from simplot import Plotter
from moonlandermod import MoonCollisionException, EarthCollisionException, \
    OutputWriter


class Simulation(object):
    """The 'main' class for a simulation. This class carries out the top level
    logic for a simulation.
    """

    def __init__(self, alpha, v0, iterations, sim_name):
        self.sim_name = sim_name
        self.alpha = alpha
        self.v0 = v0
        self.max_iterations = iterations
        self.system = LocalSystem(alpha, v0, iterations)
        self.plotter = Plotter(sim_name)
        scale = self.system.get_spacial_scale()
        self.plotter.set_planet_radii(self.system.earth.radius/scale,
                                      self.system.moon.radius/scale)
        self.dt = 1.00E-04
        self.tlog = np.array([i*self.dt for i in range(iterations+1)])
        self.step = 0

    def run(self):
        try:
            for i in xrange(self.max_iterations):
                self.system.step(self.tlog[self.step], self.dt)
                self.step += 1

        except MoonCollisionException as e:
            #TODO
            print "\n**** hit moon ****"
            # Increment step since an exception was thrown
            self.step += 2
        except EarthCollisionException:
            print "\n**** hit Earth ****"
            # Increment step since an exception was thrown
            self.step += 2
        finally:
            self.plotter.add_logs(self.tlog, self.system.get_logs(), self.step)
            print "\nstep =", self.step
            print "\nfinished"

    def write_data(self, outfilename):
        writer = OutputWriter(outfilename, self.max_iterations,
                              self.system.get_body_names())
        writer.writeheader(self.alpha, self.v0,
                           self.system.get_spacial_scale(),
                           self.system.get_temporal_scale())
        logs = [self.tlog]
        for logpair in self.system.get_logs():
            logs.extend(logpair)
        for i in xrange(self.step):
            vals = [i] + [log[i] for log in logs]
            writer.writedataline(vals)
        writer.close()


    def plot_system(self, show=True, save=True, plot_name=None):
        self.plotter.plot_system(show, save, plot_name)

    def plot_lander_moon_distance(self, show=True, save=True, plot_name=None):
        self.plotter.plot_lander_moon_distance(show, save, plot_name)

