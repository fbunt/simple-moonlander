from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt

from localsystem import LocalSystem
from simplot import Plotter
from moonlandermod import MoonCollisionException, EarthCollisionException, \
    EarthMoonCollisionException, OutputWriter


class Simulation(object):
    """
    The 'main' class for a simulation. This class carries out the top level
    logic for the simulation.
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
        self.step = 1

    def run(self):
        """
        Run the simulation.
        """
        try:
            for i in xrange(self.max_iterations):
                self.system.step(self.tlog[self.step], self.dt)

                if i%200 == 0:
                    # Print the step percentage. "\r" keeps everything on the
                    # same line.
                    out = "\r" + str(self.step/self.max_iterations) + "%"
                    sys.stdout.write(out + " "*(10-len(out)))
                    sys.stdout.flush()
                self.step += 1

        except MoonCollisionException:
            print "\n**** hit moon ****"
            # Increment step since an exception was thrown
            self.step += 1
        except EarthCollisionException:
            print "\n**** hit Earth ****"
            # Increment step since an exception was thrown
            self.step += 1
        except EarthMoonCollisionException:
            print "\n**** Earth collided with the Moon ***"
            self.step += 1
        finally:
            self.plotter.add_logs(self.tlog, self.system.get_logs(), self.step)
            print "\nstep =", self.step
            print "\nfinished"

    def write_data(self, outfilename):
        """
        Write the results of a simulation to a file.

        Args:
          outfilename (str):  the name of the file to write data to
        """
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
        """
        Create a plot of the simulation as it was when it ended.

        Args:
          show (bool)[default: True]: if True, plt.show() is called
          save (bool)[default: True]: if True, the plot is saved as a PDF
          plot_name (str)[default: None]: if a str is provided, the plot is
                saved as plot_name + '_system_plot.pdf'
        """
        self.plotter.plot_system(show, save, plot_name)

    def plot_lander_moon_distance(self, show=True, save=True, plot_name=None):
        """
        Create a plot of the Lander-Moon distance vs step count.

        Args:
          show (bool)[default: True]: if True, plt.show() is called
          save (bool)[default: True]: if True, the plot is saved as a PDF
          plot_name (str)[default: None]: if a str is provided, the plot is
                saved as plot_name + '_moon_dist_plot.pdf'
        """
        self.plotter.plot_lander_moon_distance(show, save, plot_name)

