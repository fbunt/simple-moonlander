import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

class _BodyLog(object):
    def __init__(self, logpair, pos):
        self._x, self._y = logpair
        self.pos = pos

    def x(self, full=0):
        if full:
            return self._x
        else:
            return self._x[:self.pos]

    def y(self, full=0):
        if full:
            return self._y
        else:
            return self._y[:self.pos]

    def xf(self):
        return self._x[self.pos-1]

    def yf(self):
        return self._y[self.pos-1]


class Plotter(object):
    """A class for plotting simulation output logs
    """
    def __init__(self, base_plot_name):
        self.base_plot_name = base_plot_name
        self.pos = 0

    def set_planet_radii(self, rearth, rmoon):
        self.r_earth = rearth
        self.r_moon = rmoon

    def add_logs(self, tlog, bodylogs, pos):
        self.tlog = tlog
        self.lander = _BodyLog(bodylogs[0], pos)
        self.earth = _BodyLog(bodylogs[1], pos)
        self.moon = _BodyLog(bodylogs[2], pos)
        self.pos = pos

    def plot_system(self, show=True, save=True, plot_name=None):
        plt.figure()
        pearth = plt.Circle((self.earth.xf(), self.earth.yf()),
                            self.r_earth, ec="b", fc="w",
                            ls="dashed")
        plt.gca().add_artist(pearth)
        pmoon = plt.Circle((self.moon.xf(), self.moon.yf()),
                           self.r_moon, ec="k", fc="w", ls="dashed")
        plt.gca().add_artist(pmoon)
        # Center of mass
        plt.plot([0],[0], "g+")
        plt.plot(self.earth.x(), self.earth.y(), "b-", label="Earth")
        plt.plot(self.moon.x(), self.moon.y(), "r-", label="Moon")
        plt.plot(self.lander.x(), self.lander.y(), "m-", label="Lander")
        plt.plot(self.earth.xf(), self.earth.yf(), "bo")
        plt.plot(self.moon.xf(), self.moon.yf(), "ro")

        plt.xlabel("x (Moon-Earth Distance")
        plt.ylabel("y (Moon-Earth Distance)")
        plt.title("Earth-Moon-Satellite System")
        plt.legend(loc=0)
        plt.axis("scaled")
        if save:
            if not plot_name:
                plot_name = self.base_plot_name + "_system_plot.pdf"
            else:
                if not plot_name.endswith(".pdf"):
                    plot_name += ".pdf"
            pp = PdfPages(plot_name)
            pp.savefig()
            pp.close()
        if show:
            plt.show()
        plt.close("all")

    def plot_lander_moon_distance(self, show=True, save=True, plot_name=None):
        rlm = np.sqrt((self.lander.x() - self.moon.x())**2
                + (self.lander.y() - self.moon.y())**2)
        plt.figure()
        plt.plot(self.tlog[:self.pos], rlm, 'b-')
        plt.xlabel("t (T)")
        plt.ylabel("Moon-Lander Distance")
        plt.title("Distance to the Moon")
        if save:
            if not plot_name:
                plot_name = self.base_plot_name + "_moon_dist_plot.pdf"
            else:
                if not plot_name.endswith(".pdf"):
                    plot_name += ".pdf"
            pp = PdfPages(plot_name)
            pp.savefig()
            pp.close()
        if show:
            plt.show()
        plt.close("all")

