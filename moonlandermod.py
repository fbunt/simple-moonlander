import numpy as np
from numpy import sqrt, cos, sin, radians, pi, degrees, power
import matplotlib.pyplot as plt
from scipy.constants import G
import warnings


EARTH_MOON_DISTANCE = 3.85000000E+08
EARTH_MASS = 5.97219E+24
EARTH_RADIUS = 6.371E+06
MOON_MASS = 7.34767E+22
MOON_RADIUS = 1.7375E+06
LANDER_MASS = 1000
EARTH_LANDER_INIT_DISTANCE = EARTH_RADIUS + 4.0E+05


class MoonCollisionException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class EarthCollisionException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Body(object):
    """A base class representing any body moving about the local piece
    of the solar system.
    """
    def __init__(self, name, mass, x0, y0, vx0, vy0, iterations):
        self.name = name
        self.mass = mass
        self.x = x0
        self.y = y0
        self.vx = vx0
        self.vy = vy0
        self.xlog = np.zeros(iterations+1)
        self.ylog = np.zeros(iterations+1)
        self.vxlog = np.zeros(iterations+1)
        self.vylog = np.zeros(iterations+1)
        self.logpos = 0
        self.xlog[self.logpos] = self.x
        self.ylog[self.logpos] = self.y
        self.vxlog[self.logpos] = self.vx
        self.vylog[self.logpos] = self.vy
        self.logpos += 1

        self.xforce = 0.
        self.yforce = 0.

    def addforce(self, fx, fy):
        self.xforce += fx
        self.yforce += fy

    def step(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.xlog[self.logpos] = x
        self.ylog[self.logpos] = y
        self.vxlog[self.logpos] = vx
        self.vylog[self.logpos] = vy
        self.logpos += 1


class Planet(Body):
    """A class representing a planet-like body such as Earth or the
    Moon.
    """
    def __init__(self, name, mass, radius, x0, y0, vx0, vy0, iterations):
        super(Planet, self).__init__(name, mass, x0, y0, vx0, vy0, iterations)
        self.radius = radius


class _OutputFormatter(object):
    """A class that creates formatted lines of data to be written to a
    file. The data is grouped such that each line will have a column
    to itself and the columns will be aligned with the values above
    and below. This class also can provide a correctly aligned column
    title line and a divider.
    """

    space = " "
    min_step_column_width = 9
    min_time_column_width = 11
    min_body_column_width = 20

    def __init__(self, maxiters, body_names):
        self.step_col_title = "# step"
        self.time_col_title = "time elap"
        self.body_col_titles = []
        self.body_col_width = self.min_body_column_width
        self.net_width = 0

        self.step_col_width = len(str(maxiters))
        if self.step_col_width < self.min_step_column_width:
            self.step_col_width = self.min_step_column_width
        self.time_col_width = self.min_time_column_width
        # sanitize titles
        self.body_col_titles = self._get_body_col_titles(body_names)
        # expand body column width to fit titles if necessary
        for s in self.body_col_titles:
            if len(s) > self.min_body_column_width:
                self.body_col_width = len(s) + 2
        # calculate net width of all columns for divider
        self.net_width = self.step_col_width + self.time_col_width \
                + self.body_col_width*len(self.body_col_titles)

    def get_column_labels(self):
        label = self.step_col_title \
                + self.space*(self.step_col_width-len(self.step_col_title))
        label += self.time_col_title \
                + self.space*(self.time_col_width-len(self.time_col_title))
        for s in self.body_col_titles:
            label += s + self.space*(self.body_col_width-len(s))
        return label

    def get_header_divider(self):
        div = "#" + "-"*(self.net_width-1)
        return div

    def construct_output_line(self, step, t, *bodyvals):
        if len(bodyvals) != len(self.body_col_titles):
            warnings.warn("Number of values supplied did not match the " +
                    "number expected.", RuntimeWarning)
        line = self._get_step_col(step)
        line += self._get_time_col(t)
        line += self._get_body_cols(bodyvals)
        return line

    def _get_body_col_titles(self, bodies):
        titles = []
        for s in bodies:
            s = s.strip()
            titles.append(s + " x")
            titles.append(s + " y")
        return titles

    def _get_step_col(self, step):
        return self._get_formatted_col(step, self.step_col_width)

    def _get_time_col(self, t):
        return self._get_formatted_col(t, self.time_col_width)

    def _get_body_cols(self, bodyvals):
        cols = ''
        for v in bodyvals:
            cols += self._get_body_col(v)
        return cols

    def _get_body_col(self, bval):
        return self._get_formatted_col(bval, self.body_col_width)

    def _get_formatted_col(self, val, width):
        return str(val) + self.space*(width-len(str(val)))


class OutputWriter(object):
    def __init__(self, filename, maxiterations, body_names):
        self.outfile = open(filename, 'w')
        self.data_formatter = _OutputFormatter(maxiterations, body_names)

    def write(self, string):
        self.outfile.write(string)

    def writeln(self, string=''):
        self.write(string + '\n')

    def writeheader(self, alpha, v0, lenscale, tscale):
        self.writeln('# alpha = ' + str(np.degrees(alpha)) + ' degrees')
        self.writeln('# raw v0 = ' + str(v0))
        self.writeln('# v0 = ' + str(lenscale/tscale*v0) + ' m/s')
        self.writeln('# Length Scale = ' + str(lenscale) + ' m')
        self.writeln('# Time Scale = ' + str(tscale) + ' s')
        self.writeln('#')
        self.writeln(self.data_formatter.get_column_labels())
        self.writeln(self.data_formatter.get_header_divider())

    def writedataline(self, datavalues):
        (step, t), datavalues = datavalues[:2], datavalues[2:]
        self.writeln(
                self.data_formatter.construct_output_line(step, t, *datavalues))

    def close(self):
        #self.outfile.write('end')
        self.outfile.close()

