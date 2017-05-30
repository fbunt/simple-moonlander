import numpy as np
from numpy import sqrt, cos, sin, radians, pi, degrees, power
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
    pass

class EarthCollisionException(Exception):
    pass

class EarthMoonCollisionException(Exception):
    pass

class Body(object):
    """
    A base class representing any body moving about the system.

    This class acts as a container rather than implementing its own physics.
    Use `step()` to update the values.

    Attributes:
      name (str): Body's name
      mass (float): Body's mass
      x (float: x position
      y (float): y position
      vx (float): x velocity component
      vy (float): y velocity component
      xlog (np.ndarray of float): array of x position values
      ylog (np.ndarray of float): array of y position values
      vxlog (np.ndarray of float): array of vx position values
      vylog (np.ndarray of float): array of vy position values
    """

    def __init__(self, name, mass, x0, y0, vx0, vy0, iterations):
        self.name = name
        self.mass = mass
        self.x = x0
        self.y = y0
        self.vx = vx0
        self.vy = vy0
        # len = iterations + 1 to account for initial positions
        self.xlog = np.zeros(iterations+1)
        self.ylog = np.zeros(iterations+1)
        self.vxlog = np.zeros(iterations+1)
        self.vylog = np.zeros(iterations+1)
        self._logpos = 0
        self.xlog[self._logpos] = self.x
        self.ylog[self._logpos] = self.y
        self.vxlog[self._logpos] = self.vx
        self.vylog[self._logpos] = self.vy
        self._logpos += 1

    def step(self, x, y, vx, vy):
        """
        Update and record the body's new position and velocity
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.xlog[self._logpos] = x
        self.ylog[self._logpos] = y
        self.vxlog[self._logpos] = vx
        self.vylog[self._logpos] = vy
        self._logpos += 1


class Planet(Body):
    """
    A class representing a planet-like body such as Earth or the Moon.

    This class adds a radius attribute to the `Body` class.

    Attributes:
      radius (float): Planet's radius
    """

    def __init__(self, name, mass, radius, x0, y0, vx0, vy0, iterations):
        super(Planet, self).__init__(name, mass, x0, y0, vx0, vy0, iterations)
        self.radius = radius


class _OutputFormatter(object):
    """
    Formats data values to be written to a file.

    The data is spaced such that values on each line will be aligned in columns
    with the values above and below them. A correctly aligned column title line
    and divider line can also be provided.
    """

    _space = " "
    _min_step_column_width = 9
    _min_time_column_width = 11
    _min_body_column_width = 20

    def __init__(self, maxiters, body_names):
        self._step_col_title = "# step"
        self._time_col_title = "time elap"
        self._body_col_titles = []
        self._body_col_width = self._min_body_column_width
        self._net_width = 0

        self._step_col_width = len(str(maxiters))
        if self._step_col_width < self._min_step_column_width:
            self._step_col_width = self._min_step_column_width
        self._time_col_width = self._min_time_column_width
        # sanitize titles
        self._body_col_titles = self._get_body_col_titles(body_names)
        # expand body column width to fit titles if necessary
        for s in self._body_col_titles:
            if len(s) > self._min_body_column_width:
                self._body_col_width = len(s) + 2
        # calculate net width of all columns for divider
        self._net_width = self._step_col_width + self._time_col_width \
                + self._body_col_width*len(self._body_col_titles)

    def get_column_labels(self):
        """
        Create line containing spaced column titles.
        """
        label = self._step_col_title \
                + self._space*(self._step_col_width-len(self._step_col_title))
        label += self._time_col_title \
                + self._space*(self._time_col_width-len(self._time_col_title))
        for s in self._body_col_titles:
            label += s + self._space*(self._body_col_width-len(s))
        return label

    def get_header_divider(self):
        """
        Create and return a divider line made of dashes
        """
        div = "#" + "-"*(self._net_width-1)
        return div

    def construct_output_line(self, step, t, *bodyvals):
        """
        Create and return a formatted line of data values.

        Args:
          step (int): the iteration number
          t (float): time value
          *bodyvalues (var arg list of float): data values
        """
        if len(bodyvals) != len(self._body_col_titles):
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
        return self._get_formatted_col(step, self._step_col_width)

    def _get_time_col(self, t):
        return self._get_formatted_col(t, self._time_col_width)

    def _get_body_cols(self, bodyvals):
        cols = ''
        for v in bodyvals:
            cols += self._get_body_col(v)
        return cols

    def _get_body_col(self, bval):
        return self._get_formatted_col(bval, self._body_col_width)

    def _get_formatted_col(self, val, width):
        return str(val) + self._space*(width-len(str(val)))


class OutputWriter(object):
    """
    A wrapper around a data file that can be used to write formatted output to
    the file.
    """

    def __init__(self, filename, maxiterations, body_names):
        """
        Args:
          filename (str): name of the data file to open
          maxiterations (int): the max number of simulation iterations expected
          body_names (list of str): names of bodies in the simulation. Used for
                for data column titles.
        """
        self._outfile = open(filename, 'w')
        self._data_formatter = _OutputFormatter(maxiterations, body_names)
        self._closed = False

    def write(self, string):
        """
        Writes a string to the data file but does not append "\n".
        """
        if self._closed:
            raise IOError("Could not perform write. File was closed.")
        self._outfile.write(string)

    def writeln(self, string=''):
        """
        Write a line to the data file. "\n" is appended to the end of the string

        This method can be used to write a blank line (just "\n") to the file.

        Args:
          string (str)[default: empty str]: string to be written
        """
        self.write(string + '\n')

    def writeheader(self, alpha, v0, lenscale, tscale):
        """
        Write a header block to the data file.

        Args:
          alpha (float):  starting angle (in radians) of the lander about Earth
          v0 (float):  initial v boost given to the lander (natural units)
          lenscale (float):  natural length scale for simulation
          tscale (float):  natural temporal scale for simulation
        """
        self.writeln('# alpha = ' + str(np.degrees(alpha)) + ' degrees')
        self.writeln('# raw v0 = ' + str(v0))
        self.writeln('# v0 = ' + str(lenscale/tscale*v0) + ' m/s')
        self.writeln('# Length Scale = ' + str(lenscale) + ' m')
        self.writeln('# Time Scale = ' + str(tscale) + ' s')
        self.writeln('#')
        self.writeln(self._data_formatter.get_column_labels())
        self.writeln(self._data_formatter.get_header_divider())

    def writedataline(self, datavalues):
        """
        Writes a line of data values to a file.

        Args:
          datavalues (list): list of data values to be formatted into a line
        """
        (step, t), datavalues = datavalues[:2], datavalues[2:]
        line = self._data_formatter.construct_output_line(step, t, *datavalues)
        self.writeln(line)

    def close(self):
        """
        Close the underlying file.
        """
        #self.outfile.write('end')
        self._outfile.close()

