import numpy as np
from numpy import sqrt, cos, sin, power
from scipy.constants import G

from rk4 import rk4
from moonlandermod import MoonCollisionException, EarthCollisionException, \
    Planet, Body, EARTH_MOON_DISTANCE, EARTH_MASS, EARTH_RADIUS, MOON_MASS, \
    MOON_RADIUS, LANDER_MASS, EARTH_LANDER_INIT_DISTANCE

class LocalSystem(object):
    """
    Represents the local chunck of the solar system. It performs all physics
    calculations.

    This class contains three bodies: the Earth, Moon, and a lander. Each call
    to `step()` moves these three bodies forward in time by a small amount. At
    the end of each "step", the system is checked for any collisions. If a
    collision is detected, an appropriate collision exception is raised.
    """

    def __init__(self, alpha, vkick, iterations, check_em_collisions=False):
        """
        Initialize Earth, Moon, and Lander bodies and physics variables.
        """
        self.check_earth_moon_collision = check_em_collisions
        # Reduced mass
        mu = (EARTH_MASS * MOON_MASS) / (EARTH_MASS + MOON_MASS)
        # Earth / Moon distances from center of mass (COM)
        r_com_earth = mu / EARTH_MASS
        r_com_moon = mu / MOON_MASS
        # Constants used for force calculations
        self.const_e = r_com_moon
        self.const_m = r_com_earth
        # Natural Spacial scale (earth-moon dist)
        self.L = 3.85000000E+08
        # Natural Temporal scale
        self.T = sqrt(mu * self.L**3 / (G * EARTH_MASS * MOON_MASS))
        # Earth
        x0 = -r_com_earth
        vy0 = -self.T/self.L * sqrt(G * MOON_MASS * r_com_earth/self.L)
        self.earth = Planet("Earth", EARTH_MASS, EARTH_RADIUS, x0, 0, 0, vy0,
                            iterations)
        # Moon
        x0 = r_com_moon
        vy0 = self.T/self.L * sqrt(G * EARTH_MASS * r_com_moon/self.L)
        self.moon = Planet("Moon", MOON_MASS, MOON_RADIUS, x0, 0, 0, vy0,
                           iterations)
        # Lander
        r = EARTH_LANDER_INIT_DISTANCE/self.L
        x0 = self.earth.x + r * cos(alpha)
        y0 = self.earth.y + r * sin(alpha)
        v = self.T/self.L * sqrt(G * EARTH_MASS / (r * self.L))
        vx0 = -(v + vkick) * sin(alpha)
        vy0 = (v + vkick) * cos(alpha)
        self.lander = Body("Lander", LANDER_MASS, x0, y0, vx0, vy0, iterations)

        self.bodies = [self.lander, self.earth, self.moon]

    def step(self, t, dt):
        """
        Advance the system one step in time using fourth-order Runge-Kutta.
        """
        xvec = [self.earth.x, self.earth.y, self.earth.vx, self.earth.vy,
                self.moon.x, self.moon.y, self.moon.vx, self.moon.vy,
                self.lander.x, self.lander.y, self.lander.vx, self.lander.vy]
        dxdt = self._deriv(t, xvec)
        self._unpack_and_advance(rk4(xvec, dxdt, t, dt, self._deriv))
        self._check_for_collisions()

    def get_body_names(self):
        """
        Retuns a string list containing the names of the bodies.
        """
        return [b.name for b in self.bodies]

    def get_logs(self):
        """
        Returns a list of tuples where each tuple is the x and y postion logs
        for a body.
        """
        return [(b.xlog, b.ylog) for b in self.bodies]

    def get_spacial_scale(self):
        """
        Returns the spacial scale used for the physics calculations.
        """
        return self.L

    def get_temporal_scale(self):
        """
        Returns the temporal scale used for the physics calculations.
        """
        return self.T

    def _deriv(self, t, xvec):
        """
               [0    1     2     3    4    5     6     7    8    9     10    11]
        xvec = [xe,  ye,  vxe,  vye,  xm,  ym,  vxm,  vym,  xl,  yl,  vxl,  vyl]
        """
        rem3 = power((xvec[0]-xvec[4])*(xvec[0]-xvec[4])
                     + (xvec[1]-xvec[5])*(xvec[1]-xvec[5]),
                     1.5)
        rel3 = power((xvec[8]-xvec[0])*(xvec[8]-xvec[0])
                    + (xvec[9]-xvec[1])*(xvec[9]-xvec[1]),
                     1.5)
        rml3 = power((xvec[8]-xvec[4])*(xvec[8]-xvec[4])
                     + (xvec[9]-xvec[5])*(xvec[9]-xvec[5]),
                     1.5)
        dxdt = np.zeros(12)
        dxdt[0] = xvec[2]         # dxe/dt
        dxdt[1] = xvec[3]         # dye/dt
        dxdt[2] = - xvec[0]/rem3  # dvxe/dt
        dxdt[3] = - xvec[1]/rem3  # dvye/dt
        dxdt[4] = xvec[6]         # dxm/dt
        dxdt[5] = xvec[7]         # dym/dt
        dxdt[6] = - xvec[4]/rem3  # dvxm/dt
        dxdt[7] = - xvec[5]/rem3  # dvym/dt
        dxdt[8] = xvec[10]        # dxl/dt
        dxdt[9] = xvec[11]        # dyl/dt
        # dvxl/dt
        dxdt[10] = -self.const_e*(xvec[8]-xvec[0])/rel3 \
                  - self.const_m*(xvec[8]-xvec[4])/rml3
        # dvyl/dt
        dxdt[11] = -self.const_e*(xvec[9]-xvec[1])/rel3 \
                  - self.const_m*(xvec[9]-xvec[5])/rml3
        return dxdt

    def _unpack_and_advance(self, data):
        vals, data = data[:4], data[4:]
        self.earth.step(*vals)
        vals, data = data[:4], data[4:]
        self.moon.step(*vals)
        vals, data = data[:4], data[4:]
        self.lander.step(*vals)

    def _check_for_collisions(self):
        x = self.lander.x
        y = self.lander.y
        xe = self.earth.x
        ye = self.earth.y
        xm = self.moon.x
        ym = self.moon.y
        rel = sqrt( (xe - x)*(xe - x) + (ye - y)*(ye - y) )
        if rel*self.L <= EARTH_RADIUS:
            raise EarthCollisionException("Hit the Earth")
        rml = sqrt( (xm - x)*(xm - x) + (ym - y)*(ym - y) )
        if rml*self.L <= MOON_RADIUS:
            raise MoonCollisionException("Hit the moon")
        if self.check_earth_moon_collision:
            rem = sqrt( (xe - xm)*(xe - xm) + (ye - ym)*(ye - ym) )
            if rem <= (EARTH_RADIUS + MOON_RADIUS):
                raise EarthMoonCollisionException("Earth bumped Moon")


