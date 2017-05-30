def rk4(x, dxdt, t, dt, derivative):
    """
    An implementation of fourth-order Runge-Kutta algorithm.

    Author: Dr. Andrew Ware of the University of Montana
    """
    th = t + 0.50*dt

    k1 = dt * dxdt                      # First step
    xh = x + 0.50 * k1

    dxdth = derivative(th, xh)

    k2 = dt * dxdth                     # Second step
    xh = x + 0.50 * k2

    dxdth = derivative(th, xh)

    k3 = dt * dxdth                     # Third step
    xh = x + k3

    dxdth = derivative(t + dt, xh)

    k4 = dt * dxdth                     # Fourth step

    xout = x + ( k1 + 2.0 * k2 + 2.0 * k3 + k4 )/6.0

    return xout
