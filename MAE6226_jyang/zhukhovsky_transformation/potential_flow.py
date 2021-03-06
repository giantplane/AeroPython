import numpy
from matplotlib import pyplot

def get_velocity_doublet(strength, xd, yd, X, Y):
    """Returns the velocity field generated by a doublet.
       Arguments
       ---------
       strength -- strength of the doublet.
       xd, yd -- coordinates of the doublet.
       X, Y -- mesh grid.
    """
    u = - strength/(2*numpy.pi)*((X-xd)**2-(Y-yd)**2)/((X-xd)**2+(Y-yd)**2)**2
    v = - strength/(2*numpy.pi)*2*(X-xd)*(Y-yd)/((X-xd)**2+(Y-yd)**2)**2
    return u, v

def get_stream_function_doublet(strength, xd, yd, X, Y):
    """Returns the stream-function generated by a doublet.
       Arguments
       ---------
       strength -- strength of the doublet.
       xd, yd -- coordinates of the doublet.
       X, Y -- mesh grid.
    """
    psi = - strength/(2*numpy.pi)*(Y-yd)/((X-xd)**2+(Y-yd)**2)
    return psi

def get_velocity_vortex(strength, xv, yv, X, Y):
    """Returns the velocity field generated by a vortex.
       Arguments
       ---------
       strength -- strength of the vortex.
       xv, yv -- coordinates of the vortex.
       X, Y -- mesh grid.
    """
    u = +strength/(2*numpy.pi)*(Y-yv)/((X-xv)**2+(Y-yv)**2)
    v = -strength/(2*numpy.pi)*(X-xv)/((X-xv)**2+(Y-yv)**2)
    return u, v

def get_stream_function_vortex(strength,xv, yv, X, Y):
    """Returns the stream-function generated by a vortex.
       Arguments
       ---------
       strength -- strength of the vortex. 
       xv, yv -- coordinates of the vortex.
       X, Y -- mesh grid.
    """
    psi = strength/(4*numpy.pi)*numpy.log((X-xv)**2+(Y-yv)**2)
    return psi

def plot_grid(Z, XI,nsub):
    """Z: circle plane grid, XI: airfoil plane grid
       Arguments
       ---------
       Z, XI -- real and imaginary parts are x and y coordinates
       nsub -- number of subplots
    """
    pyplot.figure(figsize=(14, 7))
    pyplot.subplot(1,nsub,1)
    pyplot.scatter(Z.real, Z.imag, s=0.5, c='r')
    pyplot.axis('equal')
    pyplot.title('grid1')
    pyplot.subplot(1,nsub,2)
    pyplot.scatter(XI.real, XI.imag, s=0.5, c='r')
    pyplot.axis('equal')
    pyplot.title('grid2');

def plot_contourstreamline(Z, XI, psi,nsub):
    pyplot.figure(figsize=(14, 7))
    pyplot.subplot(1,nsub,1)
    pyplot.plot(Z.real[0,:],Z.imag[0,:],'-')
    pyplot.contour(Z.real, Z.imag, psi, 71, colors='k', linestyles='solid')
    pyplot.axis('equal')
    pyplot.title('streamline1')
    pyplot.subplot(1,nsub,2)
    pyplot.plot(XI.real[0,:],XI.imag[0,:],'-')
    pyplot.contour(XI.real, XI.imag, psi, 51, colors='k', linestyles='solid')
    pyplot.axis('equal')
    pyplot.title('streamline2');

