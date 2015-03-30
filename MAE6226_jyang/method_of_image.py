import numpy
import math
from matplotlib import pyplot
from func import *

N = 50                                # Number of points in each direction
x_start, x_end = -2.0, 2.0            # x-direction boundaries
y_start, y_end = -1.0, 1.0            # y-direction boundaries
x = numpy.linspace(x_start, x_end, N)    # computes a 1D-array for x
y = numpy.linspace(y_start, y_end, N)    # computes a 1D-array for y
X, Y = numpy.meshgrid(x, y)              # generates a mesh grid



strength_source = 1.0                  # strength of the source
x_source, y_source = 0.0, 0.5          # location of the source

# creates a source (object of class Source)
_source = Source(strength_source, x_source, y_source)

# computes the velocity field and the stream-function on the mesh grid
_source.velocity(X, Y)
_source.stream_function(X, Y)

# creates the image of the source and computes velocity and stream-function
_source_image = Source(strength_source, x_source, -y_source)
_source_image.velocity(X, Y)
_source_image.stream_function(X, Y)

# superposition of the source and its image
u = _source.u + _source_image.u
v = _source.v + _source_image.v
psi = _source.psi + _source_image.psi

size = 10
pyplot.figure(1,figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
pyplot.scatter(_source.x, _source.y, color='#CD2305', s=80, marker='o')
pyplot.scatter(_source_image.x, _source_image.y, color='#CD2305', s=80, marker='D')
pyplot.axhline(0., color='k', linestyle='--', linewidth=4);


strength_vortex = 1.0                  # strength of the vortex
x_vortex, y_vortex = 0.0, 0.5          # location of the vortex

# creates a vortex and computes velocity and stream-function
_vortex = Vortex(strength_vortex, x_vortex, y_vortex)
_vortex.velocity(X, Y)
_vortex.stream_function(X, Y)

# creates the image of the vortex and computes velocity and stream-function
_vortex_image = Vortex(-strength_vortex, x_vortex, -y_vortex)
_vortex_image.velocity(X, Y)
_vortex_image.stream_function(X, Y)

# superposition of the vortex and its image
u = _vortex.u + _vortex_image.u
v = _vortex.v + _vortex_image.v
psi = _vortex.psi + _vortex_image.psi

# plots the streamlines
size = 10
pyplot.figure(2,figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
pyplot.scatter(_vortex.x, _vortex.y, color='#CD2305', s=80, marker='o')
pyplot.scatter(_vortex_image.x, _vortex_image.y, color='#CD2305', s=80, marker='D')
pyplot.axhline(0., color='k', linestyle='--', linewidth=4);

strength_vortex = 1.0                  # absolute value of each vortex strength
x_vortex1, y_vortex1 = -0.1, 0.5       # location of the first vortex
x_vortex2, y_vortex2 = +0.1, 0.5       # location of the second vortex

# creates two vortices at different locations
vortex1 = Vortex(+strength_vortex, x_vortex1, y_vortex1)
vortex2 = Vortex(-strength_vortex, x_vortex2, y_vortex2)

# computes the velocity and stream-function for each vortex
vortex1.velocity(X, Y)
vortex1.stream_function(X, Y)
vortex2.velocity(X, Y)
vortex2.stream_function(X, Y)

# creates an image for each vortex
vortex1_image = Vortex(-strength_vortex, x_vortex1, -y_vortex1)
vortex2_image = Vortex(+strength_vortex, x_vortex2, -y_vortex2)

# computes the velcoity and stream-function of each image
vortex1_image.velocity(X, Y)
vortex1_image.stream_function(X, Y)
vortex2_image.velocity(X, Y)
vortex2_image.stream_function(X, Y)

# superposition of the vortex pair and its image
u = vortex1.u + vortex2.u + vortex1_image.u + vortex2_image.u
v = vortex1.v + vortex2.v + vortex1_image.v + vortex2_image.v
psi = vortex1.psi + vortex2.psi + vortex1_image.psi + vortex2_image.psi

# plot the streamlines
size = 10
pyplot.figure(3,figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
pyplot.scatter(vortex1.x, vortex1.y, color='#CD2305', s=80, marker='o')
pyplot.scatter(vortex2.x, vortex2.y, color='g', s=80, marker='o')
pyplot.scatter(vortex1_image.x, vortex1_image.y, color='#CD2305', s=80, marker='D')
pyplot.scatter(vortex2_image.x, vortex2_image.y, color='g', s=80, marker='D')
pyplot.axhline(0., color='k', linestyle='--', linewidth=4);

u_inf = 1.0    # free-stream speed

# calculates the velocity and stream-function of the free-stream flow
u_freestream = u_inf * numpy.ones((N, N), dtype=float)
v_freestream = numpy.zeros((N, N), dtype=float)
psi_freestream = u_inf * Y

strength_doublet = 1.0                # strength of the doublet
x_doublet, y_doublet = 0.0, 0.3       # location of the doublet

# creates a doublet (object of class Doublet)
_doublet = Doublet(strength_doublet, x_doublet, y_doublet)

# computes the velocity and stream-function of the doublet on the mesh
_doublet.velocity(X, Y)
_doublet.stream_function(X, Y)

# creates the image of the doublet
_doublet_image = Doublet(strength_doublet, x_doublet, -y_doublet)

# computes the velocity and stream-function of the image on the mesh
_doublet_image.velocity(X, Y)
_doublet_image.stream_function(X, Y)

# superposition of the doublet and its image to the uniform flow
u = u_freestream + _doublet.u + _doublet_image.u
v = v_freestream + _doublet.v + _doublet_image.v
psi = psi_freestream + _doublet.psi + _doublet_image.psi

# plots the streamlines
size = 10
pyplot.figure(4,figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
pyplot.scatter(_doublet.x, _doublet.y, color='r', s=80, marker='o')
pyplot.scatter(_doublet_image.x, _doublet_image.y, color='r', s=80, marker='D')
pyplot.axhline(0., color='k', linestyle='--', linewidth=4);
pyplot.show()
