import math
import numpy
from matplotlib import pyplot
from func_vortexSourcePanel import *

# reads of the geometry from a data file
with open ('../lessons/resources/naca0012.dat') as file_name:
        x, y = numpy.loadtxt(file_name, dtype=float, delimiter='\t', unpack=True)


val_x, val_y = 0.1, 0.2
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
x_start, x_end = x_min-val_x*(x_max-x_min), x_max+val_x*(x_max-x_min)
y_start, y_end = y_min-val_y*(y_max-y_min), y_max+val_y*(y_max-y_min)

size = 10
pyplot.figure(1,figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.grid(True)
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.plot(x, y, color='k', linestyle='-', linewidth=2);

N = 40                            # number of panels
panels = define_panels(x, y, N)   # discretizes of the geometry into panels

# plots the geometry and the panels
val_x, val_y = 0.1, 0.2
x_min, x_max = min(panel.xa for panel in panels), max(panel.xa for panel in panels)
y_min, y_max = min(panel.ya for panel in panels), max(panel.ya for panel in panels)
x_start, x_end = x_min-val_x*(x_max-x_min), x_max+val_x*(x_max-x_min)
y_start, y_end = y_min-val_y*(y_max-y_min), y_max+val_y*(y_max-y_min)

size = 10
pyplot.figure(2,figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.grid(True)
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.plot(x, y, color='k', linestyle='-', linewidth=2)
pyplot.plot(numpy.append([panel.xa for panel in panels], panels[0].xa),
                 numpy.append([panel.ya for panel in panels], panels[0].ya),
                          linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305');
# defines and creates the object freestream
u_inf = 1.0                                # freestream spee
alpha = 4.0                                # angle of attack (in degrees)
freestream = Freestream(u_inf, alpha)      # instantiation of the object freestream

A = build_matrix(panels)                   # compu tes the singularity matrix
b = build_rhs(panels, freestream)          # computes the freestream RHS

# solves the linear system
variables = numpy.linalg.solve(A, b)

for i, panel in enumerate(panels):
    panel.sigma = variables[i]

gamma = variables[-1]

# computes the tangential velocity at the center-point of each panel
get_tangential_velocity(panels, freestream,gamma)

# computes the surface pressure coefficients
get_pressure_coefficient(panels, freestream)

# plots the surface pressure coefficient
# plots the surface pressure coefficient
val_x, val_y = 0.1, 0.2
x_min, x_max = min( panel.xa for panel in panels ), max( panel.xa for panel in panels )
cp_min, cp_max = min( panel.cp for panel in panels ), max( panel.cp for panel in panels )
x_start, x_end = x_min-val_x*(x_max-x_min), x_max+val_x*(x_max-x_min)
y_start, y_end = cp_min-val_y*(cp_max-cp_min), cp_max+val_y*(cp_max-cp_min)

pyplot.figure(figsize=(10, 6))
pyplot.grid(True)
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('$C_p$', fontsize=16)
pyplot.plot([panel.xc for panel in panels if panel.loc == 'extrados'],
                 [panel.cp for panel in panels if panel.loc == 'extrados'],
                          color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
pyplot.plot([panel.xc for panel in panels if panel.loc == 'intrados'],
                 [panel.cp for panel in panels if panel.loc == 'intrados'],
                          color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
pyplot.legend(['extrados', 'intrados'], loc='best', prop={'size':14})
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.gca().invert_yaxis()
pyplot.title('Number of panels : %d' % N);

# calculates the accuracy
accuracy = sum([panel.sigma*panel.length for panel in panels])
print '--> sum of source/sink strengths:', accuracy

pyplot.show()
