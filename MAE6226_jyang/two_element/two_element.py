import math
import numpy
import sys
from matplotlib import pyplot
from func_vortexSourcePanel import *

# reads of the geometry from a data file
with open ('../../lessons/resources/MainFoil_N=100.csv') as file_name:
        xmain, ymain = numpy.loadtxt(file_name, dtype=float, delimiter=',', unpack=True)
with open ('../../lessons/resources/FlapFoil_N=100.csv') as file_name:
        xflap, yflap = numpy.loadtxt(file_name, dtype=float, delimiter=',', unpack=True)


N = 100                                        # number of panels
panels_main = numpy.empty(N, dtype=object)
panels_flap= numpy.empty(N, dtype=object)
for i in range(N):
    panels_main[i] = Panel(xmain[i], ymain[i], xmain[i+1], ymain[i+1])
    panels_flap[i] = Panel(xflap[i], yflap[i], xflap[i+1], yflap[i+1])


# plots the geometry and the panels
val_x, val_y = 0.1, 0.2
x_min, x_max = min(panel.xa for panel in panels_main), max(panel.xa for panel in panels_flap)
y_min, y_max = min(panel.ya for panel in panels_flap), max(panel.ya for panel in panels_main)
x_start, x_end = x_min-val_x*(x_max-x_min), x_max+val_x*(x_max-x_min)
y_start, y_end = y_min-val_y*(y_max-y_min), y_max+val_y*(y_max-y_min)

size = 10
pyplot.figure(2,figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.grid(True)
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.plot(xmain, ymain, color='k', linestyle='-', linewidth=2)
pyplot.plot(xflap, yflap, color='k', linestyle='-', linewidth=2)
pyplot.plot(numpy.append([panel.xa for panel in panels_main], panels_main[0].xa),
                 numpy.append([panel.ya for panel in panels_main], panels_main[0].ya),
                          linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305');
pyplot.plot(numpy.append([panel.xa for panel in panels_flap], panels_flap[0].xa),
                 numpy.append([panel.ya for panel in panels_flap], panels_flap[0].ya),
                          linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305');

# put together two panels
panels = numpy.concatenate((panels_main, panels_flap))

# defines and creates the object freestream
u_inf = 1.0                                # freestream spee
alpha = 0.0                                # angle of attack (in degrees)
freestream = Freestream(u_inf, alpha)      # instantiation of the object freestream


A = build_matrix(panels)                   # compu tes the singularity matrix
b = build_rhs(panels, freestream)          # computes the freestream RHS

# solves the linear system
solution = numpy.linalg.solve(A, b)

for i, panel in enumerate(panels):
    panel.sigma = solution[i]
    gamma_m, gamma_f = solution[-2], solution[-1]

# calculates the accuracy
accuracy = sum([panel.sigma*panel.length for panel in panels])
print '--> sum of source/sink strengths:', accuracy

print 'calculating cp'
get_tangential_vel(panels, freestream, solution)

get_pressure_coefficient(panels,freestream)
print panels[2].cp
N = len(panels)
print panels[N/2+9].cp
pyplot.show()
