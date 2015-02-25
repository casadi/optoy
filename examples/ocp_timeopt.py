from casadi import *
from pylab import *
from optoy import *

######################
## Problem setup    ##
######################

# Time
t = time()

# An optimization variable for the end time
T = var(lb=0,init=4)

# States: position and velocity
p = state(2,init=vertcat([3*sin(2*pi/T.init*t),3*cos(2*pi/T.init*t)]))
v = state(2)

# Control
u = control(2)

# Disturbance
w = dist(2,cov=50*DMatrix([[1,0],[0,1]]))

# Specify system dynamics
p.dot = v
v.dot = -10*(p-u)-v*sqrt(sum_square(v)+1)+w

# Specify some parameters for circular obstacles
#            ( position,      radius)
#
circles = [  (vertcat([2,2]),      1),
             (vertcat([0.5,-2]), 1.5),
          ]

# List of path constraints
h = []
for center, radius in circles:
  h.append(
     norm_2(p-center) >= radius  # Don't hit the obstacles
  )

######################
## Problem solve    ##
######################

ocp(T,h+[p.start[0]==0],regularize=[0.1*u/sqrt(2)],N=30,T=T,verbose=True,periodic=True,integration_intervals=2)

######################
## Plotting results ##
######################


# Plot the nominal trajectory
plot(value(p[0]),value(p[1]),'o-')
xlabel('x position')
xlabel('x position')
title('time-optimal trajectory')
axis('equal')

# Plot the obstacles
theta = linspace(0,2*pi,1000)
for center, radius in circles:
  fill(radius*cos(theta) + center[0],radius*sin(theta) + center[1],'r') 

# Plot the covariance ellipsoids
circle = array([[sin(x),cos(x)] for x in linspace(0,2*pi,100)]).T

show()

