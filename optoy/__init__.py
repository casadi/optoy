"""
 Optoy. Because optimization is fun!
"""
import os

with open(os.path.join(os.path.dirname(__file__) , 'conf.py')) as f:
  exec(f.read())



from static import OptimizationVariable as var, OptimizationParameter as par, \
                   minimize, maximize, value
from dynamic import OptimizationState as state, OptimizationControl as control, ocp,\
                    time

from extensions.robustness import OptimizationDisturbance as dist, Prob, Sigma as cov
