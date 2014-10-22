"""
 Optoy. Because optimization is fun!
"""
import os

with open(os.path.join(os.path.dirname(__file__) , 'conf.py')) as f:
  exec(f.read())

from static import *
