import pytest
from optoy import *
import optoy
print optoy

DMatrix.setPrecision(14)

def test_simple():
  x = var()
  y = var()
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2,[x**2+y**2<=1, x+y>=0])
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-0.7864151510041)<1e-8
  assert abs(y.sol-0.61769830751729)<1e-8

test_simple()
