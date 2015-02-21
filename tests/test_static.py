import pytest
from optoy import *

def test_1():
  x = var()
  y = var()
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2)
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-1)<1e-8
  assert abs(y.sol-1)<1e-8

def test_2():
  x = var(lb=2)
  y = var()
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2)
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-2)<1e-7
  assert abs(y.sol-4)<1e-7

  x = var()
  y = var()
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2,[x>=2])
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-2)<1e-7
  assert abs(y.sol-4)<1e-7

  x = var()
  y = var()
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2,[2<=x])
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-2)<1e-7
  assert abs(y.sol-4)<1e-7

def test_3():
  x = var(lb=2)
  y = var(ub=3)
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2)
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-2)<1e-7
  assert abs(y.sol-3)<1e-7

  x = var()
  y = var()
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2,[x>=2,y<=3])
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-2)<1e-7
  assert abs(y.sol-3)<1e-7

def test_4():
  x = var(lb=2,ub=5)
  y = var(lb=0,ub=3)
  a = par(value=7)
  print "cost = ", minimize(y-a*x)
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-5)<1e-7
  assert abs(y.sol-0)<1e-7
  
  a.value = -3
  print "cost = ", minimize(y-a*x)
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-2)<1e-7
  assert abs(y.sol-0)<1e-7

  print value(x)
  print value(a)

  assert abs(value(x*a)-(-6))<1e-7

  """  Not yet implemented
  x = var()
  y = var()
  a = par(value=7)
  print "cost = ", minimize(y-a*x,[2<=x<=5,0<=y<=3])
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-5)<1e-7
  assert abs(y.sol-0)<1e-7
  """

def test_simple():
  x = var()
  y = var()
  print "cost = ", minimize((1-x)**2+100*(y-x**2)**2,[x**2+y**2<=1, x+y>=0])
  print "sol = ", x.sol, y.sol
  assert abs(x.sol-0.7864151510041)<1e-8
  assert abs(y.sol-0.61769830751729)<1e-8

