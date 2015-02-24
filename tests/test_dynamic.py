import pytest
from optoy import *

from casadi import *
DMatrix.setPrecision(14)

def test_1():
  x = state()
  y = state()
  q = state()
  u = control()
  T = var(lb=0,init=10)
  x.dot = (1-y**2)*x-y+u
  y.dot = x
  q.dot = x**2+y**2

  ocp(T,[u>=-1,u<=1,q.start==0,x.start==1,y.start==0,x.end==0,y.end==0],T=T,N=20)

  print T.sol, x.sol
  assert abs(T.sol-2.9615750900664)<1e-7
  
  #assert abs(y.start.sol-0)<1e-7
  #assert abs(y.end.sol-0)<1e-7
  #assert abs(x.start.sol-1)<1e-7
  #assert abs(x.end.sol-0)<1e-7

def test_2():
  xy = state(2)
  q = state()
  u = control()
  T = var(lb=0,init=10)
  x = xy[0]
  y = xy[1]

  xy.dot = vertcat([(1-y**2)*x-y+u,x])


  q.dot = x**2+y**2

  ocp(T,[u>=-1,u<=1,q.start==0,xy.start==vertcat([1,0]),xy.end==vertcat([0,0])],T=T,N=20,verbose=True)

  print T.sol, xy.sol
  assert abs(T.sol-2.9615750900664)<1e-7

def test_3():
    
  xy = state(2)
  u = control()
  T = var(lb=0,init=10)
  x = xy[0]
  y = xy[1]

  xy.dot = vertcat([(1-y**2)*x-y+u,x])

  ocp(T,[u>=-1,u<=1,xy.start==vertcat([1,0])],T=T,N=20,verbose=True,periodic=True)

  assert abs(xy.sol[0][0]-1)<1e-7
  assert abs(xy.sol[0][1]-0)<1e-7
  assert abs(xy.sol[-1][0]-1)<1e-7
  assert abs(xy.sol[-1][1]-0)<1e-7
    

  assert abs(T.sol-3.7292171609519)<1e-7

