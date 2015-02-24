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

def test_4():
    
  xy = state(2)
  u = control()
  T = var(lb=0,init=10)
  x = xy[0]
  y = xy[1]

  xy.dot = vertcat([(1-y**2)*x-y+u,x])

  ocp(T,[u>=-1,u<=1,xy.start==vertcat([1,0])],regularize=[10*(x**2+y**2)],T=T,N=20,verbose=True,periodic=True)

  assert abs(xy.sol[0][0]-1)<1e-7
  assert abs(xy.sol[0][1]-0)<1e-7
  assert abs(xy.sol[-1][0]-1)<1e-7
  assert abs(xy.sol[-1][1]-0)<1e-7
    

  assert abs(T.sol-5.1479130912039)<1e-7

def test_5():
  t = time()
  p = state(2,init=vertcat([3*sin(2*pi/4*t),3*cos(2*pi/4*t)]))
  v = state(2)

  u = control(2)

  p.dot = v
  v.dot = -10*(p-u)-v*sqrt(sum_square(v)+1)

  hyper = [  (vertcat([1,1]),   vertcat([0,0]),   4),
             (vertcat([0.5,2]), vertcat([1,0.5]), 4)]

  h = [ sumAll(((p-pref)/s)**n)>=1 for s,pref,n in hyper]

  T = var(lb=0,init=4)

  ocp(T,h+[p.start[0]==0],regularize=[0.1*u/sqrt(2)],N=20,T=T,verbose=True,periodic=True,integration_intervals=2)

  assert abs(T.sol-1.3742639155424)<1e-7

def test_6():
  t = time()
  p = state(2,init=vertcat([3*sin(2*pi/4*t),3*cos(2*pi/4*t)]))
  v = state(2)

  u = control(2)

  w = dist(2,cov=50*DMatrix([[1,0],[0,1]]))

  p.dot = v
  v.dot = -10*(p-u)-v*sqrt(sum_square(v)+1)+w

  hyper = [  (vertcat([1,1]),   vertcat([0,0]),   4),
             (vertcat([0.5,2]), vertcat([1,0.5]), 4)]

  h = [ Prob(sumAll(((p-pref)/s)**n)>=1) for s,pref,n in hyper]

  T = var(lb=0,init=4)

  ocp(T,h+[p.start[0]==0],regularize=[0.1*u/sqrt(2)],N=20,T=T,verbose=True,periodic=True,integration_intervals=2)
  
  Sigma = cov(p)
  assert len(Sigma)==20
  assert Sigma[0].shape==(2,2)

  assert abs(Sigma[0][0,0]-0.017781237199029)<1e-7
  assert abs(Sigma[0][1,0]-Sigma[0][0,1])<1e-7

  assert abs(T.sol-1.3958353362253)<1e-7

