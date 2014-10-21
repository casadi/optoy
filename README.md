optoy
=====

Because optimization is fun!

Optoy combines the power of [casadi](casadi.org) with a very compact Python user interface.
Start optimizing in minutes...



Static optimisation
===================

```python
from optoy import *

x = var()
y = var()
print "cost = ", minimize((1-x)**2+100*(y-x**2)**2,[x**2+y**2<=1, x+y>=0])
print "sol = ", x.sol, y.sol
```

Dynamic optimisation
====================
```python
from optoy import *

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
plot(x.sol)
plot(y.sol)
plot(u.sol)
show()

```
