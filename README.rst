optoy
=====

Because optimization is fun!

|unix| |cover| |docs|

.. |unix| image:: https://api.travis-ci.org/casadi/optoy.svg
    :target: http://travis-ci.org/casadi/optoy
    :alt: Build Status of the master branch on Linux
    
.. |docs| image:: https://readthedocs.org/projects/optoy/badge/?version=latest
    :target: https://readthedocs.org/projects/optoy/?badge=latest
    :alt: Documentation Status

.. |cover| image:: https://coveralls.io/repos/casadi/optoy/badge.svg?branch=master
    :target: https://coveralls.io/r/casadi/optoy?branch=master
    :alt: Coverage Status

Optoy combines the power of `casadi <http://casadi.org>`_ with a very compact Python user interface.
Start optimizing in minutes...

Installation:

.. code-block:: bash

    $ pip install git+git://github.com/casadi/optoy.git


Static optimization
===================

.. code-block:: python

    from optoy import *

    x = var()
    y = var()
    print "cost = ", minimize((1-x)**2+100*(y-x**2)**2,[x**2+y**2<=1, x+y>=0])
    print "sol = ", x.sol, y.sol


Dynamic optimization
====================

.. code-block:: python

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

    plot(x.sol)
    plot(y.sol)
    plot(u.sol)

.. image:: https://cloud.githubusercontent.com/assets/329032/6380987/2a6fec10-bd3e-11e4-81b9-3b047d6e7066.png

Get started with some `examples <http://nbviewer.ipython.org/github/casadi/optoy/tree/master/examples/>`_ or by reading the `API documentation <http://optoy.readthedocs.org/en/latest/optoy.html>`_ .
