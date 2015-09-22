#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

from casadi import *
from casadi.tools import *


def try_expand(f):
    """ Attempts to cast an MXFunction as SXFunxtion """
    try:
        r = SXFunction(f)
        return r
    except:
        return f

class OptimizationObject(MX):

    """

        Base class for optimization objects
    """

    def create(self, shape, name):
        if not isinstance(shape, tuple):
            shape = (shape,)
        MX.__init__(self, MX.sym(name, Sparsity.dense(*shape)))
        self._mapping[hash(self)] = self

    def __iter__(self):
        while True:
            yield 123

    @classmethod
    def getDependent(cl, v):
        return set()


class OptimizationContext:
    eval_cache = {}


def get_subclasses(c):
    """
      Returns all subclasses (any level) of a given class
    """
    for i in c.__subclasses__():
        yield i
        for j in get_subclasses(i):
            yield j


def get_primitives(el, dep=True):
    """
      Out of a list of expression, retrieves all primitive expressions

      The result is sorted into a dictionary with the key originating
      from the 'shorthand' class attribute of OptimzationObject subclasses

    """
    backup = MX.__eq__

    def cmpbyhash(self, b):
        return hash(self) == hash(b)

    MX.__eq__ = cmpbyhash

    # Get an exhausive list of all casadi symbols that make up f and gl
    vars = set(symvar(veccat(el) if isinstance(el, list) else el))

    if dep:
        while True:
            newvars = set()
            for v in vars:
                for cl in get_subclasses(OptimizationObject):
                    newvars.update(cl.getDependent(v))
            v0 = len(vars)
            vars.update(newvars)
            if v0 == len(vars):
                break

    # Find out which OptimizationParameter and
    # OptimizationVariable objects correspond to those casadi symbols
    syms = dict([(cl.shorthand, []) for cl in get_subclasses(
        OptimizationObject) if hasattr(cl, "shorthand")])

    for v in vars:
        categorized = False
        for cl in get_subclasses(OptimizationObject):
            if hasattr(cl, "shorthand"):
                name = cl.shorthand

                if hash(v) in cl._mapping:
                    syms[name].append(cl._mapping[hash(v)])
                    categorized = True
        # if not categorized:
        #  raise Exception("Unknown symbol: " + str(v))
    MX.__eq__ = backup

    return syms


class OptimizationContinousVariable(OptimizationObject):

    """
      Create a decision variable with time dependance

      Parameters
      -------------------

      shape: integer or (integer,integer)
        Matrix shape of the symbol

      name: string
        A name for the symbol to be used in printing.
        Not required to be unique

      lb: number
        Lower bound on the decision variable
        May also be set after initialization as 'x.lb = number'

      ub: number
        Upper bound on the decision variable
        May also be set after initialization as 'x.ub = number'

      init: number
        Initial guess for the optimization solver
        May also be set after initialization as 'x.init = number'

    """

    def __init__(self, shape=1, lb=-inf, ub=inf, name="x", init=0):
        self.lb, self.ub, self.init, self.name = lb, ub, init, name
        self.create(shape, name)
        self.start = MX.sym("%s(start)" % name, self.sparsity())
        self.end = MX.sym("%s(end)" % name, self.sparsity())
        self.lim_mapping[hash(self.start)] = self
        self.lim_mapping[hash(self.end)] = self


def value(e, nums={}):
    """
    Evaluates the expression numerically

     Parameters
     -------------------
       e: expression to be evaluated

       nums: optional dictionary denoting the values of Variables
         if not supplied, the optimal values are assumed

    """
    if isinstance(e, list):
        return [value(i) for i in e]
    if e in OptimizationContext.eval_cache:
        f, xp = OptimizationContext.eval_cache[e]
    else:
        try:
          syms = get_primitives(e, dep=False)
        except:
          return e
        xp = []
        for k in sorted(syms.keys()):
            xp += syms[k]

        f = MXFunction("f",xp, [e])
        OptimizationContext.eval_cache[e] = (f, xp)

    N = Inf
    for i in range(len(xp)):
        x = xp[i]
        s = nums.get(x, x.sol)
        if isinstance(s, list):
            if len(s) < N:
                N = len(s)

    if N > 0 and N != Inf:
        l = []
        for k in range(N):
            for i in range(len(xp)):
                x = xp[i]
                s = nums.get(x, x.sol)
                if isinstance(s, list):
                    f.setInput(s[k], i)
                else:
                    f.setInput(s, i)
            f.evaluate()
            l.append(f.getOutput())
        return l
    else:
        for i in range(len(xp)):
            f.setInput(nums.get(xp[i], xp[i].sol), i)
        f.evaluate()
        return f.getOutput()


class FormulationExtender(OptimizationObject):
    _mapping = {}
