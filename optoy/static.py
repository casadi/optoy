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
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#     MA  02110-1301  USA
#
#

"""
  This is the module for static optimization problems

"""

from base import *


class OptimizationVariable(OptimizationObject):

    """
      Create a decision variable

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
    shorthand = "v"
    _mapping = {}

    def __init__(self, shape=1, lb=-inf, ub=inf, name="v", init=0):
        self.lb = lb
        self.ub = ub
        self.create(shape, name)
        self.init = init
        self.sol = None


class OptimizationParameter(OptimizationObject):

    """
      Create a parameter, ie a thing that is fixed during optimization

      Parameters
      -------------------

      shape: integer or (integer,integer)
        Matrix shape of the symbol

      name: string
        A name for the symbol to be used in printing.
        Not required to be unique

      value: number or matrix
        Value that the parameter should take during optimization
        May also be set after initialization as 'x.value = number'

    """
    shorthand = "p"
    _mapping = {}

    def __init__(self, shape=1, value=0, name="p"):
        self.value = value
        self.create(shape, name)

    @property
    def sol(self):
        """
          Gets the solution
        """
        return self.value

var = OptimizationVariable
par = OptimizationParameter


def sort_constraints(gl):
    """
    Rewrites and determines nature of constraints, either g(x)<=0 or g(x)==0.

    A user may write x>=y  where x and y are variables.
    In the `gl_pure` output, everything is brought to the left hand side

    Parameters
    ----------

    gl : list of constraints, optional

    Returns
    -------
    gl_pure : list of constraints in standard form
              The constraints are rewritten as g(x)<=0 or g(x)==0

    gl_equality : list of bools
                  For each entry in `gl_pure`, this list contains a boolean.

    """
    gl_pure = []
    gl_equality = []
    for g in gl:
        if g.isOperation(OP_LE) or g.isOperation(OP_LT):
            gl_pure.append(g.getDep(0) - g.getDep(1))
            gl_equality.append(False)
        elif g.isOperation(OP_EQ):
            gl_pure.append(g.getDep(0) - g.getDep(1))
            gl_equality.append(True)
        else:
            raise Exception("Constrained type unknown. Use ==, >= or <= .")
    return gl_pure, gl_equality


def minimize(f, gl=[], verbose=False):
    """

    Miminimizes an objective function subject to a list of constraints.
    The standard NLP form reads::

        mininimze       f(x,p)
            x

         subject to     g(x,p) <= 0
                        h(x,p)  = 0

    with x the decision variables, p constant parameters,
    f the objective, g the inequality constraints,
    and h the equality constraints.

    Parameters
    ----------

    f : symbolic expression
        objective function

    gl : list of constraints, optional
         Equality and inequality constraints can be mixed.
         Each entry in the constraint list should be
              lhs<=rhs  ,   lhs>=rhs  or  lhs==rhs
         where lhs and rhs are expressions.

    verbose : bool, optional
               Specify the verbosity of the output

    Returns
    -------

    If numerical solution was succesful,
    returns cost at the optimal solution.
    Otherwise raises an exception.

    Example
    -------

    >>> x = var()
    >>> y = var()
    >>> cost = (1-x)**2+100*(y-x**2)**2
    >>> minimize(cost)
    <BLANKLINE>
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    <BLANKLINE>
    <optoy.static.StaticOptimization instance at 0x...>
    >>> print round(value(cost),8)
    0.0
    >>> print round(value(x),8), round(value(y),8)
    1.0 1.0

    See Also
    --------
    maximize : flip the sign of the objective

    """
    return StaticOptimization(f,gl,verbose)


class StaticOptimization:
  def __init__(self,f, gl=[], verbose=False):
    
    if not isinstance(gl, list):
        raise Exception("Constraints must be given as a list")

    # Determine nature of constraints, either g(x)<=0 or g(x)==0
    gl_pure, gl_equality = sort_constraints(gl)

    self.gl_equality =gl_equality

    # Get all symbolic primitives
    syms = get_primitives([f] + gl_pure)
    self.x = x = syms["v"]
    self.p = p = syms["p"]

    # Create structures
    self.X = X = struct_symMX([entry(str(hash(i)), shape=i.sparsity()) for i in x])
    self.P = P = struct_symMX([entry(str(hash(i)), shape=i.sparsity()) for i in p])
    self.G = G = struct_MX([entry(str(i), expr=g) for i, g in enumerate(gl_pure)])

    # Subsitute the casadi symbols for the structured variants
    original = MXFunction("original",x + p, nlpOut(f=f, g=G))

    nlp = MXFunction("nlp",nlpIn(x=X, p=P), original(X[...] + P[...]))
    nlp = try_expand(nlp)

    options = {}
    if not verbose:
        options["print_time"] = False
        options["print_level"] = 0
        options["verbose"] = False
    # Allocate an ipopt solver
    self.solver = solver = NlpSolver("solver","ipopt", nlp, options)
    self._solve()



  def _solve(self):
    solver = self.solver

    # Set bounds on variables, set initial value
    x0 = self.X(0)
    lbx = self.X(0)
    ubx = self.X(0)

    for i in self.x:
        h = str(hash(i))
        lbx[h] = i.lb
        ubx[h] = i.ub
        x0[h] = i.init

    # Set constraint bounds
    lbg = self.G(0)
    ubg = self.G(0)

    for i, eq in enumerate(self.gl_equality):
        if eq:
            lbg[str(i)] = ubg[str(i)] = 0
        else:
            lbg[str(i)] = -inf
            ubg[str(i)] = 0

    # Set parameter values
    par = self.P(0)

    for i in self.p:
        h = str(hash(i))
        par[h] = i.value

    # Solve the problem numerically
    sol = solver(x0=x0,lbg=lbg,ubg=ubg,lbx=lbx,ubx=ubx,p=par)

    # Raise an exception if not converged
    if self.solver.getStat('return_status') != "Solve_Succeeded":
        raise Exception(
            "Problem failed to solve. Add verbose=True to see what happened.")

    # Add the solution to the OptimizationObjects
    opt = self.X(sol["x"])
    for i in self.x:
        i.sol = opt[str(hash(i))]

  update = _solve
