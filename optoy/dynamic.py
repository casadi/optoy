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

from static import *

class OptimizationTime(OptimizationObject):

    """
       time
    """
    shorthand = "t"
    _mapping = {}

    def __init__(self):
        self.create(1, "t")

timebase = OptimizationTime()


def time():
    return timebase


def value_time(e, t):
    try:
        return DMatrix(e)
    except:
        f = MXFunction(getSymbols(e), [e])
        f.init()
        f.setInput(t)
        f.evaluate()
        return f.getOutput()


def try_expand(f):
    if not f.isInit():
        f.init()
    try:
        r = SXFunction(f)
        r.init()
        return r
    except:
        return f


class OptimizationState(OptimizationContinousVariable):

    """
      Create a state variable

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
    shorthand = "x"
    _mapping = {}
    lim_mapping = {}

    def __init__(self, shape=1, lb=-inf, ub=inf, name="x", init=0):
        OptimizationContinousVariable.__init__(
            self,
            shape=shape,
            lb=lb,
            ub=ub,
            name=name,
            init=init)

    @classmethod
    def getDependent(cl, v):
        newvars = set()
        if hash(v) in cl._mapping:
            newvars.update(set(getSymbols(cl._mapping[hash(v)].dot)))
        if hash(v) in cl.lim_mapping:
            newvars.update(set(getSymbols(cl.lim_mapping[hash(v)])))
        return newvars


class OptimizationControl(OptimizationContinousVariable):

    """
      Create a control variable

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
    shorthand = "u"
    _mapping = {}
    lim_mapping = {}

    def __init__(self, shape=1, lb=-inf, ub=inf, name="u", init=0):
        OptimizationContinousVariable.__init__(
            self,
            shape=shape,
            lb=lb,
            ub=ub,
            name=name,
            init=init)

    @classmethod
    def getDependent(cl, v):
        newvars = set()
        if hash(v) in cl.lim_mapping:
            newvars.update(set(getSymbols(cl.lim_mapping[hash(v)])))
        return newvars

state   = OptimizationState
control = OptimizationControl

def ocp(f, gl=[], regularize=[], verbose=False, N=20, T=1.0,
        periodic=False, integration_intervals=1, exact_hessian=None):
    """

    Solves an optimal control problem (OCP)::

        mininimze         E(x(T),v)
         x(t), u(t), v

         subject to           dx/dt      = f(x(t),u(t),v,p)
                     h(x(t),u(t),v,p)   <= 0
                       r(x(0),x(T),v,p) <= 0

    with x states, u controls, p static parameters (constant, not optimized for),
    v variables (constant, optimized for), f the system dynamics,
    h the path constraints, and r boundary conditions.

    In optoy, the system dynamics is specified with the .dot attribute on a state:
    
    >>> x = state()
    >>> x.dot = 1-x**2


    Parameters
    -------------------

    N :  int, optional 
         number of control intervals
    T :  float, symbolic expression, optional
         time horizon
    periodic :  bool
                indicate whether the problem is periodic
    regularize:  list of symbolic vector expressions
                 

    f : symbolic expression
        A major objective function.
        Make use of the .end attribute of expressions

    gl : list of constraints, optional
         Equality and inequality constraints can be mixed.
         Each entry in the constraint list should be
              lhs<=rhs  ,   lhs>=rhs  or  lhs==rhs
         where lhs and rhs are expressions.
         Path constraints and boundary constraints can be mixed.
         Use .start  and .end to obtain the value of a state at the boundaries

    verbose : bool, optional
               Specify the verbosity of the output

    Returns
    -------------------

    If numerical solution was succesful,
    returns cost at the optimal solution.
    Otherwise raises an exception.

    """
    if not isinstance(gl, list):
        raise Exception("Constraints must be given as a list")
    f = f + OptimizationParameter()

    # Determine nature of constraints, either g(x)<=0 or g(x)==0
    gl_pure, gl_equality = sort_constraints(gl)

    # Get all symbolic primitives
    syms = get_primitives([f, T] + gl_pure)

    # For states and controls, retrieve the limits (value start and end time)
    lims = [i.start for i in syms["x"]] + [i.end for i in syms["x"]] + \
        [i.start for i in syms["u"]] + [i.end for i in syms["u"]]

    # Create structures
    states = struct_symMX(
        [entry(str(hash(i)), shape=i.sparsity()) for i in syms["x"]])
    controls = struct_symMX(
        [entry(str(hash(i)), shape=i.sparsity()) for i in syms["u"]])
    variables = struct_symMX(
        [entry(str(hash(i)), shape=i.sparsity()) for i in syms["v"]])

    # Identify extensions if applicable
    ext_symnames = sorted(set(syms.keys()) - {"x", "u", "v", "p", "T"})

    # Create structures for extension variables
    ext_structures = dict([(k, struct_symMX(
        [entry(str(hash(i)), shape=i.sparsity()) for i in syms[k]])) for k in ext_symnames])

    # Compose a list of extension symbols
    ext_syms = []
    for k in ext_symnames:
        ext_syms += syms[k]

    # Identify extension classes
    ext_cl = [i.__class__ for i in syms["T"]][:1]

    # Compose the NLP variables
    X = struct_symMX([entry("V", struct=variables), entry(
        "X", struct=states, repeat=N + 1), entry("U", struct=controls, repeat=N)])
    P = struct_symMX([entry(str(hash(i)), shape=i.sparsity())
                      for i in syms["p"]])

    # Create the ode function
    ode_out = MXFunction(syms["x"] +
                         syms["u"] +
                         syms["p"] +
                         syms["v"] +
                         ext_syms, [((T +
                                      0.0) /
                                     N) *
                                    vertcat([i.dot for i in syms["x"]])])
    ode_out.setOption("name", "ode_out")
    ode_out.init()

    # Group together all symbols that are constant over an integration interval
    nonstates = struct_symMX([entry("u",
                                    struct=controls),
                              entry("p",
                                    struct=P),
                              entry("v",
                                    struct=variables)] + [entry(k,
                                                                struct=ext_structures[k]) for k in ext_symnames])

    # Compose arguments made of states/nonstates parts
    ode_out_ins = states[...] + nonstates["u", ...] + \
        nonstates["p", ...] + nonstates["v", ...]
    for k in ext_symnames:
        ode_out_ins += nonstates[k, ...]

    # Change the ode function signature to accept states/nonstates
    ode = MXFunction(
        daeIn(
            x=states, p=nonstates), daeOut(
            ode=ode_out(ode_out_ins)[0]))
    ode.init()

    # Create the integrator
    intg = explicitRK(ode, 1, 4, integration_intervals)
    intg = try_expand(intg)

    Pw0 = P[...] + X["V", ...] + \
        [None if i.nlp_var else DMatrix.zeros(i.shape) for i in ext_syms]

    # Some extensions may wish to introduce new constraints
    ext_constr = []
    for cl in ext_cl:
        ext_constr_e, gl, gl_pure, gl_equality = cl.process(
            intg, syms, N, T, X, P, Pw0, states, gl, gl_pure, gl_equality)
        ext_constr += ext_constr_e

    # Set path constraints bounds
    h_out = MXFunction(syms["x"] +
                       syms["u"] +
                       syms["p"] +
                       syms["v"] +
                       ext_syms, [a for a in gl_pure if dependsOn(a, syms["x"] +
                                                                  syms["u"])])
    h_out.setOption("name", "h_out")
    g_out = MXFunction(syms["p"] +
                       syms["v"] +
                       ext_syms +
                       lims, [a for a in gl_pure if not dependsOn(a, syms["x"] +
                                                                  syms["u"])])
    g_out.setOption("name", "g_out")
    f_out = MXFunction(syms["p"] + syms["v"] + ext_syms + lims, [f])
    f_out.setOption("name", "f_out")
    reg_out = MXFunction(syms["x"] +
                         syms["u"] +
                         syms["p"] +
                         syms["v"] +
                         ext_syms, [sumAll(vertcat([inner_prod(i, i) for i in regularize])) *
                                    T /
                                    N])
    reg_out.setOption("name", "reg_out")

    # Expand if possible
    h_out = try_expand(h_out)
    g_out = try_expand(g_out)
    f_out = try_expand(f_out)
    reg_out = try_expand(reg_out)

    # Diagnostics
    if dependsOn(f, syms["x"]):
        raise Exception(
            "Objective function cannot contain pure state variables. Try adding .start or .end")

    Lims = X["X", 0, ...] + X["X", -1, ...] + X["U", 0, ...] + X["U", -1, ...]

    # Construct NLP constraints
    G = struct_MX(
        [entry(str(i), expr=g) for i, g in enumerate(g_out(Pw0 + Lims))] +
        [entry("path", expr=[h_out(X["X", k, ...] + X["U", k, ...] + Pw0) for k in range(N)])] +
        ext_constr +
        [entry("shooting", expr=[X["X", k + 1] - intg(x0=X["X", k], p=veccat([X["U", k]] + Pw0))[0] for k in range(N)])] +
        ([entry("periodic", expr=[X["X", -1] - X["X", 0]])]
         if periodic else [])
    )

    # Build a regularization expression
    # We dont use a helper state because we wisth to directly influence the
    # objective
    reg = sumAll(
        vertcat([reg_out(X["X", k, ...] + X["U", k, ...] + Pw0)[0] for k in range(N)]))

    # Construct the nlp
    nlp = MXFunction(
        nlpIn(
            x=X, p=P), nlpOut(
            f=f_out(
                Pw0 + Lims)[0] + reg, g=G))
    nlp.setOption("name", "nlp")
    nlp.init()

    # Some extensions may wish to set default options
    for cl in ext_cl:
        exact_hessian = cl.setOptions(exact_hessian)

    # If there were no objections, default to True
    if exact_hessian is None:
        exact_hessian = True

    # Allocate an ipopt solver
    solver = NlpSolver("ipopt", nlp)
    solver.setOption(
        "hessian_approximation",
        "exact" if exact_hessian else "limited-memory")
    if not verbose:
        solver.setOption("print_time", False)
        solver.setOption("print_level", 0)
        solver.setOption("verbose", False)
    solver.init()

    # Set bounds on variables, set initial value
    x0 = X(solver.input("x0"))
    lbx = X(solver.input("lbx"))
    ubx = X(solver.input("ubx"))

    for i in syms["v"]:
        hs = str(hash(i))
        lbx["V", hs] = i.lb
        ubx["V", hs] = i.ub
        x0["V", hs] = i.init

    for j in "xu":
        for i in syms[j]:
            hs = str(hash(i))
            lbx[j.capitalize(), :, hs] = i.lb
            ubx[j.capitalize(), :, hs] = i.ub
            for k in range(N + 1):
                if k == N and j == "u":
                    continue
                x0[j.capitalize(), k, hs] = value_time(
                    i.init, t=(k + 0.0) * T.init / N)

    # Set parameter values
    par = P(solver.input("p"))

    for i in syms["p"]:
        h = str(hash(i))
        par[h] = i.value

    # Set constraint bounds
    lbg = G(solver.input("lbg"))
    ubg = G(solver.input("ubg"))

    # Set normal constraints bounds
    for i, eq in enumerate(
            [e for g, e in zip(gl, gl_equality) if not dependsOn(g, syms["x"] + syms["u"])]):
        if eq:
            lbg[str(i)] = ubg[str(i)] = 0
        else:
            lbg[str(i)] = -Inf
            ubg[str(i)] = 0

    # Set path constraints bounds
    for i, eq in enumerate(
            [e for g, e in zip(gl, gl_equality) if dependsOn(g, syms["x"] + syms["u"])]):
        if eq:
            lbg["path", :, i] = ubg["path", :, i] = 0
        else:
            lbg["path", :, i] = -Inf
            ubg["path", :, i] = 0

    # Some extensions may wish to set bounds
    for cl in ext_cl:
        cl.setBounds(lbx, ubx, x0, lbg, ubg)

    lbg["shooting", :] = ubg["shooting", :] = 0

    if periodic:
        lbg["periodic"] = ubg["periodic"] = 0

    # Solve the problem numerically
    solver.evaluate()

    # Raise an exception if not converged
    if solver.getStat('return_status') != "Solve_Succeeded":
        raise Exception(
            "Problem failed to solve. Add verbose=True to see what happened.")

    # Add the solution to the OptimizationObjects
    opt = X(solver.output("x"))

    # Extract solutions
    for i in syms["v"]:
        i.sol = opt["V", str(hash(i))]
    for i in syms["x"]:
        i.sol = opt["X", :, str(hash(i))]
    for i in syms["u"]:
        i.sol = opt["U", :, str(hash(i))]

    # Some extensions may wish to extract more solutions
    for cl in ext_cl:
        cl.extractSol(solver)

    # Return optimal cost
    return float(solver.getOutput("f"))
