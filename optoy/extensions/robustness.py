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

from ..dynamic import *

OptimizationContext.Jeval_cache = {}


class OptimizationDisturbance(OptimizationObject):

    """
      Create a disturbance source term

      Parameters
      -------------------

      shape: integer or (integer,integer)
        Matrix shape of the symbol

      name: string
        A name for the symbol to be used in printing.
        Not required to be unique

      cov: symmertric matrix
        Disturbance covariance matrix

    """
    shorthand = "w"
    _mapping = {}
    ode_param = True
    nlp_var = False

    def __init__(self, shape=1, name="w", cov=None):
        self.create(shape, name)
        self.cov = cov
        self.sol = 0


def Sigma(e, nums=None):
    """
    Evaluates the covariance of an expression numerically

    Parameters
    -------------------
    e: symbolic expression
       the quantity you want the covariance of

    nums: dictionary, optional
          dictionary denoting the values of variables
          if not supplied, the optimal values are assumed

    """
    if nums is None: nums = {}
    if isinstance(e, list):
        return [value(i) for i in e]
    if e in OptimizationContext.Jeval_cache:
        Js, xp = OptimizationContext.Jeval_cache[e]
    else:

        syms = get_primitives(e, dep=False)
        xp = []
        for k in sorted(syms.keys()):
            xp += syms[k]

        f = MXFunction(xp, [e])
        f.init()

        Js = [
            f.jacobian(
                i,
                0) for i in range(
                len(xp)) if isinstance(
                xp[i],
                OptimizationState)]
        for j in Js:
            j.init()

        OptimizationContext.Jeval_cache[e] = (Js, xp)

    Pse = syms["x"][0].cov

    l = []
    for k in range(len(Pse)):
        v = []
        r = []
        for j in range(len(Js)):
            r += syms["x"][j].states_range
            for i in range(len(xp)):
                x = xp[i]
                s = nums.get(x, x.sol)
                if isinstance(s, list):
                    Js[j].setInput(s[k], i)
                else:
                    Js[j].setInput(s, i)
            Js[j].evaluate()
            v.append(Js[j].getOutput())
        r = vertcat(r)
        P = Pse[k][r, r]
        J = horzcat(v)
        l.append(mul([J, P, J.T]))
    return l


class ProbabilityFormulation(FormulationExtender):
    shorthand = "T"

    def __init__(self, shape=1, name="sqrt(h'Ph)", h=None):
        self.create(shape, name)
        self.h = h

    @classmethod
    def process(
            self, intg, syms, N, T, X, P, Pw0, states, gl, gl_pure, gl_equality):
        # Filter out the constraints that will be robustified
        self.gl_pure = gl_pure
        self.gl_equality = gl_equality

        gl = [i for i in gl if not dependsOn(i, veccat(syms["T"]))]

        gl_pure = []
        gl_equality = []
        for i, e in enumerate(self.gl_pure):
            if not dependsOn(e, veccat(syms["T"])):
                gl_pure.append(self.gl_pure[i])
                gl_equality.append(self.gl_equality[i])

        self.syms = syms
        self.states = states

        # Work with integrator sensitivities
        Af = intg.jacobian("x0", "xf")
        Bf = intg.jacobian("p", "xf")

        wdim = veccat(syms["w"]).size()

        # Construct the DPLE matrices
        As = [Af(x0=X["X", k], p=veccat([X["U", k]] + Pw0),h=1)["jac"]
              for k in range(N)]
        Bs = [Bf(x0=X["X", k], p=veccat([X["U", k]] + Pw0),h=1)["jac"]
              for k in range(N)]
        Cs = [b[:, b.size2() - wdim:] for b in Bs]

        Sigma_w = diagcat([i.cov for i in syms["w"]])

        Qs = [mul([c, Sigma_w, c.T]) for c in Cs]

        # Select a DPLE solver
        solver = "slicot"
        if not DpleSolver.hasPlugin(solver):
            print "Warning. Slicot plugin not found. You may see degraded performance"
            solver = "simple"

        # Instantiate the solver
        dple = DpleSolver("dple",
            solver, {"a": [i.sparsity() for i in As],
                     "v": [i.sparsity() for i in Qs]}
              ,{"linear_solver": "csparse"})
        Ps = horzsplit(dple(a=horzcat(As), v=horzcat(Qs))["p"], states.size)

        # Obtain path constraints jacobians
        hJs = [horzcat([jacobian(t.h, x) for x in syms["x"]])
               for t in syms["T"]]

        # Build a function out of the path constraint jac left-right
        # multiplication
        Pss = MX.sym("P", Ps[0].sparsity())
        rmargins = MXFunction("rmargins",
                              syms["x"] +
                              syms["u"] +
                              syms["p"] +
                              syms["v"] +
                              syms["w"] +
                              [Pss], [sqrt(mul([hj, Pss, hj.T])) for hj in hJs])
        rmargins = try_expand(rmargins)

        # Construct the robust path constraint
        h_robust = MXFunction(
            "h_robust",
            syms["x"] +
            syms["u"] +
            syms["p"] +
            syms["v"] +
            syms["w"] +
            syms["T"],
            [
                a for a in self.gl_pure if dependsOn(
                    a,
                    veccat(syms["T"]))])
        self.h_robust = h_robust = try_expand(h_robust)

        self.Psf = Psf = MXFunction("Psf",nlpIn(x=X, p=P), [horzcat(Ps)])

        return [entry("robust_path", expr=[h_robust(X["X", k, ...] + X["U", k, ...] + Pw0 + rmargins(
            X["X", k, ...] + X["U", k, ...] + Pw0 + [Ps[k]])) for k in range(N)])], gl, gl_pure, gl_equality

    @classmethod
    def setBounds(self, lbx, ubx, x0, lbg, ubg):
        # Set robust path constraints bounds
        for i, eq in enumerate([e for g, e in zip(
                self.gl_pure, self.gl_equality) if dependsOn(g, veccat(self.syms["T"]))]):
            lbg["robust_path", :, i] = -Inf
            ubg["robust_path", :, i] = 0

    @classmethod
    def extractSol(self, solver):
        self.Psf.setInput(solver.getOutput("x"), "x")
        self.Psf.setInput(solver.getInput("p"), "p")
        self.Psf.evaluate()

        Ps_e = horzsplit(self.Psf.getOutput(), self.states.size)
        for i in self.syms["x"]:
            i.cov = Ps_e
            i.states_range = self.states.f[str(hash(i))]

    @classmethod
    def setOptions(self, exact_hessian):
        if exact_hessian is None:
            exact_hessian = self.h_robust.nOut() == 0
        return exact_hessian


def Prob(e):
    """ h <= 0
    """
    from scipy.stats import norm

    if e.isOperation(OP_LE) or e.isOperation(OP_LT):
        h = e.getDep(0) - e.getDep(1)
    else:
        raise Exception("Prob(e): expected comparison")

    if not h.isscalar():
        raise Exception("Prob(e): expected scalar expression")
    ret = (h + ProbabilityFormulation(h.shape, h=h)) <= 0

    class Wrapper(MX):

        def g_mod(self, b):
            return (
                h - norm.ppf(b) * ProbabilityFormulation(h.shape, h=h)) >= 0

        def l_mod(self, b):
            return (
                h + norm.ppf(b) * ProbabilityFormulation(h.shape, h=h)) <= 0

        __lt__ = __le__ = l_mod
        __gt__ = __ge__ = g_mod

    return Wrapper(ret)
