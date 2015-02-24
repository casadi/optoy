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
  mapping = {}

  def __init__(self):
    self.create(1,"t")

timebase = OptimizationTime()

def time():
  return timebase

def value_time(e,t):
  try:
    return DMatrix(e)
  except:
    f = MXFunction(getSymbols(e),[e])    
    f.init()
    f.setInput(t)
    f.evaluate()
    return f.getOutput()

def try_expand(f):
  if not f.isInit(): f.init()
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
  mapping = {}
  lim_mapping = {}
  
  def __init__(self,shape=1,lb=-inf,ub=inf,name="x",init=0):
    OptimizationContinousVariable.__init__(self,shape=shape,lb=lb,ub=ub,name=name,init=init)

  @classmethod
  def getDependent(cl,v):
    newvars = set()
    if hash(v) in cl.mapping:
      newvars.update(set(getSymbols(cl.mapping[hash(v)].dot)))
    if hash(v) in cl.lim_mapping: newvars.update(set(getSymbols(cl.lim_mapping[hash(v)])))
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
  mapping = {}
  lim_mapping = {}
    
  def __init__(self,shape=1,lb=-inf,ub=inf,name="u",init=0):
    OptimizationContinousVariable.__init__(self,shape=shape,lb=lb,ub=ub,name=name,init=init)
    

  @classmethod
  def getDependent(cl,v):
    newvars = set()
    if hash(v) in cl.lim_mapping: newvars.update(set(getSymbols(cl.lim_mapping[hash(v)])))
    return newvars

def ocp(f,gl=[],regularize=[],verbose=False,N=20,T=1.0,periodic=False,integration_intervals=1,exact_hessian=True):
  """

   Miminimizes an objective function subject to a list of constraints
   
   Parameters
   -------------------
    
    N:   number of control intervals
    T:   time horizon
    periodic:  indicate whether the problem is periodic
    regularize:   list of symbolic vector expressions
        
    f:    symbolic expression
       objective function
       
    gl:   list of constraints
       each constraint should have one of these form:
             * lhs<=rhs
             * lhs>=rhs
             * lhs==rhs
             
             where lhs and rhs are expression
             
    Returns
    -------------------
    
    If numerical solution was succesful,
    returns cost at the optimal solution.
    Otherwise raises an exception.
   
  """
  if not isinstance(gl,list): raise Exception("Constraints must be given as a list")
  f = f + OptimizationParameter()

  # Determine nature of constraints, either g(x)<=0 or g(x)==0
  gl_pure, gl_equality = sort_constraints(gl)
  
  # Get all symbolic primitives
  syms =  get_primitives([f,T]+gl_pure)

  # For states and controls, retrieve the limits (value start and end time)
  lims = [i.start for i in syms["x"]] + [i.end for i in syms["x"]] + [i.start for i in syms["u"]]  + [i.end for i in syms["u"]]
  
  # Create structures
  states   = struct_symMX([entry(str(hash(i)),shape=i.sparsity()) for i in syms["x"]])
  controls = struct_symMX([entry(str(hash(i)),shape=i.sparsity()) for i in syms["u"]])
  
  X = struct_symMX([entry(str(hash(i)),shape=i.sparsity()) for i in syms["v"]]+[entry("X",struct=states,repeat=N+1),entry("U",struct=controls,repeat=N)])
  P = struct_symMX([entry(str(hash(i)),shape=i.sparsity()) for i in syms["p"]])
  
  ode_out = MXFunction(syms["x"]+syms["u"]+syms["p"]+syms["v"],[((T+0.0)/N)*vertcat([i.dot for i in syms["x"]])])
  ode_out.init()
  
  nonstates = struct_symMX([entry("controls",struct=controls),entry("p",struct=P)]+[entry(str(hash(i)),shape=i.sparsity()) for i in syms["v"]])
  
  ode = MXFunction(daeIn(x=states,p=nonstates),daeOut(ode=ode_out(states[...]+nonstates["controls",...]+nonstates["p",...]+nonstates[...][2:])[0]))
  ode.init()
  
  intg=explicitRK(ode,1,4,integration_intervals)
  intg = try_expand(intg)

  h_out = MXFunction(syms["x"]+syms["u"]+syms["p"]+syms["v"],[a for a in gl_pure if dependsOn(a,syms["x"]+syms["u"])])
  g_out = MXFunction(syms["p"]+syms["v"]+lims,[a for a in gl_pure if not dependsOn(a,syms["x"]+syms["u"])])
  f_out = MXFunction(syms["p"]+syms["v"]+lims,[f])
  reg_out = MXFunction(syms["x"]+syms["u"]+syms["p"]+syms["v"],[sumAll(vertcat([ inner_prod(i,i) for i in regularize]))*T/N])
  reg_out.setOption("name","reg_out")

  for i in [h_out, g_out, f_out, intg, reg_out]: i.init()
    
  Pw = P[...]+X[...][:len(syms["v"])]

  Lims = X["X",0,...]+X["X",-1,...]+X["U",0,...]+X["U",-1,...]
  
  # Construct NLP constraints
  G = struct_MX(
    [entry(str(i),expr=g) for i,g in enumerate(g_out(Pw+Lims))] + 
    [entry("path",expr=[ h_out(X["X",k,...]+X["U",k,...]+Pw) for k in range(N)]),
     entry("shooting",expr=[ X["X",k+1] - intg(integratorIn(x0=X["X",k],p=veccat([X["U",k]]+Pw)))[0] for k in range(N)])] +
    ([entry("periodic",expr=[ X["X",-1]-X["X",0] ])] if periodic else [])
  )

  reg = sumAll(vertcat([reg_out(X["X",k,...]+X["U",k,...]+Pw)[0] for k in range(N)]))
  
  nlp = MXFunction(nlpIn(x=X,p=P),nlpOut(f=f_out(Pw+Lims)[0]+ reg,g=G))
  nlp.setOption("name","nlp")
  nlp.init()

  # Allocate an ipopt solver
  solver = NlpSolver("ipopt",nlp)
  solver.setOption("hessian_approximation","exact" if exact_hessian else "limited-memory")
  if not verbose:
    solver.setOption("print_time",False)
    solver.setOption("print_level",0)
    solver.setOption("verbose",False)
  solver.init()

  # Set bounds on variables, set initial value
  x0  = X(solver.input("x0"))
  lbx = X(solver.input("lbx"))
  ubx = X(solver.input("ubx"))

  for i in syms["v"]:
    hs = str(hash(i))
    lbx[hs] = i.lb
    ubx[hs] = i.ub
    x0[hs]  = i.init
    
  for j in "xu":
    for i in syms[j]:
      hs = str(hash(i))
      lbx[j.capitalize(),:,hs] = i.lb
      ubx[j.capitalize(),:,hs] = i.ub
      for k in range(N+1):
        if k==N and j=="u": continue
        x0[j.capitalize(),k,hs]  = value_time(i.init,t=(k+0.0)*T.init/N)
  
  # Set parameter values
  par = P(solver.input("p"))
  
  for i in syms["p"]:
    h = str(hash(i))
    par[h] = i.value

  # Set constraint bounds
  lbg = G(solver.input("lbg"))
  ubg = G(solver.input("ubg"))
  
  # Set normal constraints bounds
  for i,eq in enumerate([e for g,e in zip(gl,gl_equality) if not dependsOn(g,syms["x"]+syms["u"])]):
    if eq:
      lbg[str(i)] = ubg[str(i)] = 0
    else:
      lbg[str(i)] = -Inf
      ubg[str(i)] = 0

  # Set path constraints bounds
  for i,eq in enumerate([e for g,e in zip(gl,gl_equality) if dependsOn(g,syms["x"]+syms["u"])]):
    if eq:
      lbg["path",:,i] = ubg["path",:,i] = 0
    else:
      lbg["path",:,i] = -Inf
      ubg["path",:,i] = 0
  
  lbg["shooting",:] = ubg["shooting",:] = 0

  if periodic:
    lbg["periodic"] = ubg["periodic"] = 0

  # Solve the problem numerically
  solver.evaluate()
  
  # Raise an exception if not converged
  if solver.getStat('return_status')!="Solve_Succeeded":
    raise Exception("Problem failed to solve. Add verbose=True to see what happened.")
    
  # Add the solution to the OptimizationObjects
  opt = X(solver.output("x"))
  
  # Extract solutions
  for i in syms["v"]: i.sol = opt[str(hash(i))]
  for i in syms["x"]: i.sol = opt["X",:,str(hash(i))]
  for i in syms["u"]: i.sol = opt["U",:,str(hash(i))]    
    
  # Return optimal cost
  return float(solver.getOutput("f"))
