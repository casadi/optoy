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

from base import *


class OdeSimulator:
    def __init__(self,ode,intg="rk",T=1):
	self.ode =ode
	self.intg = intg
	self.T = T
	self.constructed = False

    def construct(self,x,*args):
        x = MX.sym("x",x.shape[0])
	argsMX = [MX.sym("u",a.sparsity()) for a in args]

        ode_fun = MXFunction("ode",daeIn(x=x,p=veccat(argsMX)),daeOut(ode=self.ode(x,*argsMX)))
        
        integr = Integrator("integrator",self.intg,ode_fun,{"tf":self.T})
        self.integr = integr
        self.constructed = True

    def __call__(self,x,*args):
	if not self.constructed:
		self.construct(x,*args)
        return self.integr(x0=x,p=veccat(args))["xf"]
