# from opentamp.pma.backtrack_ll_solver_OSQP import *
from sco_py.expr import AffExpr, BoundExpr, QuadExpr
from sco_py.sco_osqp.prob import Prob
from sco_py.sco_osqp.solver import Solver
from sco_py.sco_osqp.variable import Variable
from opentamp.pma.backtrack_ll_solver_OSQP import BacktrackLLSolverOSQP

import numpy as np
import torch

class ToySolver(BacktrackLLSolverOSQP):
    def get_resample_param(self, a):
        return None

    def obj_pose_suggester(self, plan, anum, resample_size=1, st=0):
        # return [{"pose": plan.params['g'].value}]
        return []  # for now, does not give any hints
