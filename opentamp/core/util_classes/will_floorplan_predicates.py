from opentamp.core.internal_repr.predicate import Predicate
from opentamp.core.internal_repr.plan import Plan
from opentamp.core.util_classes.common_predicates import ExprPredicate
from opentamp.core.util_classes.namo_predicates import NEAR_TOL
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.core.util_classes.torch_funcs import GaussianBump, ThetaDir
from opentamp.errors_exceptions import PredicateException

from opentamp.sco_py.sco_py.expr import Expr, AffExpr, EqExpr, LEqExpr
from opentamp.core.util_classes.prob_expr import LEqEpsExpr

from collections import OrderedDict
import numpy as np
import os
import pybullet as P
import sys
import time
import traceback
import torch

dmove = 0.15

class RobotInRoom(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.robot, self.room) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))]),
            (self.room, [("low_bound", np.array([0, 1], dtype=np.int_)),
            ("high_bound", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # unused constraints, pass some BS in
        A = np.array([
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        ])
        b = np.zeros((4,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((4, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = LEqExpr(dummy_expr, val)
        super(RobotInRoom, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class LocationInRoom(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.location, self.room) = params
        attr_inds = OrderedDict(
            [(self.location, [("value", np.array([0, 1], dtype=np.int_))]),
            (self.room, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # unused constraints, pass some BS in
        A = np.zeros((1,4))
        b = np.zeros((1,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((1, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(LocationInRoom, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

    def test(self, time, negated=False, tol=1e-4):
        return True

class TargetInRoom(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.location, self.room) = params
        attr_inds = OrderedDict(
            [(self.location, [("value", np.array([0, 1], dtype=np.int_))]),
            (self.room, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # unused constraints, pass some BS in
        A = np.zeros((1,4))
        b = np.zeros((1,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((1, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(TargetInRoom, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

    def test(self, time, negated=False, tol=1e-4):
        return True


class TaskComplete(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        (self.robot, self.target) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))]),
            (self.target, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.array([[1., 0., -1., 0.],
                      [0., 1., 0., -1.]])
        b = np.zeros((2,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(TaskComplete, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class RobotAtLocation(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.robot, self.location) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))]),
            (self.location, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.eye(2)
        b = np.zeros((2,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(RobotAtLocation, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class RobotAtTarget(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        (self.robot, self.target) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))]),
            (self.target, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.array([[1., 0., -1., 0.],
                      [0., 1., 0., -1.]])
        b = np.zeros((2,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(RobotAtTarget, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class IsMP(ExprPredicate):

   # IsMP Robot

   def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
       self.r, = params
       ## constraints  |x_t - x_{t+1}| < dmove
       ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
       attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int_)), ("theta", np.array([0], dtype=np.int_))])])
       A = np.array([[1, 0, 0, -1, 0, 0],
                     [0, 1, 0, 0, -1, 0],
                     [0, 0, 1, 0, 0, -1],
                     [-1, 0, 0, 1, 0, 0],
                     [0, -1, 0, 0, 1, 0],
                     [0, 0, -1, 0, 0, 1]])
       b = np.zeros((6, 1))
       e = LEqExpr(AffExpr(A, b), np.ones((6, 1)) * dmove)
       super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2, tol=1e-2)
