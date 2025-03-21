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

class DoorConnectsLocs(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.door, self.loc1, self.loc2) = params
        attr_inds = OrderedDict(
            [(self.door, [("value", np.array([0, 1], dtype=np.int_))]),
            (self.loc1, [("value", np.array([0, 1], dtype=np.int_))]),
            (self.loc2, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # unused constraints, pass some BS in
        A = np.zeros((1,6))
        b = np.zeros((1,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((1, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(DoorConnectsLocs, self).__init__(
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


class IsCentral(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.robot,) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))])]
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
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        b = np.zeros((4,1))
        dummy_expr = AffExpr(A, b)
        val = np.array([[-1.5, -1.5, 3.5, 3.5]]).T # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = LEqExpr(dummy_expr, val)
        super(IsCentral, self).__init__(
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




class RobotAtRoomTarg(ExprPredicate):
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
            (self.room, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.array([[1.,0.,-1.,0.],[0.,1.,0.,-1.]])
        b = np.zeros((2,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(RobotAtRoomTarg, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class RobotAtEntry(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.robot, self.door) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))]),
            (self.door, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.array([[1.,0.,-1.,0.],[0.,1.,0.,-1.]])
        b = np.zeros((2,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(RobotAtEntry, self).__init__(
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
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.robot,) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(s
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.zeros((2,2))
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

    def test(self, time, negated=False, tol=1e-4):
        return True

class MovedToLoc(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.robot,) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.zeros((2,2))
        b = np.zeros((2,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(MovedToLoc, self).__init__(
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


class FacedLoc(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.robot,) = params
        attr_inds = OrderedDict(
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.zeros((2,2))
        b = np.zeros((2,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(FacedLoc, self).__init__(
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



class RobotNearTarget(ExprPredicate):
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
        A = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.], [-1., 0., 1., 0.], [0., -1., 0., 1.]])
        b = np.zeros((4,1))
        dummy_expr = AffExpr(A, b)
        val = np.ones((4,1)) * 1.2 # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = LEqExpr(dummy_expr, val)
        super(RobotNearTarget, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )


class RobotNearVantage(ExprPredicate):
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
        A = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.], [-1., 0., 1., 0.], [0., -1., 0., 1.]])
        b = np.zeros((4,1))
        dummy_expr = AffExpr(A, b)
        val = np.ones((4,1)) * 0.5 # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = LEqExpr(dummy_expr, val)
        super(RobotNearVantage, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class RobotLookingDistance(ExprPredicate):
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
            [(self.robot, [("pose", np.array([0, 1], dtype=np.int_)), ("theta", np.array([0], dtype=np.int_))]),
            (self.location, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # self.r, = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        # attr_inds = OrderedDict(
        #     [
        #         (self.robot, [("pose", np.array([0, 1], dtype=np.int_))]),
        #         (self.location, [("value", np.array([0, 1], dtype=np.int_))]),
        #     ]
        # )
        col_expr = Expr(self.f, grad=self.grad_f)
        val = np.array([[.25, .25, .25, .25]]).T
        # val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)
        super(RobotLookingDistance, self).__init__(name, e, attr_inds, params, expected_param_types, priority=0)

    def f(self, x):
        # return np.array([diff, -diff])
        val = x[:2] + np.array([np.cos(x[2]), np.sin(x[2])]) - x[3:]
        return np.concatenate([val, -val])

    def grad_f(self, x):
        # return np.array([grad[0], -grad[0]])
        # breakpoint()
        
        arr = np.block([[np.eye(2), np.array([[-np.sin(x[2]), np.cos(x[2])]]).T, -np.eye(2)], [-np.eye(2), np.array([[np.sin(x[2]), -np.cos(x[2])]]).T, np.eye(2)]]).reshape((4, 5))
        return arr


class IsStationary(ExprPredicate):

   # IsMP Robot

   def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
       self.r, = params
       ## constraints  |x_t - x_{t+1}| < dmove
       ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
       attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int_)), ("theta", np.array([0], dtype=np.int_))])])
       A = np.array([[1, 0, 0, -1, 0, 0],
                     [0, 1, 0, 0, -1, 0],
                     [0, 0, 1, 0, 0, -1]])
       b = np.zeros((3, 1))
       e = EqExpr(AffExpr(A, b), np.zeros((3, 1)))
       super(IsStationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2, tol=1e-2)


class StationaryBase(ExprPredicate):

   # IsMP Robot

   def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
       self.r, = params
       ## constraints  |x_t - x_{t+1}| < dmove
       ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
       attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int_))])])
       A = np.array([[1, 0, -1, 0],
                     [0, 1, 0, -1],
                     [-1, 0, 1, 0],
                     [0, -1, 0, 1]])
       b = np.zeros((4, 1))
       e = EqExpr(AffExpr(A, b), np.zeros((4,1)))
       super(StationaryBase, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2, tol=1e-2)




class StationaryPoint(ExprPredicate):

   # IsMP Robot

   def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
       self.r, = params
       ## constraints  |x_t - x_{t+1}| < dmove
       ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
       attr_inds = OrderedDict([(self.r, [("theta", np.array([0], dtype=np.int_))])])
       A = np.array([[1,-1],
                     [-1, 1]])
       b = np.zeros((2, 1))
       e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
       super(StationaryPoint, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2, tol=1e-2)


 
class PointingAtTarget(ExprPredicate):

    # RobotAt Robot Targ

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
        (self.r, self.obs) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [
                    ("pose", np.array([0, 1], dtype=np.int_)),
                    ("theta", np.array([0], dtype=np.int_))
                ]),
                (self.obs, [
                    ("value", np.array([0, 1], dtype=np.int_)),
                ])
            ]
        )
        col_expr = Expr(self.f, grad=self.grad_f)
        val = 0.25 * np.ones((4,1))

        # val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)
        super(PointingAtTarget, self).__init__(name, e, attr_inds, params, expected_param_types, tol=1e-3, priority=0)

    def f(self, x):
        # breakpoint()
        relative_obs_pose = (x[3:]-x[:2]).reshape((-1,))
        relative_obs_dist = np.linalg.norm(relative_obs_pose)

        # return np.array([diff, -diff])
        f_res =  np.array([[relative_obs_dist * np.cos(x[2]).item() - relative_obs_pose[0]],
                         [relative_obs_dist * np.sin(x[2]).item() - relative_obs_pose[1]]])

        f_res = np.concatenate([f_res, -f_res], axis=0)
        
        return f_res

    def grad_f(self, x):
        # breakpoint()
        relative_obs_pose = x[3:]-x[:2]
        relative_obs_dist = np.linalg.norm(relative_obs_pose)

        grad = np.array([[(relative_obs_pose[0].item()/relative_obs_dist* -1) * np.cos(x[2]).item() + 1, 
                          (relative_obs_pose[1].item()/relative_obs_dist* -1) * np.cos(x[2]).item(), 
                          -relative_obs_dist * np.sin(x[2]).item(), 
                          relative_obs_pose[0].item()/relative_obs_dist * np.cos(x[2]).item() - 1, 
                          relative_obs_pose[1].item()/relative_obs_dist * np.cos(x[2]).item()],
                         [(relative_obs_pose[0].item()/relative_obs_dist * -1) * np.sin(x[2]).item(), 
                          (relative_obs_pose[1].item()/relative_obs_dist * -1) * np.sin(x[2]).item() + 1, 
                          relative_obs_dist * np.cos(x[2]).item(), 
                          relative_obs_pose[0].item()/relative_obs_dist * np.sin(x[2]).item(), 
                          relative_obs_pose[1].item()/relative_obs_dist * np.sin(x[2]).item() - 1]])
        # return np.array([grad[0], -grad[0]])

        grad = np.concatenate([grad, -grad], axis=0)

        return grad

class PointingAtTargetDotProd(ExprPredicate):

    # RobotAt Robot Targ

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
        (self.r, self.obs) = params
        ## constraints  |x_t - x_{t+1}| < dmoveget_vectorget_vector
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [
                    ("pose", np.array([0, 1], dtype=np.int_)),
                    ("theta", np.array([0], dtype=np.int_))
                ]),
                (self.obs, [
                    ("value", np.array([0, 1], dtype=np.int_)),
                ])
            ]
        )
        col_expr = Expr(self.f, grad=self.grad_f)

        val = np.ones((2, 1)) * 0.1
        e = LEqExpr(col_expr, val)
        super(PointingAtTargetDotProd, self).__init__(name, e, attr_inds, params, expected_param_types, tol=1e-3, priority=1)

    def f(self, x):
        # breakpoint()
        relative_obs_pose = (x[3:]-x[:2]).reshape((-1,))
        relative_obs_dist = np.linalg.norm(relative_obs_pose)

        # return np.array([diff, -diff])
        f_val = relative_obs_dist - np.cos(x[2]) * relative_obs_pose[0] - np.sin(x[2]) * relative_obs_pose[1]
        f_res =  np.array([f_val, -f_val])
        
        return f_res

    def grad_f(self, x):
        # breakpoint()
        relative_obs_pose = x[3:]-x[:2]
        relative_obs_dist = np.linalg.norm(relative_obs_pose)

        grad_int = np.array([
            - relative_obs_pose[0]/relative_obs_dist + np.cos(x[2]),
            - relative_obs_pose[1]/relative_obs_dist + np.sin(x[2]),
            np.sin(x[2]) * relative_obs_pose[0] - np.cos(x[2]) * relative_obs_pose[1],
            relative_obs_pose[0]/relative_obs_dist - np.cos(x[2]),
            relative_obs_pose[1]/relative_obs_dist - np.sin(x[2])
        ]).reshape(1,-1)

        grad = np.concatenate([grad_int, -grad_int])

        return grad

class CertainPosition(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.target,) = params
        attr_inds = OrderedDict(
            [(self.target, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # unused constraints, pass some BS in
        A = np.zeros((1, 2))
        b = np.zeros((1,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((1, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(CertainPosition, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=1
        )

    def test(self, time, negated=False, tol=None):
        time_trunc = min(self.target.belief.samples.shape[2]-1, time)

        diff_vec = self.target.belief.samples[:,:,time_trunc].detach().numpy() - self.target.value[:,0]

        if negated:
            return not np.max(np.abs(diff_vec) >= 1.0, axis=0).mean() <= 0.1
        
        return np.max(np.abs(diff_vec) >= 1.0, axis=0).mean() <= 0.1


class PathClear(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
        self.r, self.rt, _, _ = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.rt, [("value", np.array([0, 1], dtype=np.int_))]),
            ]
        )
        col_expr = Expr(self.f, grad=self.grad_f)
        val = -np.ones((1, 1)) * 0.4
        # val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)
        super(PathClear, self).__init__(name, e, attr_inds, params, expected_param_types, priority=0)

    def f(self, x):
        diff_vec = x[:2] - x[2:]
        norm = np.sum(diff_vec * diff_vec)
        # return np.array([diff, -diff])
        return -norm

    def grad_f(self, x):
        diff = x[:2] - x[2:]
        grad = np.array([2 * diff[0], 2 * diff[1], -2 * diff[0], -2 * diff[1]]).reshape(1, -1)
        # return np.array([grad[0], -grad[0]])
        # breakpoint()
        return -grad


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

class BAvoidObs(ExprPredicate):

   # IsMP Robot

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
        self.r, self.rt = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int_))])
            ]
        )
        col_expr = Expr(self.vector_f, grad=None)
        val = -np.ones((self.rt.belief.samples.shape[0], 1)) * 0.2
        # val = np.zeros((1, 1))
        e = LEqEpsExpr(col_expr, val, conf=0.95)
        super(BAvoidObs, self).__init__(name, e, attr_inds, params, expected_param_types, priority=1)

    def f(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        diff_vec = torch.from_numpy(x).to(device) - self.rt.belief.samples[:,:,-1].T.to(device)
        norm = torch.sum(diff_vec*diff_vec, axis=0).reshape(-1, 1)
        norm = norm.detach().cpu().numpy()
        # return np.array([diff, -diff])
        # if not np.isnan(x).any():
        #     breakpoint()
        norm_thresh = np.minimum(-norm, -0.05)
        norm_thresh = np.maximum(norm_thresh, -1.0)
        return np.sum(norm_thresh, axis=0)

    ## give another expression than the one defining the eval constraints -- for grads and stuff
    def get_expr(self, negated=False):
        col_expr = Expr(self.f, grad=self.grad_f)
        val = -np.ones((1, 1)) * 0.2 * self.rt.belief.samples.shape[0]
        # val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)
        return e

    def vector_f(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        diff_vec = torch.from_numpy(x).to(device) - self.rt.belief.samples[:,:,-1].T.to(device)
        norm = torch.sum(diff_vec*diff_vec, axis=0).reshape(-1, 1)
        norm = norm.detach().cpu().numpy()
        # return np.array([diff, -diff])
        # if not np.isnan(x).any():
        #     breakpoint()
        norm_thresh = np.minimum(-norm, -0.05)
        norm_thresh = np.maximum(norm_thresh, -1.0)
        return norm_thresh


    def grad_f(self, x):
        diff = torch.from_numpy(x).T - self.rt.belief.samples[:,:,-1]
        grad = 2 * diff
        eval_x = torch.from_numpy(self.f(x))
        grad = torch.where(-1.0 < eval_x, grad, torch.zeros(grad.shape)) ## only zero constrains out when 
        grad = torch.where(eval_x < -.2, grad, torch.zeros(grad.shape)) ## only zero constrains out when 
        grad_flip = -grad.detach().numpy()
        return np.sum(grad_flip, axis=0).reshape(1, -1)
