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

import roboticstoolbox as rtb
from roboticstoolbox import ET

dmove = 0.25


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
        A = np.array([[1.,0.,-1.,0.],[0.,1.,0.,-1.]])
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


class GripperClosed(ExprPredicate):
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
            [(self.robot, [("pose", np.array([9], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        A = np.array([[1.]])
        b = np.zeros((1,1))
        dummy_expr = AffExpr(A, b)
        val = -np.ones((2, 1)) * 0.4 # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(GripperClosed, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class RobotFacingLocation(ExprPredicate):
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
            [(self.robot, [("pose", np.array([2], dtype=np.int_))]),
            (self.location, [("value", np.array([0, 1], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }
        
        # INCONTACT_COEFF = 1e1
        col_expr = Expr(self.f, grad=self.grad_f)
        # val = np.zeros((1, 1))
        val = np.zeros((2, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(RobotFacingLocation, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1,
            tol=1e-3
        )

    def f(self, x):
        # transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) 
        # kin_tree = rtb.Robot(transforms)

        # kin_tree.q = x[:9]
        # joint_pose = kin_tree.fkine(kin_tree.q)
        # end_pose = np.concatenate([np.array([joint_pose.x, joint_pose.y, joint_pose.z])])

        # diff_vec = x[:2] - x[2:]
        # norm = np.sum(diff_vec * diff_vec)
        # return np.array([diff, -diff])
        return np.array([np.cos(x[0]) * np.linalg.norm(x[1:]) - x[1], np.sin(x[0]) * np.linalg.norm(x[1:]) - x[2]])

    def grad_f(self, x):
        # transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) 
        # kin_tree = rtb.Robot(transforms)

        # kin_tree.q = x[:9]
        # jacobian = kin_tree.jacob0(kin_tree.q)
        # pos_jacobian = jacobian[:3, :]
        # # return np.array([grad[0], -grad[0]])
        # breakpoint()
        grad = np.array([[-np.sin(x[0]) * np.linalg.norm(x[1:]), np.cos(x[0]) *x[1] / np.linalg.norm(x[1:])- 1, np.cos(x[0]) *x[2] / np.linalg.norm(x[1:])],
                        [np.cos(x[0]) * np.linalg.norm(x[1:]), np.sin(x[0]) * x[1] / np.linalg.norm(x[1:]), np.sin(x[0]) * x[2] / np.linalg.norm(x[1:]) - 1]]).reshape(2, 3)

        return grad



class GripperAtOrigin(ExprPredicate):
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
            [(self.robot, [("pose", np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }
        
        # INCONTACT_COEFF = 1e1
        col_expr = Expr(self.f, grad=self.grad_f)
        # val = np.zeros((1, 1))
        val = np.array([[0.,0.,1.]]).T # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(GripperAtOrigin, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1,
            tol=1e-3
        )

    def f(self, x):
        transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) 
        kin_tree = rtb.Robot(transforms)

        kin_tree.q = x[:9]
        joint_pose = kin_tree.fkine(kin_tree.q)
        end_pose = np.concatenate([np.array([joint_pose.x, joint_pose.y, joint_pose.z])])

        # diff_vec = x[:2] - x[2:]
        # norm = np.sum(diff_vec * diff_vec)
        # return np.array([diff, -diff])
        return end_pose.reshape(-1, 1)

    def grad_f(self, x):
        transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) 
        kin_tree = rtb.Robot(transforms)

        kin_tree.q = x[:9]
        jacobian = kin_tree.jacob0(kin_tree.q)
        pos_jacobian = jacobian[:3, :]
        # return np.array([grad[0], -grad[0]])
        # breakpoint()
        return np.block([pos_jacobian, np.zeros((3,1))])

class GripperAtLocation(ExprPredicate):
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
            [(self.robot, [("pose", np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int_))]),
            (self.location, [("value", np.array([0, 1, 2], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }
        
        # INCONTACT_COEFF = 1e1
        col_expr = Expr(self.f, grad=self.grad_f)
        # val = np.zeros((1, 1))
        val = np.zeros((3, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(GripperAtLocation, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1,
            tol=1e-3
        )

    def f(self, x):
        transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) 
        kin_tree = rtb.Robot(transforms)

        kin_tree.q = x[:9]
        joint_pose = kin_tree.fkine(kin_tree.q)
        end_pose = np.concatenate([np.array([joint_pose.x, joint_pose.y, joint_pose.z])])

        # diff_vec = x[:2] - x[2:]
        # norm = np.sum(diff_vec * diff_vec)
        # return np.array([diff, -diff])
        return end_pose.reshape(-1, 1) - x[10:]

    def grad_f(self, x):
        transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) 
        kin_tree = rtb.Robot(transforms)

        kin_tree.q = x[:9]
        jacobian = kin_tree.jacob0(kin_tree.q)
        pos_jacobian = jacobian[:3, :]
        # return np.array([grad[0], -grad[0]])
        # breakpoint()
        return np.block([pos_jacobian, np.zeros((3,1)), -np.eye(3)])

class JointsInRange(ExprPredicate):
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
            [(self.robot, [("pose", np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int_))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        self.joint_bounds = np.array([
            [-5/6*np.pi, np.pi],
            [-np.pi, np.pi/6],
            [0., np.pi],
            [-8/9*np.pi, 8/9*np.pi],
            [-7/12*np.pi, 7/12*np.pi],
            [-8/9*np.pi, 8/9*np.pi],
            [-np.pi/2, 0]
        ])  # bounds for all arm joints

        A = np.block([[np.zeros((7,3)), np.eye(7)],[np.zeros((7, 3)), -np.eye(7)]])
        b = np.zeros((14,1))
        ineq_expr = AffExpr(A, b)
        val = np.concatenate([self.joint_bounds[:,1], -self.joint_bounds[:,0]]).reshape((-1, 1))
        # val = np.zeros((1, 1))
        e = LEqExpr(ineq_expr, val)
        super(JointsInRange, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1,
            tol=1e-4
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
        A = np.array([[1.,0.,-1.,0.],[0.,1.,0.,-1.]])
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
       attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int_))])])
       A = np.block([[np.eye(10), -np.eye(10)],[-np.eye(10), np.eye(10)]])
       b = np.zeros((20, 1))
       e = LEqExpr(AffExpr(A, b), dmove*np.ones((20, 1)))
       super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2, tol=1e-3)


class StationaryBase(ExprPredicate):

   # IsMP Robot

   def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
       self.r, = params
       ## constraints  |x_t - x_{t+1}| < dmove
       ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
       attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1, 2], dtype=np.int_))])])
       A = np.block([[np.eye(3), -np.eye(3)]])
       b = np.zeros((3, 1))
       e = EqExpr(AffExpr(A, b), np.zeros((3, 1)))
       super(StationaryBase, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2, tol=1e-3)
