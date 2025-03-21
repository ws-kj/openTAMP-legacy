import pickle as pickle
from datetime import datetime
import numpy as np
import os
import pprint
import queue
import random
import sys
import time

from PIL import Image
from scipy.cluster.vq import kmeans2 as kmeans

from opentamp.software_constants import *
from opentamp.core.internal_repr.plan import Plan
import opentamp.core.util_classes.transform_utils as T
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.utils.sample_list import SampleList
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.servers.server import Server
from opentamp.policy_hooks.search_node import *
from opentamp.errors_exceptions import PredicateException

import torch
import matplotlib.pyplot as plt

LOG_DIR = 'experiment_logs/'

class MotionServer(Server):
    def __init__(self, hyperparams):
        super(MotionServer, self).__init__(hyperparams)
        self.in_queue = self.motion_queue
        self.out_queue = self.task_queue
        self.label_type = 'optimal'
        # self.opt_wt = hyperparams['opt_wt']
        self.motion_log = LOG_DIR + hyperparams['weight_dir'] + '/MotionInfo_{0}_log.txt'.format(self.id)
        self.log_infos = []
        self.infos = {'n_ff': 0, 
                      'n_postcond': 0, 
                      'n_precond': 0, 
                      'n_midcond': 0, 
                      'n_explore': 0, 
                      'n_plans': 0}

        self.avgs = {key: [] for key in self.infos}

        self.fail_infos = {'n_fail_ff': 0, 
                           'n_fail_postcond': 0, 
                           'n_fail_precond': 0, 
                           'n_fail_midcond': 0, 
                           'n_fail_explore': 0, 
                           'n_fail_plans': 0}

        self.fail_avgs = {key: [] for key in self.fail_infos}

        self.fail_rollout_infos = {'n_fail_rollout_ff': 0, 
                                   'n_fail_rollout_postcond': 0, 
                                   'n_fail_rollout_precond': 0, 
                                   'n_fail_rollout_midcond': 0, 
                                   'n_fail_rollout_explore': 0}

        self.init_costs = []
        self.rolled_costs = []
        self.final_costs = []
        self.plan_times = []
        self.plan_horizons = []
        self.log_iter = 0
        self.opt_rollout_info = {'{}_opt_rollout_success'.format(taskname): [] for taskname in self.task_list}
        with open(self.motion_log, 'w+') as f:
            f.write('')


    def gen_plan(self, node):
        node.gen_plan(self.agent.hl_solver, 
                      self.agent.openrave_bodies, 
                      self.agent.ll_solver)
        
        plan = node.curr_plan

        ## belief-space planning hook
        if 'observation_model' in self._hyperparams.keys():
            plan.set_observation_model(node.observation_model)

        if type(plan) is str: return plan
        if not len(plan.actions): return plan

        for a in range(min(len(plan.actions), plan.start+1)):
            task = self.agent.encode_action(plan.actions[a])

            self.agent.set_symbols(plan, task, a, targets=node.targets)
        
        plan.start = min(plan.start, len(plan.actions)-1)
        ts = (0, plan.actions[plan.start].active_timesteps[0])

        ## NOTE: we allow for failed prefixes of plans, owing to nondeterminisim from random effects
        ##  in belief-space plans (more things may fail silently because of this...)
        # try:
        #     failed_prefix = plan.get_failed_preds(active_ts=ts, tol=1e-3)
        # except Exception as e:
        #     failed_prefix = ['ERROR IN FAIL CHECK', e]

        # if len(failed_prefix) and node.hl:
        #     print('BAD PREFIX! -->', plan.actions[:plan.start], 'FAILED', failed_prefix, node._trace)
        #     plan.start = 0

        # ts = (0, plan.actions[plan.start].active_timesteps[0])
        
        # if node.freeze_ts <= 0:
        ## if this is the initial planned action, populate the state_inds with random init-state from node (don't do this later)
        if plan.start == 0:
            set_params_attrs(plan.params, self.agent.state_inds, node.x0, ts[1], use_symbols=True)

        plan.freeze_actions(plan.start)
        # cur_t = node.freeze_ts if node.freeze_ts >= 0 else 0

        # breakpoint()

        return plan


    def refine_plan(self, node):
        # breakpoint()
        
        start_t = time.time()
        if node is None: return

        plan = self.gen_plan(node)

        if type(plan) is str or not len(plan.actions): return

        cur_t = node.freeze_ts if node.freeze_ts >= 0 else 0
        cur_step = 2
        self.n_plans += 1

        # breakpoint()

        while cur_t >= 0:
            path, refine_success, replan_fail_step = self.collect_trajectory(plan, node, cur_t)
            
            ## if plan about to fail since hitting expansion limit
            if node.expansions >= EXPAND_LIMIT or (replan_fail_step == -1 and refine_success):
                self.log_node_info(node, replan_fail_step == -1, path)
            
            prev_t = cur_t
            cur_t -= cur_step
            if (refine_success and replan_fail_step == -1) and len(path) and path[-1].success: continue

            # parse the failed predicates for the plan, and push the change to task queue
            if not (refine_success and replan_fail_step == -1): self.parse_failed(plan, node, prev_t, replan_fail_step)
            while len(plan.get_failed_preds((cur_t, cur_t))) and cur_t > 0:
                cur_t -= 1

            node.freeze_ts = cur_t

            plan = self.gen_plan(node)

    def collect_trajectory(self, plan, node, cur_t):        
        x0 = None
        if cur_t < len(node.ref_traj): x0 = node.ref_traj[cur_t]
        if cur_t == 0: x0 = node.x0

        wt = self.explore_wt if node.label.lower().find('rollout') >= 0 or node.nodetype.find('dagger') >= 0 else 1.
        # verbose = self.verbose and (self.id.find('r0') >= 0 or np.random.uniform() < 0.05)
        # self.agent.store_hist_info(node.info)
        
        init_t = time.time()

        ## preprocessing for beliefs vector (used for certainty constraints, e.g.)
        if plan.start == 0 and len(plan.belief_params) > 0:
            print('Planning on new problem')
            
            planned_obs = {}
            node.conditioned_obs = {}
            node.conditioned_targ = {}
            node.replan_start = 0

            unnorm_loglikelihood = plan.observation_model.get_unnorm_obs_log_likelihood(plan.params, node.concr_prob.init_state.preds, node.conditioned_obs, 0)
            new_goal_idx = torch.argmax(unnorm_loglikelihood).item()

            ## get a true belief state, to plan against in the problem (if not given by a rollout)
            if not node.belief_true:
                self.agent.gym_env.reset_to_state(node.x0)
                node.belief_true = self.agent.gym_env.sample_belief_true()
                print('Node Belief True: ', node.belief_true)

            self.config['skolem_populate_fcn'](plan)

            for param in plan.belief_params:
                if self._hyperparams['assume_true']:
                    planned_obs[param.name] = node.belief_true[param.name]
                else:
                    planned_obs[param.name] = param.belief.samples[new_goal_idx,:,0]

            for param in plan.belief_params:
                if param.is_symbol():
                    param.value[:, 0] = planned_obs[param.name].detach().numpy()
                    print('Planned value: ', param.value[:, 0])
                else:
                    param.pose[:, 0] = planned_obs[param.name].detach().numpy()

            
            plan.observation_model.set_active_planned_observations(planned_obs)
            
            plan.set_mc_lock(self.config['mc_lock'])
        
        print('Full Conditioned Obs At Start: ', node.conditioned_obs)
        print('Full Conditioned Targs At Start: ', node.conditioned_targ)
        print('Planning from: ', plan.start)

        ## set the true state of belief variables from sim
        if len(plan.belief_params) > 0:
            self.agent.gym_env.set_belief_true(node.belief_true)
            goal = node.belief_true
            
        ## disable predicates for all actions coming before the start of current planning action
        for anum in range(plan.start):
            a = plan.actions[anum]
            for pred_dict in a.preds:
                ## disable the predicate by using no-eval notation
                pred_dict['active_timesteps'] = (pred_dict['active_timesteps'][0], pred_dict['active_timesteps'][0]-1)
                # pred_dict['pred'].active_range = (pred_dict['pred'].active_range[0], pred_dict['pred'].active_range[0]-1)
    

        ## reset beliefs and observations to beginning of refine_start
        if len(plan.belief_params) > 0:
            for param in plan.belief_params:
                param.belief.samples = param.belief.samples[:, :, :plan.actions[plan.start].active_timesteps[0]+1]
            del_list = []
            for t in node.conditioned_obs.keys():
                if t[0] >= plan.actions[plan.start].active_timesteps[0]:
                    del_list.append(t)
            for t in del_list:
                del node.conditioned_obs[t]
        
        ## if provided, construct initalization path for optimizer
        init_traj = []
        
        # with open('sample_trajectory.pkl','rb') as f:
        #     init_traj = pickle.load(f)

        # plan.state_inds = self.agent.state_inds


        if len(node.path):
            init_traj = np.zeros((1, node.path[0].get(STATE_ENUM).shape[1]))
            for s in node.path:
                init_traj = np.concatenate((init_traj, s.get(STATE_ENUM)[1:, :]), axis=0)

            plan.state_inds = self.agent.state_inds

        refine_success = self.agent.ll_solver._backtrack_solve(plan,
                                                      anum=plan.start,
                                                      n_resamples=5,
                                                      init_traj=init_traj,
                                                      st=cur_t, 
                                                      conditioned_obs=node.conditioned_obs,
                                                      conditioned_targ=node.conditioned_targ)

        
        ## for belief-space replanning, only replan if indeed belief-space, and plan against sampled obs dict
        replan_fail_step = -1
        if refine_success and len(plan.belief_params) > 0:
            print('Refining from: ', node.replan_start)

            # enable deterministic predicates for plan
            # for a in plan.actions:
            #     if a.non_deterministic:
            #         for pred_dict in a.preds:
            #             # reactivate the predicate
            #             pred_dict['active_timesteps'] = pred_dict['store_active_timesteps']
                        # pred_dict['pred'].active_range = (pred_dict['active_timesteps'][0] - a.active_timesteps[0], pred_dict['active_timesteps'][1] - a.active_timesteps[0])
            
            # activate eval of optimistic predicates
            for a_idx in range(node.replan_start, len(plan.actions)):
                for pred_dict in plan.actions[a_idx].preds:
                    if pred_dict['pred'].optimistic:
                        pred_dict['active_timesteps'] = (pred_dict['active_timesteps'][0], pred_dict['active_timesteps'][0])


            ## reset beliefs and observations to beginning of refine_start
            for param in plan.belief_params:
                param.belief.samples = param.belief.samples[:, :, :plan.actions[node.replan_start].active_timesteps[0]+1]

            ## remove observations occuring after replan start    
            # del_list = []
            # for t in node.conditioned_obs.keys():
            #     if t[0] >= plan.actions[node.replan_start].active_timesteps[0]:
            #         del_list.append(t)
            # for t in del_list:
            #     del node.conditioned_obs[t]

            true_obs = {}
            ## filter them forward, under the new assumption for the observation
            for anum in range(node.replan_start, len(plan.actions)):
                active_ts = plan.actions[anum].active_timesteps
                
                plan.rollout_beliefs(active_ts)
                
                if plan.actions[anum].non_deterministic:
                    ## perform filtering to update, using the goal inferred from at the time, add new observation planned against
                    obs = plan.filter_beliefs(active_ts, provided_goal=goal, past_obs=node.conditioned_obs)
                    true_obs[plan.actions[anum].active_timesteps] = obs
                    print(true_obs)
                else:
                    ## just propagate beliefs forward, no inference needed
                    for param in plan.belief_params:
                        new_samp = torch.cat((param.belief.samples, param.belief.samples[:, :, -1:]), dim=2)
                        param.belief.samples = new_samp
                anum += 1
            
            ## get minimal step for observation failure, if such a step exists
            # breakpoint()

            missing_key = False
            for t in true_obs.keys():
                for obj in true_obs[t].keys():
                    if not missing_key:
                        if t not in node.conditioned_obs.keys():
                            replan_fail_step = t[1]
                            missing_key = True
                            continue
                        if torch.linalg.vector_norm(true_obs[t][obj] - node.conditioned_obs[t][obj]) >= 0.25 and (replan_fail_step == -1 or replan_fail_step > t[1]):
                            replan_fail_step = t[1]
                            missing_key = True
                            continue
            
            ## see if plan with new beliefs is still valid
            ## reset the observation to new sample if *NOT* solved
            # fail_step, fail_pred, _ = node.get_failed_pred()
            if replan_fail_step > -1:
                ## some observation disagrees

                ## remove all observations after the replan_start
                del_list = []

                for t in node.conditioned_obs.keys():
                    if t[0] >= plan.actions[node.replan_start].active_timesteps[0]:
                        del_list.append(t)
                for t in del_list:
                    del node.conditioned_obs[t]
                
                ## add successful observations before the first disagreed timestep
                for t in true_obs.keys():
                    if t[0] <= replan_fail_step:
                        node.conditioned_obs[t] = true_obs[t]
                replan_success = False

        ## path of samples in imitation
        path = []

        ## for now need refine success, TODO NaN error handle
        if refine_success and replan_fail_step >= 0:
            self.agent.reset_to_state(node.x0)
            self.agent.populate_sim_with_facts(node.concr_prob)

            fact_dict = self.agent.gym_env.return_dict_with_facts()

            if self.config['step_debug']:
                visualize_failure = input("Do you want to visualize this failure? Please reply Y/N: ")
                while visualize_failure not in ['Y', 'N']:
                    visualize_failure = input("Please reply Y/N: ")
                if visualize_failure == 'Y':
                    tasks = self.config['get_task_nums'](plan, self.agent)
                    st = plan.actions[0].active_timesteps[0]
                    for task_idx, task in enumerate(tasks):
                        hist_info=self.config['build_hist_info'](path)
                        aux_info=self.config['build_aux_info'](plan, node.conditioned_targ, task_idx)
                        new_path, x0 = self.agent.run_action(plan, 
                            min(task_idx, len(plan.actions)-1), ## by default, repeat the last action
                            x0,
                            [np.array([0.0, 0.0,])], 
                            task, 
                            st,
                            reset=True,
                            save=True, 
                            record=True,
                            hist_info=hist_info,
                            aux_info=aux_info)
                        path.extend(new_path)
                    self.save_video(path, True, lab='vid_partial_plan')

        elif refine_success and replan_fail_step == -1:
            # domain-specific sample population method for agent            
            self.agent.reset_to_state(node.x0)
            self.agent.populate_sim_with_facts(node.concr_prob)

            fact_dict = self.agent.gym_env.return_dict_with_facts()

            tasks = self.config['get_task_nums'](plan, self.agent)

            print(tasks)
            #breakpoint()

            st = plan.actions[0].active_timesteps[0]
            for task_idx, task in enumerate(tasks):
                hist_info=self.config['build_hist_info'](path)
                aux_info=self.config['build_aux_info'](plan, node.conditioned_targ, task_idx)
                new_path, x0 = self.agent.run_action(plan, 
                    min(task_idx, len(plan.actions)-1), ## by default, repeat the last action
                    x0,
                    [np.array([0.0, 0.0,])],
                    task, 
                    st,
                    reset=True,
                    save=True, 
                    record=True,
                    hist_info=hist_info,
                    aux_info=aux_info)
                path.extend(new_path)
                #breakpoint()

            if self.log_iter == 0:
                self.log_path(path, 10, aux=fact_dict)

            self.log_iter = (self.log_iter+1) % 100

            end_t = time.time()

            for step in path:
                step.wt = wt

            self.plan_horizons.append(plan.horizon)
            self.plan_horizons = self.plan_horizons[-5:]
            self.plan_times.append(end_t-init_t)
            self.plan_times = self.plan_times[-5:]

            self.agent.add_task_paths([path])  ## add the given history of tasks from this successful rollout

            print('Success')

            ## if plan only, invoke a breakpoint and inspect the plan statistics
            ## reset sim state to state planned against
            if self.config['plan_only']:
                self.agent.reset_to_state(node.x0)
                self.agent.populate_sim_with_facts(node.concr_prob)
                if len(plan.belief_params)>0:
                    self.agent.gym_env.set_belief_true(node.belief_true)


                for sample in path:
                    #print(sample.get(STATE_ENUM))
                    #print(sample.get(ACTION_ENUM))
                    #print(sample.get(MJC_SENSOR_ENUM))
                    #print(sample.get(MJC_SENSOR_ENUM))
                    print(sample.get_X())
                    #breakpoint()

                print("Saving video...")
                self.save_video(path, True, lab='vid_planner')
            
                #breakpoint()

                raise Exception('Terminating after single plan attempt')
                
        return path, refine_success, replan_fail_step


    def parse_failed(self, plan, node, prev_t, replan_fail_step):        
        if replan_fail_step > -1:
            fail_step = replan_fail_step
            fail_pred = None
            fail_negated = False
        else:
            try:
                fail_step, fail_pred, fail_negated = node.get_failed_pred(st=prev_t)
            except:
                fail_pred = None

            if fail_pred is None:
                print('WARNING: Failure without failed constr?')
                return

            # NOTE: refines also done if linear constraints fail, owing to replanning
            failed_preds = plan.get_failed_preds((prev_t, fail_step), priority=-1)
        
        # deactivate eval of optimistic predicates if we had a replan failure
        if replan_fail_step > -1:
            for a_idx in range(node.replan_start, len(plan.actions)):
                for pred_dict in plan.actions[a_idx].preds:
                    if pred_dict['pred'].optimistic:
                        pred_dict['active_timesteps'] = (pred_dict['active_timesteps'][0], pred_dict['active_timesteps'][0]-1)

        if replan_fail_step == -1 and len(failed_preds):
            print('Refine failed with linear constr. viol.', 
                   node._trace, 
                   plan.actions, 
                   failed_preds, 
                   len(node.ref_traj), 
                   node.label,)
            
            plan.rollout_beliefs([0,2]) ## add a single sample, to avoid off-by-ones (latest will *not* have inference step done if stochastic)
            

            # return ## NOTE: altered return logic here, so all failures hit expansion limit
        
        if replan_fail_step == -1:
            print('Refine failed:', 
                plan.get_failed_preds((0, fail_step)), 
                fail_pred, 
                fail_step, 
                plan.actions, 
                node.label, 
                node._trace, 
                prev_t,)
            
            plan.rollout_beliefs([0,2]) ## add a single sample, to avoid off-by-ones (latest will *not* have inference step done if stochastic)

        if not node.hl and not node.gen_child(): 
            return

        n_problem = node.get_problem(fail_step, fail_pred, fail_negated)
        removed_pred = None
        ## TODO migrate out -- skolem facts for problems
        swap_room = False
        for pred in [p for p in n_problem.init_state.preds]:
            if pred.get_type() == 'TargetInRoom':
                par = pred.params[1]
                arr_in_low = np.min(par.low_bound.reshape(1, 2) <= plan.params['targ'].belief.samples[:,:,-1].numpy(), axis=1)
                arr_in_high = np.min(plan.params['targ'].belief.samples[:,:,-1].numpy() <= par.high_bound.reshape(1, 2), axis=1)
                in_room_arr = np.minimum(arr_in_low, arr_in_high)
                if in_room_arr.mean() < 0.05:  
                    n_problem.init_state.preds.discard(pred)
                    removed_pred = pred
                    swap_room = True
        if swap_room:
            for par in plan.params.values():
                if par.get_type() == 'Room':
                    arr_in_low = np.min(par.low_bound.reshape(1, 2) <= plan.params['targ'].belief.samples[:,:,-1].numpy(), axis=1)
                    arr_in_high = np.min(plan.params['targ'].belief.samples[:,:,-1].numpy() <= par.high_bound.reshape(1, 2), axis=1)
                    in_room_arr = np.minimum(arr_in_low, arr_in_high)
                    if in_room_arr.mean() >= 0.05:
                        removed_pred.params[1] = par
                        n_problem.init_state.preds.add(removed_pred)
                        break
        abs_prob = self.agent.hl_solver.translate_problem(n_problem, goal=node.concr_prob.goal)
        prefix = node.curr_plan.prefix(fail_step)

        if len(plan.belief_params) > 0:
            for anum in range(len(plan.actions)):
                    a = plan.actions[anum]
                    new_assumed_goal = {}

                    ## reset the true state planned to be a random one, with consistent index across samples
                    # new_goal_idx = np.random.randint(0, param.belief.samples.shape[0])
                    unnorm_loglikelihood = plan.observation_model.get_unnorm_obs_log_likelihood(plan.params, node.concr_prob.init_state.preds, node.conditioned_obs, fail_step)
                    new_goal_idx = torch.argmax(unnorm_loglikelihood).item()

                    if a.active_timesteps[0] <= fail_step and fail_step < a.active_timesteps[1]:
                        if replan_fail_step > -1:
                            for tmp_a_idx in range(node.replan_start, anum):
                                node.conditioned_targ[tmp_a_idx] = plan.params['targ'].value[:,0].copy()
                        
                        for param in plan.belief_params:
                            ## set new assumed value for planning to sample from belief -- random choice
                            self.config['skolem_populate_fcn'](plan, repln=True)
                            
                            if self._hyperparams['assume_true']:
                                new_assumed_goal[param.name] = node.belief_true[param.name]
                            else:
                                new_assumed_goal[param.name] = param.belief.samples[new_goal_idx,:,a.active_timesteps[0]]
                            if param.is_symbol():
                                param.value[:, 0] = new_assumed_goal[param.name]
                                print('Planned value: ', param.value[:, 0])
                            else:
                                param.pose[:, a.active_timesteps[0]] = new_assumed_goal[param.name].detach().numpy()

                            new_assumed_goal[param.name] = torch.tensor(new_assumed_goal[param.name])
                        
                        if replan_fail_step > -1: ## update the counter only when you get failures from a replan
                            node.replan_start = anum

                        ## populate with new goal
                        plan.observation_model.set_active_planned_observations(new_assumed_goal)

                    for pred_dict in a.preds:
                        ## disable all actions strictly prior, and optimistic actions possibly including current one, in replanning
                        if (pred_dict['active_timesteps'][0] < fail_step and fail_step <= pred_dict['active_timesteps'][1]) and pred_dict['pred'].optimistic:
                            ## disable the predicate by using no-eval notation
                            pred_dict['active_timesteps'] = (pred_dict['active_timesteps'][0], pred_dict['active_timesteps'][0]-1)
                            # pred_dict['pred'].active_range = (pred_dict['pred'].active_range[0], pred_dict['pred'].active_range[0]-1)

        hlnode = HLSearchNode(abs_prob,
                             node.domain,
                             n_problem,
                             priority=node.priority+1,
                             prefix=prefix,
                             llnode=node,
                             x0=node.x0,
                             targets=node.targets,
                             expansions=node.expansions+1,
                             label=self.id,
                             nodetype=node.nodetype,
                             info=node.info, 
                             replan_start=node.replan_start,
                             conditioned_obs=node.conditioned_obs,
                             conditioned_targ=node.conditioned_targ,
                             observation_model=node.observation_model,
                             belief_true=node.belief_true,
                             path=node.path)
        self.push_queue(hlnode, self.task_queue)
        print(self.id, 'Failed to refine, pushing to task node.')


    def run(self):
        step = 0
        while not self.stopped:
            node = self.pop_queue(self.in_queue)

            if node is None:
                time.sleep(0.01)
                if self.debug or self.plan_only:
                    break # stop iteration after one loop

                continue

            # turn abstract HL plan into motion plan, and collect trajectory samples + populate opt_sample buffers
            self.set_policies()
            self.write_log()
            self.refine_plan(node)

            inv_cov = self.agent.get_inv_cov()
                        
            if not self.plan_only:
                ## send LL samples to policy server, if training imitation
                for task in self.agent.task_list:
                    data = self.agent.get_opt_samples(task, clear=True)
                    opt_samples = [sample for sample in data if not len(sample.source_label) or sample.source_label.find('opt') >= 0]
                    expl_samples = [sample for sample in data if len(sample.source_label) and sample.source_label.find('opt') < 0]
                    
                    if len(opt_samples):
                        self.update_policy(opt_samples, label='optimal', inv_cov=inv_cov, task=task)

                    if len(expl_samples):
                        self.update_policy(expl_samples, label='dagger', inv_cov=inv_cov, task=task)

                # send HL samples to policy server
                self.run_hl_update()
                
                # add cont sample TODO where does this apply?
                cont_samples = self.agent.get_cont_samples()
                if len(cont_samples):
                    self.update_cont_network(cont_samples)

                step += 1

            if self.debug or self.plan_only:
                break # stop iteration after one loop


    def _log_solve_info(self, path, success, node, plan):
        self.n_failed += 0. if success else 1.
        n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans']
        with n_plans.get_lock():
            n_plans.value += 1

        if self.verbose and len(path):
            if node.nodetype.find('dagger') >= 0 and np.random.uniform() < 0.05:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_dgr'.format(success))
            elif np.random.uniform() < 0.05:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_opt'.format(success), annotate=True)
            elif not success and np.random.uniform() < 0.5:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_opt_fail'.format(success), annotate=True)

        if self.verbose and self.render:
            for ind, batch in enumerate(info['to_render']):
                for next_path in batch:
                    if len(next_path):
                        print('BACKUP VIDEO:', next_path[-1].task)
                        self.save_video(next_path, next_path[-1]._postsuc, lab='_{}_backup_solve'.format(ind))

        self.log_path(path, 10)
        for step in path: step.source_label = node.nodetype

        if success and len(path):
            print(self.id, 
                  'succ. refine:', 
                  node.label, 
                  plan.actions[0].name, 
                  'rollout succ:', 
                  path[-1]._postsuc, 
                  path[-1].success, 
                  'goal:', 
                  self.agent.goal(0, path[-1].targets), )

        if len(path) and path[-1].success:
            n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_total']
            with n_plans.get_lock():
                n_plans.value += 1

        n_plan = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}'.format(node.nodetype)]
        with n_plan.get_lock():
            n_plan.value += 1

        if not success:
            print('Opt failure from', node.label, node.nodetype)
            n_fail = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}_failed'.format(node.nodetype)]
            with n_fail.get_lock():
                n_fail.value += 1


    def update_expert_demos(self, demos):
        for path in demos:
            for key in self.expert_demos:
                self.expert_demos[key].append([])
            for s in path:
                for t in range(s.T):
                    if not s.use_ts[t]: continue
                    self.expert_demos['acs'][-1].append(s.get(ACTION_ENUM, t=t))
                    self.expert_demos['obs'][-1].append(s.get_prim_obs(t=t))
                    self.expert_demos['ep_rets'][-1].append(1)
                    self.expert_demos['rews'][-1].append(1)
                    self.expert_demos['tasks'][-1].append(s.get(FACTOREDTASK_ENUM, t=t))
                    self.expert_demos['use_mask'][-1].append(s.use_ts[t])
        if self.cur_step % 5:
            np.save(self.expert_data_file, self.expert_demos)


    def log_node_info(self, node, success, path):
        key = 'n_ff'
        if node.label.find('post') >= 0:
            key = 'n_postcond'
        elif node.label.find('pre') >= 0:
            key = 'n_precond'
        elif node.label.find('mid') >= 0:
            key = 'n_midcond'
        elif node.label.find('rollout') >= 0:
            key = 'n_explore'

        self.infos[key] += 1
        self.infos['n_plans'] += 1
        for altkey in self.avgs:
            if altkey != key:
                self.avgs[altkey].append(0)
            else:
                self.avgs[altkey].append(1)

        failkey = key.replace('n_', 'n_fail_')
        if not success:
            self.fail_infos[failkey] += 1
            self.fail_infos['n_fail_plans'] += 1
            self.fail_avgs[failkey].append(0)
        else:
            self.fail_avgs[failkey].append(1)

        with self.policy_opt.buf_sizes[key].get_lock():
            self.policy_opt.buf_sizes[key].value += 1


    def get_log_info(self):
        info = {
                'time': time.time() - self.start_t,
                'optimization time': np.mean(self.plan_times),
                'plan length': np.mean(self.plan_horizons),
                'opt duration per ts': np.mean(self.plan_times) / np.mean(self.plan_horizons),
                }

        for key in self.infos:
            info[key] = self.infos[key]

        for key in self.fail_infos:
            info[key] = self.fail_infos[key]

        for key in self.fail_rollout_infos:
            info[key] = self.fail_rollout_infos[key]

        wind = 100
        for key in self.avgs:
            if len(self.avgs[key]):
                info[key+'_avg'] = np.mean(self.avgs[key][-wind:])

        for key in self.fail_avgs:
            if len(self.fail_avgs[key]):
                info[key+'_avg'] = np.mean(self.fail_avgs[key][-wind:])

        for key in self.opt_rollout_info:
            if len(self.opt_rollout_info[key]):
                info[key] = np.mean(self.opt_rollout_info[key][-wind:])

        if len(self.init_costs): info['mp initial costs'] = np.mean(self.init_costs[-wind:])
        if len(self.rolled_costs): info['mp rolled out costs'] = np.mean(self.rolled_costs[-wind:])
        if len(self.final_costs): info['mp optimized costs'] = np.mean(self.final_costs[-wind:])
        return info #self.log_infos


    def write_log(self):
        with open(self.motion_log, 'a+') as f:
            info = self.get_log_info()
            pp_info = pprint.pformat(info, depth=60)
            f.write(str(pp_info))
            f.write('\n\n')

