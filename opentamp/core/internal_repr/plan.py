from .action import Action
import numpy as np
import argparse
import logging
import copy

import torch
import torch.distributions as dist

# import pyro
# import pyro.distributions as dist
# import pyro.poutine as poutine
# from pyro.infer import MCMC, NUTS, HMC, Trace_ELBO, TraceEnum_ELBO, SVI


MAX_PRIORITY = 3

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

class Plan(object):
    """
    A plan has the following.

    params: dictionary of plan parameters, mapping name to object
    actions: list of Actions
    horizon: total number of timesteps for plan

    This class also defines methods for executing actions in simulation using the chosen viewer.
    """
    IMPOSSIBLE = "Impossible"

    def __init__(self, params, actions, horizon, env, determine_free=True, observation_model=None, sess=None):
        self.params = params
        self.belief_params, self.belief_inds = self.build_belief_params()  # fix order for random params
        self.backup = params
        self.actions = actions
        self.horizon = horizon
        self.observation_model = observation_model
        # self.joint_belief = [None] * self.horizon  # overall belief vectors, over time, to support global observation models
        self.log_prob_list = []
        self.time = np.zeros((1, horizon))
        self.env = env
        self.initialized = False
        self._free_attrs = {}
        self._saved_free_attrs = {}
        self.sampling_trace = []
        self.hl_preds = []
        self.start = 0
        self.num_belief_samples = 400
        self.num_warmup_steps = 50
        self.mc_lock = None
        if determine_free:
            self._determine_free_attrs()

    def build_belief_params(self):
        belief_params = []
        belief_inds = {}
        curr_idx = 0
        for param_key, param in self.params.items():
            if hasattr(param, 'belief'):
                belief_params.append(param)
                belief_inds[param_key] = (curr_idx, curr_idx + param.belief.size)
                curr_idx += param.belief.size
        return belief_params, belief_inds
    
    @staticmethod
    def create_plan_for_preds(preds, env):
        ## preds is a list of pred, negated
        ## constructs a plan with a single action that
        ## enforces all the preds
        p_dicts = []
        params = set()
        for p, neg in preds:
            p_dicts.append({'pred': p, 'hl_info': 'pre', 'active_timesteps': (0, 0),
                            'negated': neg})
            params = params.union(p.params)
        params = list(params)
        a = Action(0, 'dummy', (0,0), params, p_dicts)
        param_dict = dict([(p.name, p) for p in params])
        return Plan(param_dict, [a], 1, env, determine_free=False)

    def _determine_free_attrs(self):
        for p in self.params.values():
            for k, v in list(p.__dict__.items()):
                if type(v) == np.ndarray and k not in p._free_attrs:
                    ## free variables are indicated as numpy arrays of NaNs
                    arr = np.zeros(v.shape, dtype='int32')
                    arr[np.isnan(v)] = 1
                    p._free_attrs[k] = arr

    def has_nan(self, active_ts = None):
        if not active_ts:
            active_ts = (0, self.horizon-1)

        for p in self.params.values():
            for k, v in list(p.__dict__.items()):
                if type(v) == np.ndarray:
                    if p.is_symbol() and np.any(np.isnan(v)):
                        print('Nan found in', p.name, k, v)
                        return True
                    if not p.is_symbol() and np.any(np.isnan(v[:, active_ts[0]:active_ts[1]+1])):
                        print('Nan found in', p.name, k, v)
                        return True
        return False

    def backup_params(self):
        for p in self.params:
            self.backup[p] = self.params[p].copy(self.horizon)

    def restore_params(self):
        for p in self.params.values():
            for attr in p._free_attrs:
                p._free_attrs[attr][:] = self.backup[p.name]._free_attrs[attr][:]
                getattr(p, attr)[:] = getattr(self.backup[p.name], attr)[:]

    def save_free_attrs(self):
        for p in self.params.values():
            p.save_free_attrs()

    def restore_free_attrs(self):
        for p in self.params.values():
            p.restore_free_attrs()

    def get_free_attrs(self):
        free_attrs = {}
        for p in self.params.values():
            free_attrs[p] = p.get_free_attrs()
        return free_attrs

    def store_free_attrs(self, attrs):
        for p in self.params.values():
            p.store_free_attrs(attrs[p])

    def freeze_up_to(self, t, exclude_types=[]):
        for p in self.params.values():
            skip = False
            for excl in exclude_types:
                if excl in p.get_type(True):
                    skip = True
                    continue
            if skip: continue
            p.freeze_up_to(t)

    def freeze_actions(self, anum):
        for i in range(anum):
            st, et = self.actions[i].active_timesteps
            for param in self.actions[i].params:
                if param.is_symbol():
                    for attr in param._free_attrs:
                        param._free_attrs[attr][:,0] = 0.
            for param in list(self.params.values()):
                if param.is_symbol(): continue
                for attr in param._free_attrs:
                    param._free_attrs[attr][:,st:et+1] = 0.
        for param in list(self.params.values()):
            if param.is_symbol(): continue
            for attr in param._free_attrs:
                param._free_attrs[attr][:,0] = 0.

    def execute(self):
        raise NotImplementedError

    def get_param(self, pred_type, target_ind, partial_assignment = None,
                  negated=False, return_preds=False):
        """
        get all target_ind parameters of the given predicate type
        partial_assignment is a dict that maps indices to parameter
        """
        if partial_assignment is None:
            partial_assignment = {}
        res = []
        if return_preds:
            preds = []
        for p in self.get_preds(incl_negated = negated):
            has_partial_assignment = True
            if p.get_type() != pred_type: continue
            for idx, v in list(partial_assignment.items()):
                if p.params[idx] != v:
                    has_partial_assignment = False
                    break
            if has_partial_assignment:
                res.append(p.params[target_ind])
                if return_preds: preds.append(p)
        res = np.unique(res)
        if return_preds:
            return res, np.unique(preds)
        return res

    def get_preds(self, incl_negated=True):
        res = []
        for a in self.actions:
            if incl_negated:
                res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state'])
            else:
                res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state' and not p['negated']])

        return res

    #@profile
    def get_failed_pred(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3, incl_negated=True, hl_ignore=False):
        ## just return the first one for now
        t_min = self.horizon+1
        pred = None
        negated = False
        for action in self.actions:
            if active_ts is None:
                st, et = action.active_timesteps[0], action.active_timesteps[1]
            else:
                st, et = max(action.active_timesteps[0], active_ts[0]), min(action.active_timesteps[1], active_ts[1])

            a_st, a_et = action.active_timesteps
            if et > st:
                if a_st >= et: continue
                if a_et <= st: continue

            for pr in range(priority+1):
                for n, p, t in self.get_failed_preds(active_ts=(st,et), priority=pr, tol=tol, incl_negated=incl_negated):
                    if t < t_min and (not hl_ignore or not p.hl_ignore):
                        t_min = t
                        pred = p
                        negated = n
                if pred is not None:
                    return negated, pred, t_min
                
        ## for effects of nondeterministic actions, failure is the final t/s of latest nondeterministic action
        max_nondet_t = 0
        for action in self.actions:
            ## if action is nondeterministic and predicate is an effect of this nondeterministic action
            if action.non_deterministic and \
                (pred in action.preds and pred.active_range[0] > action.active_timesteps[0]):
                
                act_end = action.active_timesteps[1]

                if act_end >= max_nondet_t:
                    max_nondet_t = act_end

                if max_nondet_t < t_min and (not hl_ignore or not p.hl_ignore):
                    t_min = action.active_timesteps[1]

        return negated, pred, t_min

    #@profile
    def get_failed_preds(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3, incl_negated=True):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        failed = []
        for a in self.actions:
            st, et = a.active_timesteps
            if active_ts[1] > active_ts[0]:
                if st >= active_ts[1]: continue
                if et <= active_ts[0]: continue
            failed.extend(a.get_failed_preds(active_ts, priority, tol=tol, incl_negated=incl_negated))
        return failed

    def get_failed_preds_by_action(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        failed = []
        for a in self.actions:
            failed.append(a.get_failed_preds(active_ts, priority, tol=tol))
        return failed

    def get_failed_preds_by_type(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        failed = []
        for a in self.actions:
            failed.extend(a.get_failed_preds_by_type(active_ts, priority, tol=tol))
        return failed

    def satisfied(self, active_ts=None):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        success = True
        for a in self.actions:
            success &= a.satisfied(active_ts)
        return success

    def get_pr_preds(self, ts, priority):
        res = []
        for t in ts:
            for a in self.actions:
                res.extend(a.get_pr_preds(ts, priority))
        return res

    def get_active_preds(self, t):
        res = []
        for a in self.actions:
            start, end = a.active_timesteps
            if start <= t and end >= t:
                res.extend(a.get_active_preds(t))
        return res

    def check_cnt_violation(self, active_ts=None, priority = MAX_PRIORITY, tol = 1e-3):
        if active_ts is None:
            active_ts = (0, self.horizon-1)
        preds = [(negated, pred, t) for negated, pred, t in self.get_failed_preds(active_ts=active_ts, priority = priority, tol = tol)]
        cnt_violations = []
        for negated, pred, t in preds:
            viol = np.max(pred.check_pred_violation(t, negated=negated, tol=tol))
            cnt_violations.append(viol)
            if np.isnan(viol):
                print((negated, pred, t, 'NAN viol'))
            # print ("{}-{}\n".format(pred.get_type(), t), cnt_violations[-1])

        return cnt_violations

    def check_total_cnt_violation(self, active_ts=None, tol=1e-3):
        if active_ts is None:
            active_ts = (0, self.horizon-1)
        failed_preds = self.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        cost = 0
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
                    # if np.any(np.isnan(viol)):
                    #     print('Nan constr violation for {0} at ts {1}'.format(failed, t))

                    if viol is not None:
                        cost += np.max(viol)
                except:
                    pass
        return cost

    def prefix(self, fail_step):
        """
            returns string representation of actions prior to fail_step
        """
        pre = []
        for act in self.actions:
            if act.active_timesteps[1] < fail_step or (act.active_timesteps[1] <= fail_step and act.non_deterministic):
                act_str = str(act).split()
                act_str = " ".join(act_str[:2] + act_str[4:]).upper()
                pre.append(act_str)
        return pre

    def get_plan_str(self):
        """
            return the corresponding plan str
        """
        plan_str = []
        for a in self.actions:
            plan_str.append(str(a))
        return plan_str

    def find_pred(self, pred_name):
        res = []
        for a in self.actions:
            res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state' and p['pred'].get_type() == pred_name])
        return res

    def fill(self, plan, amin=0, amax=None):
        """
            fill self with trajectory from plan
        """
        if amax < 0 : return
        if amax is None:
            amax = len(self.actions)-1
        active_ts = self.actions[amin].active_timesteps[0], self.actions[amax].active_timesteps[1]
        for pname, param in self.params.items():
            if pname not in plan.params:
                raise AttributeError('Reference plan does not contain {0}'.format(pname))
            param.fill(plan.params[pname], active_ts)
        self.start = amax

    def get_values(self):
        vals = {}
        for pname, param in self.params.items():
            for attr in param._free_attrs:
                vals[pname, attr] = getattr(param, attr).copy()
        return vals

    def store_values(self, vals):
        for param, attr in vals:
            getattr(self.params[param], attr)[:] = vals[param, attr]

    def set_to_time(self, ts):
        for param in self.params.values():
            param.set_to_time(ts)

    def set_observation_model(self, observation_model):
        self.observation_model = observation_model
    
    # def set_max_likelihood_obs(self, max_likelihood_obs):
    #     self.max_likelihood_obs = max_likelihood_obs

    # def sample_priors(self):
    #     prior_samples = []
    #     for param in self.belief_params:
    #         prior_samples.append(param.belief.dist.sample_n(self.num_belief_samples * param.belief.size))
    #     return prior_samples  # all sampled tensors

    ## approximates empirical distribution in-dist with beliefs from the prior timestep
    ## adapted from Pyro tutorial part 1
    # def fit_estimation(self):
        # breakpoint()

        # def model(data):
        #     pyro.sample("belief_samp", dist.Empirical(data, torch.ones((data.shape[0],))))
        
        # setup the optimizer

    def construct_current_obs_vec(self):
        ## find maximal belief_index
        max_belief_idx = 0
        for param in self.belief_params:
            if self.belief_inds[param.name][1] > max_belief_idx:
                max_belief_idx = self.belief_inds[param.name][1]

        obs_vec = torch.zeros((max_belief_idx,))

        for param in self.belief_params:
            obs_vec[self.belief_inds[param.name][0]: self.belief_inds[param.name][1]] = self.observation_model.get_active_planned_observations()[param.name]
        
        return obs_vec

    def construct_global_belief_vec(self, vals):
        ## find maximal belief_index
        max_belief_idx = 0
        for param in self.belief_params:
            if self.belief_inds[param.name][1] > max_belief_idx:
                max_belief_idx = self.belief_inds[param.name][1]

        global_belief_vec = torch.zeros((max_belief_idx,))

        for param in self.belief_params:
            global_belief_vec[self.belief_inds[param.name][0]: self.belief_inds[param.name][1]] = vals[param.name]
        
        return global_belief_vec
    
    def construct_random_prior_init(self):
        ## constructs random
        vals = {}

        for param in self.belief_params:
            vals[param.name] = param.belief.dist.sample()

        return vals
    
    def make_full_prefix_obs(self, past_obs, obs, active_ts):
        full_prefix_obs = {}
        for po_ts in past_obs:
            for b_item in obs.keys():
                full_prefix_obs[b_item+'.'+str(po_ts[0])] = past_obs[po_ts][b_item].to(DEVICE)
        for b_item in obs.keys():
            full_prefix_obs[b_item+'.'+str(active_ts[0])] = obs[b_item].to(DEVICE)

        return full_prefix_obs
    
    def set_mc_lock(self, lock):
        global mc_lock
        mc_lock = lock
    
    # wrapper for particle filter called in 
    def particle_filter(self, active_ts, provided_goal=None, past_obs={}):
        print('Reached Filter')
        obs = self.observation_model.forward_model(copy.deepcopy(self.params), active_ts, provided_state=provided_goal)
        print('Passed Forward Model')
        particles = self.observation_model.filter_samples(copy.deepcopy(self.params), active_ts, obs) ## TODO method that generates list of likely particles
        print('Passed Particle Filter')
        return particles, obs

    ## based off of hmm example from pyro docs
    # def sample_mcmc_run(self, active_ts, provided_goal=None, past_obs={}):                
    #     ## fit a parametric approximation to the current belief state
    #     # mc_lock.acquire()

    #     # print('Acquired lock')
        
    #     self.observation_model.fit_approximation(copy.deepcopy(self.params))
    #     ## get random observation through the forward model
    #     obs = self.observation_model.forward_model(copy.deepcopy(self.params), active_ts, provided_state=provided_goal)
        
    #     print('Provided goal: ', provided_goal)
    #     print('New observation: ', obs)
    #     print('Past observation: ', past_obs)
    #     print('Active timesteps: ', active_ts)

    #     # if provided_obs:
    #     #     ## get the assumed observation in planning (typically in replans)  
    #     #     obs = provided_obs
    #     # else:
    #     #     ## get the observations assumed to be seen through planning (sampled or MLO)
    #     #     obs = self.observation_model.get_active_planned_observations()
        
    #     full_prefix_obs = self.make_full_prefix_obs(past_obs, obs, active_ts)
    #     full_samps = {}
    #     for _ in range(1):
    #         print('Done Iter')
    #         pyro.clear_param_store()

    #         ## create a model conditioned on the observed data in the course of executing plan
    #         conditional_model = poutine.condition(self.observation_model.forward_model, data=full_prefix_obs)

    #         ## No-U Turn Sampling kernel
    #         kernel = NUTS(conditional_model, full_mass=True)

    #         ## get a vector of overall observations, as a warmstart for MCMC
    #         # global_vec = self.construct_global_belief_vec(provided_goal)
            
    #         if not provided_goal:
    #             breakpoint()

    #         print('Conditioning on: ', full_prefix_obs)

    #         rand_init = self.construct_random_prior_init()

    #         print('Init inference to: ', rand_init)

    #         ## initialize and run MCMC (conditions on the active observation)
    #         mcmc = MCMC(
    #             kernel,
    #             num_samples=self.num_belief_samples,
    #             warmup_steps=self.num_warmup_steps,
    #             num_chains=1,
    #             initial_params={'belief_'+id: rand_init[id] for id in rand_init.keys()},
    #             disable_progbar=True
    #         )

    #         mcmc.run(copy.deepcopy(self.params), active_ts, past_obs=past_obs)
    #         mcmc.summary(prob=0.95)  # for diagnostics

    #         # print('Releasing lock')
    #         # mc_lock.release()
            
    #         samps =  mcmc.get_samples()
    #         for key in samps:
    #             if key in full_samps:
    #                 full_samps[key] = torch.cat((full_samps[key], samps[key]), dim=0)
    #             else:
    #                 full_samps[key] = samps[key]

    #     return full_samps, obs

    # def initialize_obs(self, anum=0, override_obs=None):
    #     for param in self.belief_params:
    #         if override_obs:
    #             param.pose[:, step] = override_obs[param.name]
    #         else:
    #             param.pose[:, step] = self.observation_model(copy.deepcopy(self.params), self.actions[anum].active_timesteps)[param.name]

    ## called once per high-level action execution
    def filter_beliefs(self, active_ts, provided_goal=None, past_obs={}):

        # max-likelihood observation feeds back on object here
        global_samples, plan_obs = self.particle_filter(active_ts, provided_goal=provided_goal, past_obs=past_obs) ## forward model + true observations
        # global_samples, plan_obs = self.sample_mcmc_run(active_ts, provided_goal=provided_goal, past_obs=past_obs)

        for key in global_samples:
            if len(global_samples[key].shape) == 1:
                global_samples[key] = global_samples[key].unsqueeze(dim=1)

        for key in global_samples:
            assert len(global_samples[key].shape) == 2
        
        # update all of the param samples objects, as induced from the belief object
        running_idx = 0
        for param in self.belief_params:
            new_samp = torch.cat((param.belief.samples, torch.unsqueeze(global_samples['belief_' + param.name], 2)), dim=2)
            param.belief.samples = new_samp
            running_idx += param.belief.size

        return plan_obs
        

    ## for now, just propogates same belief until end of action -- later, introduce a self.drift_model, similarly to self.observation_model 
    ## NOTE: assume can only use up until penultimate belief state in planner for now
    def rollout_beliefs(self, active_ts):
        for param in self.belief_params:
            for _ in range(active_ts[0], active_ts[1]-1):
                new_samp = torch.cat((param.belief.samples, param.belief.samples[:, :, -1:]), dim=2)
                param.belief.samples = new_samp
