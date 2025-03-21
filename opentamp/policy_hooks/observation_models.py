## set of methods taking forward samples for observation
import torch
import torch.distributions as dist
# import torch
# import torch.distributions as dist
from opentamp.core.util_classes.custom_dist import CustomDist
# import torch.poutine as poutine
# from torch.infer import MCMC, NUTS, SVI, TraceEnum_ELBO, config_enumerate
# from torch.infer.autoguide import AutoDelta
import numpy as np
import os

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
# print('USED DEVICE FOR MCMC INFERENCE IS: ', DEVICE)

class ObservationModel(object):
    approx_params : None  ## parameters into the parametric approximation for the current belief state
    active_planned_observations : None  ## dict of parameter names, mapping to the current observed values
    past_obs : {} ## dict of past observations to condition on, indexed {timestep: observation}

    ## get the observation that is currently active / being planned over
    def get_active_planned_observations(self):
        return self.active_planned_observations
    
    ## get the current parameters for training the (variational) approximation to a given belief state
    def get_approx_params(self):
        return self.approx_params
    
    ## get the observation that is currently active / being planned over
    def set_active_planned_observations(self, planned_obs):
        self.active_planned_observations = planned_obs

    def get_obs_likelihood(self, obs):
        raise NotImplementedError

    ## a callable giving a parametric form for the approximation to the belief posterior, used subsequently in MCMC
    def approx_model(self, data):
        raise NotImplementedError
    
    ## a callable fiting the approx_params dict on a set of samples
    def fit_approximation(self, params):
        raise NotImplementedError

    ## a callable representing the forward model
    def forward_model(self, params, active_ts, provided_state=None):
        raise NotImplementedError
    
# class PointerObservationModel(ObservationModel):
#     def __init__(self):
#         # uninitialized parameters
#         self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
#         self.active_planned_observations = {'target1': torch.empty((2,)).detach()}
    
#     @config_enumerate
#     def approx_model(self, data):
#         ## Global variable (weight on either cluster)
#         weights = torch.sample("weights"+str(os.getpid()), dist.Dirichlet(5 * torch.ones(2)))

#         ## Different Locs and Scales for each
#         with torch.plate("components"+str(os.getpid()), 2):
#             ## Uninformative prior on locations
#             locs = torch.sample("locs"+str(os.getpid()), dist.MultivariateNormal(torch.tensor([3.0, 0.0]), 20.0 * torch.eye(2)))
#             scales = torch.sample("scales"+str(os.getpid()), dist.LogNormal(0.0, 10.0))

#         with torch.plate("data"+str(os.getpid()), len(data)):
#             ## Local variables
#             assignment = torch.sample("mode_assignment"+str(os.getpid()), dist.Categorical(weights))
#             stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(100, 1, 1))
#             stack_scale = torch.tile(scales[assignment].unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
#             cov_tensor = (stack_eye * stack_scale)
#             torch.sample("belief_global"+str(os.getpid()), dist.MultivariateNormal(locs[assignment], cov_tensor), obs=data)
 

#     def forward_model(self, params, active_ts, provided_state=None, past_obs={}):        
#         ray_width = np.pi / 4  ## has 45-degree field of view on either side

#         def is_in_ray(a_pose, target):
#             if target[0] >= 0:
#                 return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= ray_width
#             else:
#                 return np.abs(np.arctan(target[1]/target[0]) - (a_pose - np.pi)) <= ray_width
        
#         ## construct Gaussian mixture model using the current approximation
#         cat_dist = dist.Categorical(probs=self.approx_params['weights'])
#         stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
#         stack_scale = torch.tile(self.approx_params['scales'].unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
#         cov_tensor = stack_eye * stack_scale
#         batched_multivar = dist.MultivariateNormal(loc=self.approx_params['locs'], covariance_matrix=cov_tensor)
#         mix_dist = dist.MixtureSameFamily(cat_dist, batched_multivar)

#         if provided_state is not None:
#             ## overrides the current belief sample with a true state
#             b_global_samp = provided_state['target1']
#         else:
#             ## sample from current Gaussian mixture model
#             #b_global_samp = torch.sample('belief_global', mix_dist)
#             b_global_samp = torch.sample('belief_target1', mix_dist)
        
        
#         samps = {}

#         if is_in_ray(params['pr2'].pose[0,active_ts[1]-1], b_global_samp.detach()):
#             ## sample around the true belief, with extremely low variation / error
#             samps['target1'] = torch.sample('target1', dist.MultivariateNormal(b_global_samp.float(), 0.01 * torch.eye(2)))
#         else:
#             ## sample from prior dist -- have no additional knowledge, don't read it
#             samps['target1'] = torch.sample('target1', dist.MultivariateNormal(torch.zeros((2,)), 0.01 * torch.eye(2)))

#         # return tensors on CPU for compatib
#         for samp in samps:
#             samps[samp].to('cpu')

#         return samps

#     def fit_approximation(self, params):        
#         def init_loc_fn(site):
#             K=2
#             data=params['target1'].belief.samples[:, :, -1]
#             if site["name"] == "weights"+str(os.getpid()):
#                 # Initialize weights to uniform.
#                 return (torch.ones(2) / 2)
#             if site["name"] == "scales"+str(os.getpid()):
#                 return torch.ones(2)
#             if site["name"] == "locs"+str(os.getpid()):
#                 return torch.tensor([[3., 3.], [3., -3.]])
#             raise ValueError(site["name"])


#         def initialize():
#             pass
#             ## clear torch optimization context, for 

#         # Choose the best among 100 random initializations.
#         # loss, seed = min((initialize(seed), seed) for seed in range(100))
#         initialize()
#         # print(f"seed = {seed}, initial_loss = {loss}")

#         global_guide = AutoDelta(
#             poutine.block(self.approx_model, expose=["weights"+str(os.getpid()), "locs"+str(os.getpid()), "scales"+str(os.getpid())]),
#             init_loc_fn=init_loc_fn,
#         )
#         adam_params = {"lr": 0.01, "betas": [0.99, 0.3]}
#         optimizer = torch.optim.Adam(adam_params)

#         svi = SVI(self.approx_model, global_guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

#         ## setup the inference algorithm
#         nsteps = 200  ## NOTE: causes strange bugs when run too long (weights concentrate to 1)

#         ## do gradient steps, TODO update with genreal belief signature 
#         for i in range(nsteps):
#             loss = svi.step(params['target1'].belief.samples[:, :, -1])
#             # print(global_guide(params['target1'].belief.samples[:, :, -1]))

#         pars = global_guide(params['target1'].belief.samples[:, :, -1])
        
#         new_p = {}

#         ## need to detach for observation_model to be serializable
#         for p in ['weights', 'locs', 'scales']:
#             new_p[p] = pars[p+str(os.getpid())].detach()

#         self.approx_params = new_p


# class NoVIPointerObservationModel(ObservationModel):
#     def __init__(self):
#         # uninitialized parameters
#         self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
#         self.active_planned_observations = {'target1': torch.empty((2,)).detach()}
    
#     def is_in_ray(self, a_pose, target):
#         ray_width = np.pi / 4  ## has 45-degree field of view on either side

#         if target[0] >= 0:
#             return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= ray_width
#         else:
#             return np.abs(np.arctan(target[1]/target[0]) - (a_pose - np.pi)) <= ray_width


#     def approx_model(self, data):
#         pass

#     def get_unnorm_obs_log_likelihood(self, params, prefix_obs, fail_ts):
#         log_likelihood = torch.zeros((params['target1'].belief.samples.shape[0],))

#         for idx in range(params['target1'].belief.samples.shape[0]):
#             ## initialize log_likelihood to prior probability

#             log_likelihood[idx] = params['target1'].belief.dist.log_prob(params['target1'].belief.samples[idx, :, fail_ts]).sum().item()

#             ## add in terms for the forward model
#             for obs_active_ts in prefix_obs:
#                 if self.is_in_ray(params['pr2'].pose[0,obs_active_ts[1]-1], params['target1'].belief.samples[idx,:,fail_ts]):
#                     ## sample around the true belief, with extremely low variation / error
#                     log_likelihood[idx] += dist.MultivariateNormal(params['target1'].belief.samples[idx,:,fail_ts], (0.01 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['target1'])
#                 else:
#                     ## sample from prior dist -- have no additional knowledge, don't read it
#                     log_likelihood[idx] += dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['target1'])

#         return log_likelihood

#     def forward_model(self, params, active_ts, provided_state=None, past_obs={}):        
#         if provided_state is not None:
#             ## overrides the current belief sample with a true state
#             b_global_samp = provided_state['target1'].to(DEVICE)
#         else:
#             ## sample from current Gaussian mixture model
#             b_global_samp = torch.sample('belief_target1', params['target1'].belief.dist).to(DEVICE)
        
#         ## sample through strict prefix of current obs
#         for obs_active_ts in past_obs:
#             if self.is_in_ray(params['pr2'].pose[0,obs_active_ts[1]-1], b_global_samp.detach().to('cpu')):
#                 ## sample around the true belief, with extremely low variation / error
#                 torch.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.01 * torch.eye(2)).to(DEVICE)))
#             else:
#                 ## sample from prior dist -- have no additional knowledge, don't read it
#                 torch.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))

        
#         ## get sample for current timestep, record and return
#         samps = {}

#         if self.is_in_ray(params['pr2'].pose[0,active_ts[1]-1], b_global_samp.detach().to('cpu')):
#             ## sample around the true belief, with extremely low variation / error
#             samps['target1'] = torch.sample('target1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))
#         else:
#             ## sample from prior dist -- have no additional knowledge, don't read it
#             samps['target1'] = torch.sample('target1.'+str(active_ts[0]), dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))

#         # return tensors on CPU for compatib
#         for samp in samps:
#             samps[samp].to('cpu')

#         return samps

#     ## no VI in the pointer observation
#     def fit_approximation(self, params):        
#         pass



# class NoVIObstacleObservationModel(ObservationModel):
#     def __init__(self):
#         # uninitialized parameters
#         self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
#         self.active_planned_observations = {'obs1': torch.empty((2,)).detach()}
#         self.obs_dist = 3.0
    
#     def is_in_ray(self, a_pose, target):
#         ray_width = np.pi / 4  ## has 45-degree field of view on either side
#         adjust_a_pose = (a_pose+np.pi/2)%(2*np.pi) -np.pi/2
#         if target[0] > 0:
#             return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose)) <= ray_width
#         else:
#             return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose - np.pi)) <= ray_width

#     def approx_model(self, data):
#         pass

#     def get_unnorm_obs_log_likelihood(self, params, prefix_obs, fail_ts):
#         log_likelihood = torch.zeros((params['obs1'].belief.samples.shape[0], ))

#         for idx in range(params['obs1'].belief.samples.shape[0]):
#             ## initialize log_likelihood to prior probability

#             log_likelihood[idx] = params['obs1'].belief.dist.log_prob(params['obs1'].belief.samples[idx, :, fail_ts]).sum().item()

#             ## add in terms for the forward model
#             for obs_active_ts in prefix_obs:
#                 obs_vec = params['obs1'].belief.samples[idx,:,fail_ts]
#                 mod_obs_vec = torch.sign(obs_vec) * (torch.abs(obs_vec))
#                 rel_vec = mod_obs_vec - params['pr2'].pose[:,obs_active_ts[1]-1]
#                 if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                     ## sample around the true belief, with extremely low variation / error
#                     log_likelihood[idx] += dist.MultivariateNormal(params['obs1'].belief.samples[idx,:,fail_ts], (0.001 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['obs1'])
#                 else:
#                     ## sample from prior dist -- have no additional knowledge, don't read it
#                     log_likelihood[idx] += dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['obs1'])

#         return log_likelihood

#     def forward_model(self, params, active_ts, provided_state=None, past_obs={}):                
#         if provided_state:
#             ## overrides the current belief sample with a true state
#             b_global_samp = np.sign(provided_state['obs1'].to(DEVICE)) * (np.abs(provided_state['obs1'].to(DEVICE)))
#         else:
#             ## sample from prior dist
#             b_global_samp = torch.sample('belief_obs1', params['obs1'].belief.dist).to(DEVICE)

#         ## sample through strict prefix of current obs
#         for obs_active_ts in past_obs:
#             mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
#             rel_vec = mod_obs - params['pr2'].pose[:,obs_active_ts[1]-1]
#             if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                 ## sample around the true belief, with extremely low variation / error
#                 torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.001 * torch.eye(2)).to(DEVICE)))
#             else:
#                 ## sample from prior dist -- have no additional knowledge, don't read it
#                 torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

#         ## get sample for current timestep, record and return
#         samps = {}

#         mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
#         rel_vec = mod_obs  - params['pr2'].pose[:,active_ts[1]-1]
#         if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#             ## sample around the true belief, with extremely low variation / error
#             samps['obs1'] = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))
#         else:
#             ## sample from prior dist -- have no additional knowledge, don't read it
#             samps['obs1'] = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

#         # return tensors on CPU for compatib
#         for samp in samps:
#             samps[samp].to('cpu')

#         return samps

#     ## no VI in the pointer observation
#     def fit_approximation(self, params):        
#         pass


class ParticleFilterTargetObservationModel(ObservationModel):
    def __init__(self):
        # uninitialized parameters
        self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
        self.obs_dist = 2.0
        self.num_obstacles = 1
        self.active_planned_observations = {'obs'+str(i+1): torch.empty((2,)).detach() for i in range(self.num_obstacles)}
        self.low_bound = None
        self.high_bound = None

    
    def is_in_ray(self, a_pose, target):
        ray_width = np.pi / 4  ## has 45-degree field of view on either side
        adjust_a_pose = (a_pose+np.pi/2)%(2*np.pi) -np.pi/2
        if target[0] > 0:
            return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose)) <= ray_width
        else:
            return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose - np.pi)) <= ray_width

    def approx_model(self, data):
        pass

    # artificially selects nearest waypoints
    def get_unnorm_obs_log_likelihood(self, params, preds, prefix_obs, fail_ts):
        log_likelihood = torch.zeros((params['targ'].belief.samples.shape[0], ))

        # determining current room
        for pred in [p for p in preds]:
            if pred.get_type() == 'TargetInRoom':
                self.low_bound = pred.params[1].low_bound
                self.high_bound = pred.params[1].high_bound

        obs_id = 'targ'
        # closest to center of current room

        # rdul_offsets = [np.array([0.0, 0.0]), np.array([0.0, -1.0]), np.array([-1.0,-1.0]), np.array([-1.0,0.0])]
        # rdul_idx = 0
        # rdul_magnitude = 1

        init_pose = params['rob'].pose[:,fail_ts]

        room_left_corn = [np.array([0.0, 2.5]), np.array([2.5,2.5]), np.array([2.5, 0.]), np.array([0.,0.])]

        curr_skolem_idx = 0
        for i in range(4):
            if ((room_left_corn[i].reshape(2,1) <= init_pose.reshape(2,1)).all() and (init_pose.reshape(2,1) <= (room_left_corn[i] + np.ones(2,) * 2.5).reshape(2,1)).all()):
                curr_skolem_idx = i

        init_skolem_idx = curr_skolem_idx

        rad = 0.25

        while True:
            low_bound = room_left_corn[curr_skolem_idx] + np.ones(2,) * (1.25 - rad)
            high_bound = room_left_corn[curr_skolem_idx] + np.ones(2,) * (1.25 + rad)

            closest_idx_arr = np.argsort(np.linalg.norm(params[obs_id].belief.samples[:, :, fail_ts] - (low_bound + np.ones((2,)) * rad), axis=1))
            
            # for i in range(params['targ'].belief.samples.shape[0]):
            closest_idx = closest_idx_arr[i]
            closest_obs = params[obs_id].belief.samples[closest_idx, :, fail_ts]
            # if not in current room
            if (low_bound.reshape(2,1) <= closest_obs.numpy().reshape(2,1)).all() and (closest_obs.numpy().reshape(2,1) <= high_bound.reshape(2,1)).all():
                log_likelihood[closest_idx] = 1.  ## propose the highest priority towards nearest to waypoint
                return log_likelihood

            # arr_in_low = np.min(low_bound.reshape(1, 2) <= params['targ'].belief.samples[:,:,fail_ts].numpy(), axis=1)
            # arr_in_high = np.min(params['targ'].belief.samples[:,:,fail_ts].numpy() <= high_bound.reshape(1, 2), axis=1)
            # in_room_arr = np.minimum(arr_in_low, arr_in_high)

            # in_room_obs = params[obs_id].belief.samples[in_room_arr, :, fail_ts]
            
            # dists_to_other_obs = np.linalg.norm(in_room_obs - closest_obs, axis=1)
            # closest_to_obs_idx_arr = np.argsort(dists_to_other_obs)

            # j = dists_to_other_obs.shape[0]


            # for j in range(len(closest_to_obs_idx_arr)):
            #     if dists_to_other_obs[closest_to_obs_idx_arr[j]] > 2.0:
            #       break

            
            # for i in range(self.num_obstacles):
            #     obs_id = 'targ'
            #     for idx in range(params[obs_id].belief.samples.shape[0]):
            #         ## initialize log_likelihood to prior probability

            #         log_likelihood[idx] = params[obs_id].belief.dist.log_prob(params[obs_id].belief.samples[idx, :, fail_ts]).sum().item()

            #         ## add in terms for the forward model
            #         for obs_active_ts in prefix_obs:
            #             obs_vec = params[obs_id].belief.samples[idx,:,fail_ts]
            #             mod_obs_vec = torch.sign(obs_vec) * (torch.abs(obs_vec))
            #             rel_vec = mod_obs_vec - params['rob'].pose[:,obs_active_ts[1]-1]
            #             if np.linalg.norm(rel_vec) <= self.obs_dist:
            #                 ## sample around the true belief, with extremely low variation / error
            #                 log_likelihood[idx] += dist.MultivariateNormal(params[obs_id].belief.samples[idx,:,fail_ts], (0.01 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts][obs_id])
            #             else:
            #                 ## sample from prior dist -- have no additional knowledge, don't read it
            #                 log_likelihood[idx] += dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.01 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts][obs_id])

            curr_skolem_idx = (curr_skolem_idx + 1) % 4

            if curr_skolem_idx == init_skolem_idx:
                rad += 0.25

            # if rdul_idx == 0:
            #     rdul_magnitude += 1

        return log_likelihood  # choose arbitrarily

    def forward_model(self, params, active_ts, provided_state=None, past_obs={}):       
        max_obs_dict = {}
        if provided_state:
            obs_id = 'targ'      
            ## overrides the current belief sample with a true state
            b_global_samp = np.sign(provided_state[obs_id].to(DEVICE)) * (np.abs(provided_state[obs_id].to(DEVICE)))

            mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
            rel_vec = mod_obs  - params['rob'].pose[:,active_ts[1]-1]

            angle_diff = (np.arctan2(rel_vec[1], rel_vec[0]) - params['rob'].theta[:,active_ts[1]-1]) % (2 * np.pi)

            if (np.linalg.norm(rel_vec) <= self.obs_dist and ((self.low_bound.reshape(2,1) <= mod_obs.numpy().reshape(2,1)).all() and (mod_obs.numpy().reshape(2,1) <= self.high_bound.reshape(2,1)).all())) and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
                ## sample around the true belief, with extremely low variation / error
                max_obs_dict[obs_id] = dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.01 * torch.eye(2)).sample()
            else:
                ## sample from prior dist -- have no additional knowledge, don't read it
                max_obs_dict[obs_id] = dist.MultivariateNormal((-torch.ones((2,))), 0.01 * torch.eye(2)).sample()

            # return tensors on CPU for compatib
            for samp in max_obs_dict:
                max_obs_dict[samp].to('cpu')

            for i in range(self.num_obstacles):
                obs_id = 'o'+str(i)      
                ## overrides the current belief sample with a true state
                b_global_samp = np.sign(provided_state[obs_id].to(DEVICE)) * (np.abs(provided_state[obs_id].to(DEVICE)))

                mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
                rel_vec = mod_obs  - params['rob'].pose[:,active_ts[1]-1]

                angle_diff = (np.arctan2(rel_vec[1], rel_vec[0]) - params['rob'].theta[:,active_ts[1]-1]) % (2 * np.pi)

                if (np.linalg.norm(rel_vec) <= self.obs_dist and ((self.low_bound.reshape(2,1) <= mod_obs.numpy().reshape(2,1)).all() and (mod_obs.numpy().reshape(2,1) <= self.high_bound.reshape(2,1)).all())) and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
                    ## sample around the true belief, with extremely low variation / error
                    max_obs_dict[obs_id] = dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.01 * torch.eye(2)).sample()
                else:
                    ## sample from prior dist -- have no additional knowledge, don't read it
                    max_obs_dict[obs_id] = dist.MultivariateNormal((-torch.ones((2,))), 0.01 * torch.eye(2)).sample()

                # return tensors on CPU for compatib
                for samp in max_obs_dict:
                    max_obs_dict[samp].to('cpu')

            
            return max_obs_dict
        
        else:
            obs_id = 'targ'  
            ## give observation from nearest particle (maximum-likelihood observation)
            obstacle_particles = params[obs_id].belief.samples[:,:, active_ts[1]-1] ## assume you'll see the current

            closest_idx = np.argmin(np.linalg.norm(obstacle_particles - (params['rob'].pose[:, active_ts[1]-1] + np.array([np.cos(params['rob'].theta[:, active_ts[1]-1]).item(), np.sin(params['rob'].theta[:, active_ts[1]-1]).item()]) * 0.5), axis=1))

            closest_particle = params['targ'].belief.samples[closest_idx, :, active_ts[1]-1]  ## assume you'll see the current proposed target

            rel_vec = closest_particle - params['rob'].pose[:,active_ts[1]-1]

            angle_diff = (np.arctan2(rel_vec[1], rel_vec[0]) - params['rob'].theta[:,active_ts[1]-1]) % (2 * np.pi)

            if (np.linalg.norm(rel_vec) <= self.obs_dist and ((self.low_bound.reshape(2,1) <= closest_particle.numpy().reshape(2,1)).all() and (closest_particle.numpy().reshape(2,1) <= self.high_bound.reshape(2,1)).all())) and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
                tmp_observation = dist.MultivariateNormal(torch.tensor(closest_particle).float(), 0.01 * torch.eye(2)).sample()
            else:
                tmp_observation = dist.MultivariateNormal((-torch.ones((2,))).to(DEVICE), 0.01 * torch.eye(2)).sample() 

            max_obs_dict[obs_id] = tmp_observation

            for i in range(self.num_obstacles): 
                obs_id = 'o'+str(i)  
                # ## give observation from nearest particle (maximum-likelihood observation)
                # obstacle_particles = params[obs_id].value[:,0] ## assume you'll see the current

                # closest_idx = np.argmin(np.linalg.norm(obstacle_particles - (params['rob'].pose[:, active_ts[1]-1] + np.array(np.cos(params['rob'].theta[:, active_ts[1]-1]), np.sin(params['rob'].theta[:, active_ts[1]]))), axis=1))

                curr_targ = closest_particle

                closest_particle = params[obs_id].belief.samples[np.random.randint(params[obs_id].belief.samples.shape[0]), :, active_ts[1]-1]  ## assume you'll see the current proposed target

                # plan to assume these are far -- in reality, always guaranteed to be >= 1.0 apart
                while np.linalg.norm(closest_particle - curr_targ) < 0.7:
                    closest_particle = params[obs_id].belief.samples[np.random.randint(params[obs_id].belief.samples.shape[0]), :, active_ts[1]-1]  ## assume you'll see the current proposed target

                rel_vec = closest_particle - params['rob'].pose[:,active_ts[1]-1]

                angle_diff = (np.arctan2(rel_vec[1], rel_vec[0]) - params['rob'].theta[:,active_ts[1]-1]) % (2 * np.pi)

                if (np.linalg.norm(rel_vec) <= self.obs_dist and ((self.low_bound.reshape(2,1) <= closest_particle.numpy().reshape(2,1)).all() and (closest_particle.numpy().reshape(2,1) <= self.high_bound.reshape(2,1)).all())) and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
                    tmp_observation = dist.MultivariateNormal(torch.tensor(closest_particle).float(), 0.01 * torch.eye(2)).sample()
                else:
                    tmp_observation = dist.MultivariateNormal((-torch.ones((2,))).to(DEVICE), 0.01 * torch.eye(2)).sample()   

                max_obs_dict[obs_id] = tmp_observation            

                # observations = {}
                # obstacle_particles = params[obs_id].belief.samples[:,:, active_ts[1]-1]

                # for idx in range(obstacle_particles.shape[0]):
                #     ## obtain sample observation for this particle
                #     rel_vec = obstacle_particles[idx, :]  - params['rob'].pose[:,active_ts[1]-1]
                #     if np.linalg.norm(rel_vec) <= self.obs_dist:
                #         tmp_observation = dist.MultivariateNormal(torch.tensor(obstacle_particles[idx, :]).float(), 0.01 * torch.eye(2)).sample()
                #     else:
                #         tmp_observation = dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.01 * torch.eye(2)).sample()

                #     # enter observation into dict
                #     is_present = False
                #     for proposal_observation in observations.keys():
                #         if torch.linalg.vector_norm(proposal_observation - tmp_observation) < 0.1:
                #             observations[proposal_observation] += 1
                #             is_present = True
                #             break

                #     if not is_present:
                #         observations[tmp_observation] = 1

                # max_count = None
                # max_obs = None    
                # for obs in observations:
                #     if max_count is None or max_count < observations[obs]:
                #         max_count = observations[obs]                ## give most-likely observation on current particle set
                #         max_obs = obs

                # max_obs_dict[obs_id] = max_obs
            
            return max_obs_dict

        ## sample through strict prefix of current obs
        # for obs_active_ts in past_obs:
        #     mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
        #     rel_vec = mod_obs - params['pr2'].pose[:,obs_active_ts[1]-1]
        #     if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
        #         ## sample around the true belief, with extremely low variation / error
        #         torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.001 * torch.eye(2)).to(DEVICE)))
        #     else:
        #         ## sample from prior dist -- have no additional knowledge, don't read it
        #         torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

        ## get sample for current timestep, record and return
        
    
    def filter_samples(self, params, active_ts, curr_obs):
        particles = {}
        # for i in range(self.num_obstacles):
        obs_id = 'targ'
        obstacle_particles = params[obs_id].belief.samples[:,:,active_ts[1]-1]

        # rejuv_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0.0, 0.0]), covariance_matrix= 0.001* torch.eye(2))
        # rejuv_samples = torch.tensor([obstacle_particles[i,:].detach().numpy() + rejuv_dist.sample_n(obstacle_particles.shape[0])[i,:].detach().numpy() for i in range(obstacle_particles.shape[0])])
        rejuv_samples = torch.tensor([obstacle_particles[i,:].detach().numpy() for i in range(obstacle_particles.shape[0])])

        probs = torch.zeros((rejuv_samples.shape[0], ))

        ## TODO: create an array of conditional probabilities p(o|s)
        for idx in range(rejuv_samples.shape[0]):
            rel_vec = rejuv_samples[idx, :]  - params['rob'].pose[:,active_ts[1]-1]

            angle_diff = (np.arctan2(rel_vec[1], rel_vec[0]) - params['rob'].theta[:,active_ts[1]-1]) % (2 * np.pi)

            if np.linalg.norm(rel_vec) <= self.obs_dist and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
                probs[idx] = torch.exp(dist.MultivariateNormal(rejuv_samples[idx, :].float(), 0.01 * torch.eye(2)).log_prob(curr_obs[obs_id]))
            else: 
                probs[idx] = torch.exp(dist.MultivariateNormal(-torch.ones((2,)), 0.01 * torch.eye(2)).log_prob(curr_obs[obs_id]))
        
        try:
            select_idx = torch.multinomial(probs, num_samples=rejuv_samples.shape[0], replacement=True)

            particles['belief_'+obs_id] = rejuv_samples[select_idx, :]
        except:
            print('WARN -- particle filtering encountered numerical error')
            ## if, e.g., the probs array is numerically challenged, then simply don't alter the particles upon this observation
            particles['belief_'+obs_id] = obstacle_particles

        for i in range(self.num_obstacles):
            obs_id = 'o'+str(i)
            obstacle_particles = params[obs_id].belief.samples[:,:,active_ts[1]-1]

            # rejuv_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0.0, 0.0]), covariance_matrix= 0.001* torch.eye(2))
            # rejuv_samples = torch.tensor([obstacle_particles[i,:].detach().numpy() + rejuv_dist.sample_n(obstacle_particles.shape[0])[i,:].detach().numpy() for i in range(obstacle_particles.shape[0])])
            rejuv_samples = torch.tensor([obstacle_particles[i,:].detach().numpy() for i in range(obstacle_particles.shape[0])])

            probs = torch.zeros((rejuv_samples.shape[0], ))

            ## TODO: create an array of conditional probabilities p(o|s)
            for idx in range(rejuv_samples.shape[0]):
                rel_vec = rejuv_samples[idx, :]  - params['rob'].pose[:,active_ts[1]-1]

                angle_diff = (np.arctan2(rel_vec[1], rel_vec[0]) - params['rob'].theta[:,active_ts[1]-1]) % (2 * np.pi)

                if np.linalg.norm(rel_vec) <= self.obs_dist and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
                    probs[idx] = torch.exp(dist.MultivariateNormal(rejuv_samples[idx, :].float(), 0.01 * torch.eye(2)).log_prob(curr_obs[obs_id]))
                else: 
                    probs[idx] = torch.exp(dist.MultivariateNormal(-torch.ones((2,)), 0.01 * torch.eye(2)).log_prob(curr_obs[obs_id]))
            
            try:
                select_idx = torch.multinomial(probs, num_samples=rejuv_samples.shape[0], replacement=True)

                particles['belief_'+obs_id] = rejuv_samples[select_idx, :]
            except:
                print('WARN -- particle filtering encountered numerical error')
                ## if, e.g., the probs array is numerically challenged, then simply don't alter the particles upon this observation
                particles['belief_'+obs_id] = obstacle_particles
        
        return particles

    ## no VI in the pointer observation
    def fit_approximation(self, params):        
        pass

class ParticleFilterObstacleObservationModel(ObservationModel):
    def __init__(self):
        # uninitialized parameters
        self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
        self.obs_dist = 6.0
        self.num_obstacles = 1
        self.active_planned_observations = {'obs'+str(i+1): torch.empty((2,)).detach() for i in range(self.num_obstacles)}

    
    def is_in_ray(self, a_pose, target):
        ray_width = np.pi / 4  ## has 45-degree field of view on either side
        adjust_a_pose = (a_pose+np.pi/2)%(2*np.pi) -np.pi/2
        if target[0] > 0:
            return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose)) <= ray_width
        else:
            return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose - np.pi)) <= ray_width

    def approx_model(self, data):
        pass

    def get_unnorm_obs_log_likelihood(self, params, prefix_obs, fail_ts):
        log_likelihood = torch.zeros((params['obs1'].belief.samples.shape[0], ))
        for i in range(self.num_obstacles):
            obs_id = 'obs'+str(i+1)
            for idx in range(params[obs_id].belief.samples.shape[0]):
                ## initialize log_likelihood to prior probability

                log_likelihood[idx] = params[obs_id].belief.dist.log_prob(params[obs_id].belief.samples[idx, :, fail_ts]).sum().item()

                ## add in terms for the forward model
                for obs_active_ts in prefix_obs:
                    obs_vec = params[obs_id].belief.samples[idx,:,fail_ts]
                    mod_obs_vec = torch.sign(obs_vec) * (torch.abs(obs_vec))
                    rel_vec = mod_obs_vec - params['pr2'].pose[:,obs_active_ts[1]-1]
                    if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
                        ## sample around the true belief, with extremely low variation / error
                        log_likelihood[idx] += dist.MultivariateNormal(params[obs_id].belief.samples[idx,:,fail_ts], (0.01 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts][obs_id])
                    else:
                        ## sample from prior dist -- have no additional knowledge, don't read it
                        log_likelihood[idx] += dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.01 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts][obs_id])

        return log_likelihood

    def forward_model(self, params, active_ts, provided_state=None, past_obs={}):       
        max_obs_dict = {}
        if provided_state:
            for i in range(self.num_obstacles):
                obs_id = 'obs'+str(i+1)        
                ## overrides the current belief sample with a true state
                b_global_samp = np.sign(provided_state[obs_id].to(DEVICE)) * (np.abs(provided_state[obs_id].to(DEVICE)))

                mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
                rel_vec = mod_obs  - params['pr2'].pose[:,active_ts[1]-1]
                if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
                    ## sample around the true belief, with extremely low variation / error
                    max_obs_dict[obs_id] = dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.01 * torch.eye(2)).sample()
                else:
                    ## sample from prior dist -- have no additional knowledge, don't read it
                    max_obs_dict[obs_id] = dist.MultivariateNormal((torch.zeros((2,))), 0.01 * torch.eye(2)).sample()

                # return tensors on CPU for compatib
                for samp in max_obs_dict:
                    max_obs_dict[samp].to('cpu')
            return max_obs_dict
        
        else:
            for i in range(self.num_obstacles): 
                obs_id = 'obs'+str(i+1)        
                ## give most-likely observation on current particle set
                obstacle_particles = params[obs_id].belief.samples[:,:,-1]

                observations = {}

                for idx in range(obstacle_particles.shape[0]):
                    ## obtain sample observation for this particle
                    rel_vec = obstacle_particles[idx, :]  - params['pr2'].pose[:,active_ts[1]-1]
                    if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
                        tmp_observation = dist.MultivariateNormal(torch.tensor(obstacle_particles[idx, :]).float(), 0.01 * torch.eye(2)).sample()
                    else: 
                        tmp_observation = dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.01 * torch.eye(2)).sample()

                    # enter observation into dict
                    is_present = False
                    for proposal_observation in observations.keys():
                        if torch.linalg.vector_norm(proposal_observation - tmp_observation) < 0.1:
                            observations[proposal_observation] += 1
                            is_present = True
                            break

                    if not is_present:
                        observations[tmp_observation] = 1

                max_count = None
                max_obs = None    
                for obs in observations:
                    if max_count is None or max_count < observations[obs]:
                        max_count = observations[obs]
                        max_obs = obs

                max_obs_dict[obs_id] = max_obs
        
            return max_obs_dict

        ## sample through strict prefix of current obs
        # for obs_active_ts in past_obs:
        #     mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
        #     rel_vec = mod_obs - params['pr2'].pose[:,obs_active_ts[1]-1]
        #     if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
        #         ## sample around the true belief, with extremely low variation / error
        #         torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.001 * torch.eye(2)).to(DEVICE)))
        #     else:
        #         ## sample from prior dist -- have no additional knowledge, don't read it
        #         torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

        ## get sample for current timestep, record and return
        
    
    def filter_samples(self, params, active_ts, curr_obs):
        particles = {}
        for i in range(self.num_obstacles):
            obs_id = 'obs'+str(i+1)
            obstacle_particles = params[obs_id].belief.samples[:,:,-1]

            # rejuv_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0.0, 0.0]), covariance_matrix= 0.001* torch.eye(2))
            # rejuv_samples = torch.tensor([obstacle_particles[i,:].detach().numpy() + rejuv_dist.sample_n(obstacle_particles.shape[0])[i,:].detach().numpy() for i in range(obstacle_particles.shape[0])])
            rejuv_samples = torch.tensor([obstacle_particles[i,:].detach().numpy() for i in range(obstacle_particles.shape[0])])

            probs = torch.zeros((rejuv_samples.shape[0], ))

            ## TODO: create an array of conditional probabilities p(o|s)
            for idx in range(rejuv_samples.shape[0]):
                rel_vec = rejuv_samples[idx, :]  - params['pr2'].pose[:,active_ts[1]-1]
                if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
                    probs[idx] = torch.exp(dist.MultivariateNormal(rejuv_samples[idx, :].float(), 0.01 * torch.eye(2)).log_prob(curr_obs[obs_id]))
                else: 
                    probs[idx] = torch.exp(dist.MultivariateNormal(torch.zeros((2,)), 0.01 * torch.eye(2)).log_prob(curr_obs[obs_id]))
            
            try:
                select_idx = torch.multinomial(probs, num_samples=rejuv_samples.shape[0], replacement=True)

                particles['belief_'+obs_id] = rejuv_samples[select_idx, :]
            except:
                print('WARN -- particle filtering encountered numerical error')
                ## if, e.g., the probs array is numerically challenged, then simply don't alter the particles upon this observation
                particles['belief_'+obs_id] = obstacle_particles

        return particles

    ## no VI in the pointer observation
    def fit_approximation(self, params):        
        pass

# class NoVIObstacleTargetObservationModel(ObservationModel):
#     def __init__(self):
#         # uninitialized parameters
#         self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
#         self.active_planned_observations = {'obs1': torch.empty((2,)).detach()}
#         self.obs_dist = 6.0
    
#     def is_in_ray(self, a_pose, target):
#         ray_width = np.pi / 4  ## has 45-degree field of view on either side
#         adjust_a_pose = (a_pose+np.pi/2)%(2*np.pi) -np.pi/2
#         if target[0] > 0:
#             return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose)) <= ray_width
#         else:
#             return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose - np.pi)) <= ray_width

#     def approx_model(self, data):
#         pass

#     def get_unnorm_obs_log_likelihood(self, params, prefix_obs, fail_ts):
#         log_likelihood = torch.zeros((params['obs1'].belief.samples.shape[0], ))

#         for idx in range(params['obs1'].belief.samples.shape[0]):
#             ## initialize log_likelihood to prior probability

#             log_likelihood[idx] = params['obs1'].belief.dist.log_prob(params['obs1'].belief.samples[idx, :, fail_ts]).sum().item()

#             ## add in terms for the forward model
#             for obs_active_ts in prefix_obs:
#                 obs_vec = params['obs1'].belief.samples[idx,:,fail_ts]
#                 mod_obs_vec = torch.sign(obs_vec) * (torch.abs(obs_vec))
#                 rel_vec = mod_obs_vec - params['pr2'].pose[:,obs_active_ts[1]-1]
#                 if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                     ## sample around the true belief, with extremely low variation / error
#                     log_likelihood[idx] += dist.MultivariateNormal(params['obs1'].belief.samples[idx,:,fail_ts], (0.001 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['obs1'])
#                 else:
#                     ## sample from prior dist -- have no additional knowledge, don't read it
#                     log_likelihood[idx] += dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['obs1'])

#             for obs_active_ts in prefix_obs:
#                 obs_vec = params['target1'].belief.samples[idx,:,fail_ts]
#                 mod_obs_vec = torch.sign(obs_vec) * (torch.abs(obs_vec))
#                 rel_vec = mod_obs_vec - params['pr2'].pose[:,obs_active_ts[1]-1]
#                 if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                     ## sample around the true belief, with extremely low variation / error
#                     log_likelihood[idx] += dist.MultivariateNormal(params['target1'].belief.samples[idx,:,fail_ts], (0.001 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['target1'])
#                 else:
#                     ## sample from prior dist -- have no additional knowledge, don't read it
#                     log_likelihood[idx] += dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['target1'])

#         return log_likelihood

#     def forward_model(self, params, active_ts, provided_state=None, past_obs={}):                
#         if provided_state:
#             ## overrides the current belief sample with a true state
#             b_global_samp = np.sign(provided_state['obs1'].to(DEVICE)) * (np.abs(provided_state['obs1'].to(DEVICE)))
#             b_global_samp_targ = np.sign(provided_state['target1'].to(DEVICE)) * (np.abs(provided_state['target1'].to(DEVICE)))
#         else:
#             ## sample from prior dist
#             b_global_samp = torch.sample('belief_obs1', params['obs1'].belief.dist).to(DEVICE)
#             b_global_samp_targ = torch.sample('belief_target1', params['target1'].belief.dist).to(DEVICE)

#         ## sample through strict prefix of current obs
#         for obs_active_ts in past_obs:
#             mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
#             rel_vec = mod_obs - params['pr2'].pose[:,obs_active_ts[1]-1]
#             if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                 ## sample around the true belief, with extremely low variation / error
#                 torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.001 * torch.eye(2)).to(DEVICE)))
#             else:
#                 ## sample from prior dist -- have no additional knowledge, don't read it
#                 torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

#         for obs_active_ts in past_obs:
#             mod_targ = torch.sign(b_global_samp_targ.detach().to('cpu')) * (torch.abs(b_global_samp_targ.detach().to('cpu')))
#             rel_vec = b_global_samp_targ.detach().to('cpu') - params['pr2'].pose[:,obs_active_ts[1]-1]
#             if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                 ## sample around the true belief, with extremely low variation / error
#                 torch.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp_targ.float().to(DEVICE), (0.01 * torch.eye(2)).to(DEVICE)))
#             else:
#                 ## sample from prior dist -- have no additional knowledge, don't read it
#                 torch.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))
        
#         ## get sample for current timestep, record and return
#         samps = {}

#         mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
#         rel_vec = mod_obs  - params['pr2'].pose[:,active_ts[1]-1]
#         if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#             ## sample around the true belief, with extremely low variation / error
#             samps['obs1'] = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))
#         else:
#             ## sample from prior dist -- have no additional knowledge, don't read it
#             samps['obs1'] = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

#         mod_targ = torch.sign(b_global_samp_targ.detach().to('cpu')) * (torch.abs(b_global_samp_targ.detach().to('cpu')))
#         rel_vec = mod_targ - params['pr2'].pose[:,active_ts[1]-1]
#         if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#             ## sample around the true belief, with extremely low variation / error
#             samps['target1'] = torch.sample('target1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp_targ.float().to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))
#         else:
#             ## sample from prior dist -- have no additional knowledge, don't read it
#             samps['target1'] = torch.sample('target1.'+str(active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

#         # return tensors on CPU for compatib
#         for samp in samps:
#             samps[samp].to('cpu')

#         return samps

#     ## no VI in the pointer observation
#     def fit_approximation(self, params):        
#         pass


# class ParticleFilterObstacleTargetObservationModel(ObservationModel):
#     def __init__(self):
#         # uninitialized parameters
#         self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
#         self.active_planned_observations = {'obs1': torch.empty((2,)).detach()}
#         self.obs_dist = 6.0
#         self.particle_noise = 0.01
    
#     def is_in_ray(self, a_pose, target):
#         ray_width = np.pi / 2  ## has 45-degree field of view on either side
#         adjust_a_pose = (a_pose+np.pi/2)%(2*np.pi) -np.pi/2
#         if target[0] > 0:
#             return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose)) <= ray_width
#         else:
#             return np.abs(np.arctan(target[1]/target[0]) - (adjust_a_pose - np.pi)) <= ray_width

#     def approx_model(self, data):
#         pass

#     def get_unnorm_obs_log_likelihood(self, params, prefix_obs, fail_ts):
#         log_likelihood = torch.zeros((params['obs1'].belief.samples.shape[0], ))

#         for idx in range(params['obs1'].belief.samples.shape[0]):
#             ## initialize log_likelihood to prior probability (probability of obstacle or target)
#             log_likelihood[idx] = params['obs1'].belief.dist.log_prob(params['obs1'].belief.samples[idx, :, fail_ts]).sum().item() \
#                 + params['target1'].belief.dist.log_prob(params['target1'].belief.samples[idx, :, fail_ts]).sum().item()

#             ## add in terms for the forward model
#             for obs_active_ts in prefix_obs:
#                 obs_vec = params['obs1'].belief.samples[idx,:,fail_ts]
#                 mod_obs_vec = torch.sign(obs_vec) * (torch.abs(obs_vec))
#                 rel_vec = mod_obs_vec - params['pr2'].pose[:,obs_active_ts[1]-1]
#                 if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                     ## sample around the true belief, with extremely low variation / error
#                     log_likelihood[idx] += dist.MultivariateNormal(params['obs1'].belief.samples[idx,:,fail_ts], (self.particle_noise * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['obs1'])
#                 else:
#                     ## sample from prior dist -- have no additional knowledge, don't read it
#                     log_likelihood[idx] += dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), self.particle_noise * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['obs1'])

#             for obs_active_ts in prefix_obs:
#                 obs_vec = params['target1'].belief.samples[idx,:,fail_ts]
#                 mod_obs_vec = torch.sign(obs_vec) * (torch.abs(obs_vec))
#                 rel_vec = mod_obs_vec - params['pr2'].pose[:,obs_active_ts[1]-1]
#                 if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                     ## sample around the true belief, with extremely low variation / error
#                     log_likelihood[idx] += dist.MultivariateNormal(params['target1'].belief.samples[idx,:,fail_ts], (self.particle_noise * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['target1'])
#                 else:
#                     ## sample from prior dist -- have no additional knowledge, don't read it
#                     log_likelihood[idx] += dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), self.particle_noise * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['target1'])

#         return log_likelihood

#     def forward_model(self, params, active_ts, provided_state=None, past_obs={}):                
#         # if provided_state:
#         #     ## overrides the current belief sample with a true state
#         #     b_global_samp = np.sign(provided_state['obs1'].to(DEVICE)) * (np.abs(provided_state['obs1'].to(DEVICE)))
#         #     b_global_samp_targ = np.sign(provided_state['target1'].to(DEVICE)) * (np.abs(provided_state['target1'].to(DEVICE)))
#         # else:
#         #     ## sample from prior dist
#         #     b_global_samp = torch.sample('belief_obs1', params['obs1'].belief.dist).to(DEVICE)
#         #     b_global_samp_targ = torch.sample('belief_target1', params['target1'].belief.dist).to(DEVICE)

#         # ## sample through strict prefix of current obs
#         # for obs_active_ts in past_obs:
#         #     mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
#         #     rel_vec = mod_obs - params['pr2'].pose[:,obs_active_ts[1]-1]
#         #     if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#         #         ## sample around the true belief, with extremely low variation / error
#         #         torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.001 * torch.eye(2)).to(DEVICE)))
#         #     else:
#         #         ## sample from prior dist -- have no additional knowledge, don't read it
#         #         torch.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

#         # for obs_active_ts in past_obs:
#         #     mod_targ = torch.sign(b_global_samp_targ.detach().to('cpu')) * (torch.abs(b_global_samp_targ.detach().to('cpu')))
#         #     rel_vec = b_global_samp_targ.detach().to('cpu') - params['pr2'].pose[:,obs_active_ts[1]-1]
#         #     if self.is_in_ray(params['pr2'].theta[0,obs_active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#         #         ## sample around the true belief, with extremely low variation / error
#         #         torch.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp_targ.float().to(DEVICE), (0.01 * torch.eye(2)).to(DEVICE)))
#         #     else:
#         #         ## sample from prior dist -- have no additional knowledge, don't read it
#         #         torch.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))
        
#         ## get sample for current timestep, record and return
#         if provided_state:
#             ## overrides the current belief sample with a true state
#             b_global_samp = np.sign(provided_state['obs1'].to(DEVICE)) * (np.abs(provided_state['obs1'].to(DEVICE)))
#             b_global_samp_targ = np.sign(provided_state['target1'].to(DEVICE)) * (np.abs(provided_state['target1'].to(DEVICE)))

#             samps = {}

#             mod_obs = torch.sign(b_global_samp.detach().to('cpu')) * (torch.abs(b_global_samp.detach().to('cpu')))
#             rel_vec = mod_obs  - params['pr2'].pose[:,active_ts[1]-1]
#             if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                 ## sample around the true belief, with extremely low variation / error
#                 samps['obs1'] = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), self.particle_noise * torch.eye(2).to(DEVICE)))
#             else:
#                 ## sample from prior dist -- have no additional knowledge, don't read it
#                 samps['obs1'] = torch.tensor([0.0, 0.0])

    

#             mod_targ = torch.sign(b_global_samp_targ.detach().to('cpu')) * (torch.abs(b_global_samp_targ.detach().to('cpu')))
#             rel_vec = mod_targ - params['pr2'].pose[:,active_ts[1]-1]
#             if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                 ## sample around the true belief, with extremely low variation / error
#                 samps['target1'] = torch.sample('target1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp_targ.float().to(DEVICE), self.particle_noise * torch.eye(2).to(DEVICE)))
#             else:
#                 ## sample from prior dist -- have no additional knowledge, don't read it
#                 samps['target1'] = torch.tensor([0.0, 0.0])

#             # return tensors on CPU for compatib
#             for samp in samps:
#                 samps[samp].to('cpu')
            
#             return samps

#         else:
#             ## give most-likely observation on current particle set
#             obstacle_particles = params['obs1'].belief.samples[:,:,-1]
#             target_particles = params['target1'].belief.samples[:,:,-1]

#             observations_obs = {}
#             observations_targ = {}

#             for idx in range(obstacle_particles.shape[0]):
#                 ## obtain sample observation for this particle
#                 rel_vec = obstacle_particles[idx, :]  - params['pr2'].pose[:,active_ts[1]-1]
#                 if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                     tmp_observation = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal(torch.tensor(obstacle_particles[idx, :]).float(), self.particle_noise * torch.eye(2).to(DEVICE)))
#                 else: 
#                     tmp_observation = torch.tensor([0.0, 0.0])

#                 # enter observation into dict
#                 is_present = False
#                 for proposal_observation in observations_obs.keys():
#                     if torch.linalg.vector_norm(proposal_observation - tmp_observation) < 0.5:
#                         observations_obs[proposal_observation] += 1
#                         is_present = True
#                         break

#                 if not is_present:
#                     observations_obs[tmp_observation] = 1
                
#                 rel_vec = target_particles[idx, :]  - params['pr2'].pose[:,active_ts[1]-1]
#                 if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                     tmp_observation = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal(torch.tensor(target_particles[idx, :]).float(), self.particle_noise * torch.eye(2).to(DEVICE)))
#                 else: 
#                     tmp_observation = torch.tensor([0.0, 0.0])

#                 # enter observation into dict
#                 is_present = False
#                 for proposal_observation in observations_targ.keys():
#                     if torch.linalg.vector_norm(proposal_observation - tmp_observation) < 0.5:
#                         observations_targ[proposal_observation] += 1
#                         is_present = True
#                         break

#                 if not is_present:
#                     observations_targ[tmp_observation] = 1

#             # breakpoint()

#             max_count = None
#             max_obs_obs = None    
#             for obs in observations_obs:
#                 if max_count is None or max_count < observations_obs[obs]:
#                     max_count = observations_obs[obs]
#                     max_obs_obs = obs

#             max_count = None
#             max_obs_targ = None
#             for obs in observations_targ:
#                 if max_count is None or max_count < observations_targ[obs]:
#                     max_count = observations_targ[obs]
#                     max_obs_targ = obs
            
#             return {'obs1': max_obs_obs, 'target1': max_obs_targ}

#             # ## sample from prior dist -- have no additional knowledge, don't read it
#             # samps['obs1'] = torch.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal((torch.zeros((2,))).to(DEVICE), 0.001 * torch.eye(2).to(DEVICE)))

#     def filter_samples(self, params, active_ts, curr_obs):
#         ## TODO: generate an list of the current obstacle particles
#         obstacle_particles = params['obs1'].belief.samples[:,:,-1]
#         target_particles = params['target1'].belief.samples[:,:,-1]

#         rejuv_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0.0, 0.0]), covariance_matrix= 0.001* torch.eye(2))
#         rejuv_samples_obs = torch.tensor([obstacle_particles[i,:].detach().numpy() + rejuv_dist.sample_n(obstacle_particles.shape[0])[i,:].detach().numpy() for i in range(obstacle_particles.shape[0])])
#         rejuv_samples_targ = torch.tensor([target_particles[i,:].detach().numpy() + rejuv_dist.sample_n(target_particles.shape[0])[i,:].detach().numpy() for i in range(target_particles.shape[0])])

#         probs_obs = torch.zeros((rejuv_samples_obs.shape[0], ))

#         ## TODO: create an array of conditional probabilities p(o|s)
#         for idx in range(rejuv_samples_obs.shape[0]):
#             rel_vec = rejuv_samples_obs[idx, :]  - params['pr2'].pose[:,active_ts[1]-1]
#             if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                 probs_obs[idx] = torch.exp(dist.MultivariateNormal(rejuv_samples_obs[idx, :].float(), 0.01 * torch.eye(2)).log_prob(curr_obs['obs1']))
#             else: 
#                 probs_obs[idx] = torch.exp(dist.MultivariateNormal(torch.zeros((2,)), 0.1 * torch.eye(2)).log_prob(curr_obs['obs1']))
        
#         try:
#             select_idx_obs = torch.multinomial(probs_obs, num_samples=rejuv_samples_obs.shape[0], replacement=True)
#         except:
#             select_idx_obs = torch.range(rejuv_samples_obs.shape[0]) ## if observation outside support, simply repeat the particles
#         probs_targ = torch.zeros((rejuv_samples_obs.shape[0], ))

#         for idx in range(rejuv_samples_targ.shape[0]):
#             rel_vec = rejuv_samples_targ[idx, :]  - params['pr2'].pose[:,active_ts[1]-1]
#             if self.is_in_ray(params['pr2'].theta[0,active_ts[1]-1], rel_vec) and np.linalg.norm(rel_vec) <= self.obs_dist:
#                 probs_targ[idx] = torch.exp(dist.MultivariateNormal(rejuv_samples_targ[idx, :].float(), 0.1 * torch.eye(2)).log_prob(curr_obs['target1']))
#             else: 
#                 probs_targ[idx] = torch.exp(dist.MultivariateNormal(torch.zeros((2,)), 0.1 * torch.eye(2)).log_prob(curr_obs['target1']))
        
#         try:
#             select_idx_targ = torch.multinomial(probs_targ, num_samples=rejuv_samples_targ.shape[0], replacement=True)
#         except:
#             select_idx_targ = torch.range(rejuv_samples_obs.shape[0]) ## if observation outside support, simply repeat the particles

#         particles = {}
#         particles['belief_obs1'] = rejuv_samples_obs[select_idx_obs, :]
#         particles['belief_target1'] = rejuv_samples_targ[select_idx_targ, :]

#         return particles


#     ## no VI in the pointer observation
#     def fit_approximation(self, params):        
#         pass