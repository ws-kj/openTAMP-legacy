""" This file provides an example tensorflow network used to define a policy. """

import functools
import operator
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def precision_mse(output, y, precision):
    return torch.matmul(torch.matmul((output-y), precision), (output-y).t())


class TorchNet(nn.Module):
    def __init__(self, config, device=None):
        # nn.Module.__init__(self)
        super(TorchNet, self).__init__()
        self.config = config
        for key in ['obs_include', 'out_include']:
            self.config[key] = list(set(config[key]))

        if device is None: device = torch.device('cpu')
        if type(device) is str: device = torch.device(device)
        self.device = device
        self.fp_tensors = None

        self.output_boundaries = config.get("output_boundaries", None)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.recur_layers = nn.ModuleList()

        self.fc_input_dim = 0
        for sensor in self.config['obs_include']:
            dim = self.config['sensor_dims'][sensor]
            self.fc_input_dim += dim

        self.output_dim = 0
        for sensor in self.config['out_include']:
            dim = self.config['sensor_dims'][sensor]
            self.output_dim += dim
        if self.config.get('loss_fn', None) is not None and self.config.get('loss_fn', None) == 'GaussianNLLLoss':
            self.output_dim *= 10 ## 5 times for the means, 5 times for the vars

        self._compute_idx()

        if self.config['build_conv']:
            self._build_conv_layers()
            
            self.conv_to_fc = config.get('conv_to_fc', 'fp')
            if self.conv_to_fc is 'fp':
                self.fp_tensor = None
                self._build_fp()

        if self.config['build_recur']:
            self._build_recur_layers()

        self._build_fc_layers()
        self._set_nonlin_and_loss()

        self.to(self.device)
        self.to_device(self.device)

        # print(config[])


    def forward(self, nn_input):
        if type(nn_input) is not torch.Tensor:
            nn_input = torch.Tensor(nn_input)
        nn_input = nn_input.to(device=self.device)

        if len(self.conv_layers):
            nn_input = self.conv_forward(nn_input)
        
        if len(self.recur_layers):
            nn_input = self.recur_forward(nn_input)

        raw_output = self.fc_forward(nn_input)
        if self.output_fn:
            if self.output_boundaries:
                nn_output = []
                for (start, end) in self.output_boundaries:
                    nn_output.append(self.output_fn(raw_output[:, start:end]))

                return torch.cat(nn_output, dim=-1)

            else:
                return self.output_fn(raw_output)

        return raw_output


    def conv_forward(self, nn_input):
        if not len(self.conv_layers):
            return nn_input

        n_pts = nn_input.size()[0]
        state_input = nn_input[:, 0:self.x_idx[-1]+1]
        image_input = nn_input[:, self.x_idx[-1]+1:self.img_idx[-1]+1]
         
        im_height = self.config['image_height']
        im_width = self.config['image_width']
        num_channels = self.config['image_channels']
        image_input = image_input.view([-1, im_width, im_height, num_channels])
        image_input = torch.movedim(image_input, 3, 1)
        for conv_layer in self.conv_layers[:-1]:
            image_input = conv_layer(image_input)
            image_input = self.act_fn(image_input)

        image_input = self.conv_layers[-1](image_input)

        if self.conv_to_fc is 'fp':
            image_input = self.compute_fp(image_input)

        image_input = image_input.view([n_pts, -1])
        return torch.cat(tensors=[image_input, state_input], dim=1)


    def fc_forward(self, nn_input):
        for fc_layer in self.fc_layers[:-1]:
            nn_input = fc_layer(nn_input)
            nn_input = self.act_fn(nn_input)
        return self.fc_layers[-1](nn_input)

    def recur_forward(self, nn_input):
        ## unpack current data into a sequence of tensors, removing all -1 / sentinel entries
        recurrent_entries = [(r[r != np.inf]).view(self.num_recur_features, -1).T for r in list(nn_input[:, self.recur_idx])]
        
        recur_input = nn.utils.rnn.pack_sequence(recurrent_entries, enforce_sorted=False)

        ## pass through recurrent layer
        output, _ = self.recur_layers[0](recur_input)

        unpacked_output = nn.utils.rnn.unpack_sequence(output)

        last_states_only = torch.stack([u[-1, :] for u in unpacked_output])
        
        ## return concatenation of non-recurrent input with final hidden states on each batch
        return torch.cat([nn_input[:, self.x_idx], last_states_only], dim=1)


    def to_device(self, device=None):
        if device is not None:
            self.device = device

        for idx, layer in enumerate(self.fc_layers):
            self.fc_layers[idx] = layer.to(self.device)

        for idx, conv_layer in enumerate(self.conv_layers):
            self.conv_layers[idx] = conv_layer.to(self.device)


        if self.fp_tensors is not None:
            new_arr = [0.]*len(self.fp_tensors)
            for idx, tensor in enumerate(self.fp_tensors):
                new_arr[idx] = tensor.to(self.device)
            self.fp_tensors = new_arr

        self.to(self.device)
        

    def _compute_idx(self):
        if 'idx' in self.config:
            self.idx = self.config['idx']
            self.x_idx, self.img_idx, self.recur_idx = [], [], []
            for (sensor, inds) in self.idx.items():
                if sensor in self.config['obs_image_data']:
                    self.img_idx.extend(inds)
                elif sensor in self.config['recur_obs_include']:
                    self.recur_idx.extend(inds)
                else:
                    self.x_idx.extend(inds)

        else:
            self.x_idx, self.img_idx, i = [], [], 0
            for sensor in self.config['obs_include']:
                dim = self.config['sensor_dims'][sensor]

                if sensor in self.config['obs_image_data']:
                    self.img_idx = self.img_idx + list(range(i, i+dim))
                elif sensor in self.config['recur_obs_include']:
                    self.recur_idx = self.recur_idx + list(range(i, i+dim))
                else:
                    self.x_idx = self.x_idx + list(range(i, i+dim))

                i += dim


    def _set_nonlin_and_loss(self):
        self.act_fn = self.config.get('act_fn', F.relu)
        if type(self.act_fn) is str: self.act_fn = getattr(F, self.act_fn)

        self.output_fn = self.config.get('output_fn', None)
        if type(self.output_fn) is str: self.output_fn = getattr(F, self.output_fn)

        self.loss_fn = self.config.get('loss_fn', F.mse_loss)
        self.use_precision = self.config['use_precision']
        # self.use_precision = False
        if self.loss_fn == 'precision_mse': 
            self.loss_fn = precision_mse
            self.use_precision = True
        if self.loss_fn == 'GaussianNLLLoss':
            self.loss_fn = torch.nn.GaussianNLLLoss(eps=1e-4)
            self.use_precision = False
        if type(self.loss_fn) is str: self.loss_fn = getattr(F, self.loss_fn)

   
    def _build_conv_layers(self):
        num_filters = self.config.get('num_filters', [])
        filter_sizes = self.config.get('filter_sizes', [])
        self.n_conv = len(num_filters)
        if self.n_conv == 0: return

        im_height = self.config['image_height']
        im_width = self.config['image_width']
        num_channels = self.config['image_channels']
        cur_channels = num_channels

        for n in range(self.n_conv):
            conv_layer = nn.Conv2d(cur_channels, num_filters[n], filter_sizes[n])
            cur_channels = num_filters[n]
            self.conv_layers.append(conv_layer)

        # Compute size of output
        temp_model = nn.Sequential(*self.conv_layers)
        rand_inputs = torch.rand(1, num_channels, im_height, im_width)
        shape = list(temp_model(rand_inputs).shape)
        self.conv_output_dim = shape
        self.fc_input_dim = functools.reduce(operator.mul, shape)

    def _build_recur_layers(self):
        hist_len = 20 ## TODO FEED IN DATA FROM HYPERPARAMS
        overall_recur_dim = len(self.recur_idx)
        self.num_recur_features = int(overall_recur_dim / hist_len)
        recur_hidden_size = self.config.get('recur_dim_hidden', 128)
        recur_num_layers = self.config.get('recur_num_layers', 1)
        self.recur_layers.append(nn.LSTM(input_size=self.num_recur_features, hidden_size=recur_hidden_size, num_layers=recur_num_layers))
        self.fc_input_dim = len(self.x_idx) + recur_hidden_size

    def _build_fc_layers(self):
        n_fc_layers = self.config.get('n_layers', 1)
        dim_hidden = self.config.get('dim_hidden', 40)
        cur_dim = self.fc_input_dim
        for n in range(n_fc_layers):
            next_dim = dim_hidden if np.isscalar(dim_hidden) else dim_hidden[n]
            fc_layer = nn.Linear(cur_dim, next_dim)
            cur_dim = next_dim
            self.fc_layers.append(fc_layer)

        fc_layer = nn.Linear(cur_dim, self.output_dim)
        self.fc_layers.append(fc_layer)


    def _build_fp(self):
        _, num_fp, num_rows, num_cols = self.conv_output_dim
        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        x_map = torch.from_numpy(x_map)
        y_map = torch.from_numpy(y_map)

        x_map = x_map.view([num_rows * num_cols])
        y_map = y_map.view([num_rows * num_cols])
        self.fp_tensors = (x_map, y_map)

        # Compute size of output
        rand_inputs = torch.rand(1, num_fp, num_rows, num_cols)
        shape = list(self.compute_fp(rand_inputs).shape)
        self.fc_input_dim = functools.reduce(operator.mul, shape) + len(self.x_idx)

    def compute_fp(self, input_layer):
        if self.fp_tensors is None: self._build_fp()
        _, num_fp, num_rows, num_cols = self.conv_output_dim
        features = input_layer.reshape([-1, num_rows*num_cols])
        func = torch.nn.Softmax()
        softmax = func(features)
        fp_x = torch.sum(torch.multiply(self.fp_tensors[0], softmax), dim=[1], keepdim=True)
        fp_y = torch.sum(torch.multiply(self.fp_tensors[1], softmax), dim=[1], keepdim=True)
        fp = torch.cat(tensors=[fp_x, fp_y], dim=1).view([-1, num_fp*2])
        return fp


    def _compute_loss_component(self, pred, y, precision=None):
        if self.use_precision:
            if len(precision.size()) > len(pred.size()):
                return torch.mean(self.loss_fn(pred, y, precision))
            else:
                y = torch.argmax(y, dim=-1).flatten()
                precision = precision.flatten()
                if torch.sum(precision) > 0:
                    sum_loss = torch.sum(self.loss_fn(pred, y, reduction='none') * precision)
                    return sum_loss / torch.sum(precision)
                else:
                    return torch.mean(self.loss_fn(pred, y, reduction='none'))  ## just average the loss as a behaviour
        else:
            if self.loss_fn == F.nll_loss:
                y = torch.argmax(y, dim=-1).flatten()
            elif type(self.loss_fn) == torch.nn.GaussianNLLLoss:
                softplus = torch.nn.Softplus()
                pred = pred.reshape(-1, 10, y.shape[1])
                return self.loss_fn(pred[:,:5, :], torch.unsqueeze(y, 1), softplus(pred[:, 5:, :]))
            else:
                pred = pred.flatten()
                y = y.flatten()
            return self.loss_fn(pred, y, reduction='mean')

    def compute_loss(self, pred, y, precision=None):
        pred = pred.to(self.device)
        y = y.to(self.device)
        if precision is not None:
            # precision = precision.to(self.device)
            precision.to(self.device)

        if self.output_boundaries:
            cur_loss = None
            n = 0
            for ind, (start, end) in enumerate(self.output_boundaries):
                next_pred = pred[:, start:end]
                next_y = y[:, start:end]

                if len(precision.size()) > len(pred.size()):
                    next_precision = precision[:, start:end][:, :, start:end]
                else:
                    next_precision = precision[:, ind]

                n += 1
                if cur_loss is None:
                    cur_loss = self._compute_loss_component(next_pred, next_y, next_precision)
                else:
                    cur_loss += self._compute_loss_component(next_pred, next_y, next_precision)
                
            return cur_loss / n
        
        return self._compute_loss_component(pred, y, precision)


class PolicyNet(TorchNet):
    def __init__(self, config, scope, device=None):
        self.scope = scope
        self.normalize = config.get('normalize', False)
        # self.normalize = False
        self.scale = None
        self.bias = None

        super().__init__(config=config, device=device)

    def act(self, X, obs, t, noise=None, eta=1.):

        if self.scope in ['primitive', 'cont']:
            is_cont = self.scope == 'cont'
            return self.task_distr(obs, self.config['prim_bounds'], eta, is_cont)

        if len(np.shape(obs)) == 1:
            flatten = True
            obs = np.expand_dims(obs, axis=0)

        if self.normalize:
            if self.scale is None or self.bias is None:
                raise ValueError('scale & bias must be set before normalization')

            obs = obs.copy()
            obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias

        with torch.no_grad():
            if self.config.get('loss_fn', None) is not None and self.config.get('loss_fn', None) == 'GaussianNLLLoss':
                # sample and return a Gaussian mixture (for now, K=5 hardcoded as in prior work)
                act_preproc = self.forward(obs).cpu().detach()
                act_preproc = act_preproc.reshape(10,-1)
                means = act_preproc[:5, :]
                variances = act_preproc[5:, :]
                softplus = torch.nn.Softplus()
                variances = softplus(variances)
                variances = torch.stack([torch.diag(a) for _, a in enumerate(variances)])

                D = torch.distributions
                mix = D.Categorical(torch.ones(5,))
                comp = D.MultivariateNormal(means, variances)
                gmm = D.MixtureSameFamily(mix, comp)

                act = gmm.sample().detach().numpy()
            else:
                act = self.forward(obs).cpu().detach().numpy()

        if noise is not None:
            act += noise

        return act.flatten() if flatten else act


    def task_distr(self, obs, bounds, eta=1., cont=False):
        if len(obs.shape) < 2:
            flatten = True
            obs = obs.reshape(1, -1)

        with torch.no_grad():
            vals = self.forward(obs).cpu().detach().numpy()
            if self.output_fn is F.log_softmax:
                vals = np.exp(vals)
            elif self.output_fn is F.softmax:
                pass
            elif not cont:
                raise NotImplementedError("Cannot use output fn {} for task prediction!".format(self.output_fn))

        res = []
        for bound in bounds:
            next_val = vals[:, bound[0]:bound[1]]
            if flatten:
                next_val = next_val.flatten()

            res.append(next_val)

        return res


    def is_initialized(self):
        if not self.normalize: return True
        return not (self.scale is None or self.bias is None)

