""" This file defines the base agent class. """
import abc
import copy

from opentamp.policy_hooks.utils.sample_list import SampleList
from opentamp.policy_hooks.utils.policy_solver_utils import ACTION_ENUM


class Agent(object, metaclass=abc.ABCMeta):
    """
    Agent superclass. The agent interacts with the environment to
    collect samples.
    """

    def __init__(self, hyperparams):
        config = {}
        config.update(hyperparams)
        self._hyperparams = config

        self.T = self._hyperparams['T']
        self.dU = self._hyperparams['sensor_dims'][ACTION_ENUM]

        self.x_data_types = list(set(self._hyperparams['state_include']))
        self.obs_data_types = list(set(self._hyperparams['obs_include']))
        
        self.prim_obs_data_types = list(set(self._hyperparams['prim_obs_include']))
        self.prim_out_data_types = list(set(self._hyperparams['prim_out_include']))

        if 'cont_obs_include' in self._hyperparams:
            self.cont_obs_data_types = list(set(self._hyperparams['cont_obs_include']))
        else:
            self.cont_obs_data_types = self.prim_obs_data_types

        if 'cont_out_include' in self._hyperparams:
            self.cont_out_data_types = list(set(self._hyperparams['cont_out_include']))
        else:
            self.cont_out_data_types = self.prim_out_data_types

        self.meta_data_types = self._hyperparams.get('meta_include', [])

        # List of indices for each data type in state X.
        self._state_idx, i = [], 0
        for sensor in self.x_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            i += dim
        self.dX = i

        # List of indices for each data type in observation.
        self._obs_idx, i = [], 0
        for sensor in self.obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dO = i

        self._prim_obs_idx, i = [], 0
        for sensor in self.prim_obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._prim_obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dPrim = i

        self._prim_out_idx, i = [], 0
        for sensor in self.prim_out_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._prim_out_idx.append(list(range(i, i+dim)))
            i += dim
        self.dPrimOut = i

        self._cont_obs_idx, i = [], 0
        for sensor in self.cont_obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._cont_obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dCont = i

        self._cont_out_idx, i = [], 0
        for sensor in self.cont_out_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._cont_out_idx.append(list(range(i, i+dim)))
            i += dim
        self.dContOut = i

        # List of indices for each data type in meta data.
        self._meta_idx, i = [], 0
        for sensor in self.meta_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._meta_idx.append(list(range(i, i+dim)))
            i += dim
        self.dM = i

        self._x_data_idx = {d: i for d, i in zip(self.x_data_types,
                                                 self._state_idx)}
        self._obs_data_idx = {d: i for d, i in zip(self.obs_data_types,
                                                   self._obs_idx)}
        self._prim_obs_data_idx = {d: i for d, i in zip(self.prim_obs_data_types,
                                                        self._prim_obs_idx)}
        self._prim_out_data_idx = {d: i for d, i in zip(self.prim_out_data_types,
                                                        self._prim_out_idx)}
        self._cont_obs_data_idx = {d: i for d, i in zip(self.cont_obs_data_types,
                                                        self._cont_obs_idx)}
        self._cont_out_data_idx = {d: i for d, i in zip(self.cont_out_data_types,
                                                        self._cont_out_idx)}
        self._meta_data_idx = {d: i for d, i in zip(self.meta_data_types,
                                                   self._meta_idx)}

    def get_init_state(self, condition):
        return self.x0[condition].copy()

    @abc.abstractmethod
    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Draw a sample from the environment, using the specified policy
        and under the specified condition, with or without noise.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self, condition):
        """ Reset environment to the specified condition. """
        pass  # May be overridden in subclass.

    def get_samples(self, condition, start=0, end=None):
        """
        Return the requested samples based on the start and end indices.
        Args:
            start: Starting index of samples to return.
            end: End index of samples to return.
        """
        return (SampleList(self._samples[condition][start:]) if end is None
                else SampleList(self._samples[condition][start:end]))

    def clear_samples(self, condition=None):
        """
        Reset the samples for a given condition, defaulting to all conditions.
        Args:
            condition: Condition for which to reset samples.
        """
        if condition is None:
            self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        else:
            self._samples[condition] = []

    def delete_last_sample(self, condition):
        """ Delete the last sample from the specified condition. """
        self._samples[condition].pop()

    def get_idx_x(self, sensor_name):
        """
        Return the indices corresponding to a certain state sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._x_data_idx[sensor_name]

    def get_idx_obs(self, sensor_name):
        """
        Return the indices corresponding to a certain observation sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._obs_data_idx[sensor_name]

    def pack_data(self, existing_mat, data_to_insert, data_types,
                      axes=None, key='X'):
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        opts = {'X': (self.dX, self._x_data_idx),
                'obs': (self.dO, self._obs_data_idx),
                'prim_obs': (self.dPrim, self._prim_obs_data_idx),
                'prim_out': (self.dPrimOut, self._prim_out_data_idx),
                'cont_obs': (self.dCont, self._cont_obs_data_idx),
                'cont_out': (self.dContOut, self._cont_out_data_idx),}
        dim, idx = opts[key]
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != dim:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dO)
            insert_shape[axes[i]] = len(idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(idx[data_types[i]][0],
                                   idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert


    def pack_data_obs(self, existing_mat, data_to_insert, data_types,
                      axes=None):
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dO:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dO)
            insert_shape[axes[i]] = len(self._obs_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._obs_data_idx[data_types[i]][0],
                                   self._obs_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def pack_data_prim_obs(self, existing_mat, data_to_insert, data_types,
                      axes=None):
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dPrim:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dPrim)
            insert_shape[axes[i]] = len(self._prim_obs_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            print(insert_shape, data_to_insert)
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._prim_obs_data_idx[data_types[i]][0],
                                   self._prim_obs_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def pack_data_prim_out(self, existing_mat, data_to_insert, data_types,
                           axes=None):
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dPrimOut:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dPrimOut)
            insert_shape[axes[i]] = len(self._prim_out_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._prim_out_data_idx[data_types[i]][0],
                                   self._prim_out_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def pack_data_cont_obs(self, existing_mat, data_to_insert, data_types,
                      axes=None):
        num_sensor = len(data_types)
        if axes is None:
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dCont:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dCont)
            insert_shape[axes[i]] = len(self._cont_obs_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._cont_obs_data_idx[data_types[i]][0],
                                   self._cont_obs_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert


    def pack_data_cont_out(self, existing_mat, data_to_insert, data_types,
                           axes=None):
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dContOut:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dContOut)
            insert_shape[axes[i]] = len(self._cont_out_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._cont_out_data_idx[data_types[i]][0],
                                   self._cont_out_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def pack_data_meta(self, existing_mat, data_to_insert, data_types,
                       axes=None):
        """
        Update the meta data matrix with new data.
        Args:
            existing_mat: Current meta data matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dM:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dM)
            insert_shape[axes[i]] = len(self._meta_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._meta_data_idx[data_types[i]][0],
                                   self._meta_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def pack_data_x(self, existing_mat, data_to_insert, data_types, axes=None):
        """
        Update the state matrix with new data.
        Args:
            existing_mat: Current state matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def unpack_data_x(self, existing_mat, data_types, axes=None):
        """
        Returns the requested data from the state matrix.
        Args:
            existing_mat: State matrix to unpack from.
            data_types: Names of the sensor to unpack.
            axes: Which axes to unpack along. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        return existing_mat[tuple(index)]
