"""Transformation tensorflow layers"""
import tensorflow as tf
import numpy as np
from scipy import constants


class hitnet_trafo(tf.keras.layers.Layer):
    '''Class to transform inputs for Hitnet
    '''
    speed_of_light = constants.c * 1e-9  # c in m / ns

    def __init__(self, labels, min_energy=0.1, max_energy=1e4):
        '''
        Parameters:
        -----------

        labels : list
            list of labels corresponding to the data array
        '''

        super().__init__()

        self.labels = labels
        self.min_energy = min_energy
        self.max_energy = max_energy

        self.azimuth_idx = labels.index('azimuth')
        self.zenith_idx = labels.index('zenith')
        self.time_idx = labels.index('time')
        self.x_idx = labels.index('x')
        self.y_idx = labels.index('y')
        self.z_idx = labels.index('z')
        self.energy_idx = labels.index('energy')

    def get_config(self):
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}

    def call(self, hit, params):
        '''
        Parameters:
        -----------

        hit : tensor
            shape (N, 5), containing hit DOM position x, y, z, time, and charge

        params : tensor
            shape (N, len(labels))

        '''

        cosphi = tf.math.cos(params[:, self.azimuth_idx])
        sinphi = tf.math.sin(params[:, self.azimuth_idx])

        sintheta = tf.math.sin(params[:, self.zenith_idx])
        dir_x = sintheta * cosphi
        dir_y = sintheta * sinphi
        dir_z = tf.math.cos(params[:, self.zenith_idx])

        dt = hit[:, 3] - params[:, self.time_idx]

        energy = params[:, self.energy_idx] - 1
        event_time = (params[:, self.time_idx]) / 25
        pmt_time = (hit[:, 3]) / 25

        out = tf.stack([
            params[:, self.x_idx] / 1000,
            params[:, self.y_idx] / 1000,
            params[:, self.z_idx] / 1000,
            dir_x,
            dir_y,
            dir_z,
            pmt_time - event_time,
            energy,
            hit[:, 0] / 1000,
            hit[:, 1] / 1000,
            hit[:, 2] / 1000,
        ],
            axis=1
        )

        return out


class chargenet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for Charget Net
    '''

    def __init__(self, labels, min_energy=0.1, max_energy=1e4):
        '''
        Parameters:
        -----------

        labels : list
            list of labels corresponding to the data array
        '''

        super().__init__()

        self.labels = labels
        self.min_energy = min_energy
        self.max_energy = max_energy

        self.azimuth_idx = labels.index('azimuth')
        self.zenith_idx = labels.index('zenith')
        self.x_idx = labels.index('x')
        self.y_idx = labels.index('y')
        self.z_idx = labels.index('z')
        self.energy_idx = labels.index('energy')

    def get_config(self):
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}

    def call(self, charge, params):
        '''
        Parameters:
        -----------

        charge : tensor
            shape (N, 2), containing the event total charge and number of hit DOMs

        params : tensor
            shape (N, len(labels))

        '''

        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])

        energy = params[:, self.energy_idx] - 1

        out = tf.stack([
            charge[:, 0] / 40 - 1,
            charge[:, 1] / 40 - 1,  # n_channels
            params[:, self.x_idx] / 1000,
            params[:, self.y_idx] / 1000,
            params[:, self.z_idx] / 1000,
            dir_x,
            dir_y,
            dir_z,
            energy
        ],
            axis=1
        )

        return out
