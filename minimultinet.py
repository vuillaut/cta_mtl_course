import torch
import torch.nn as nn
import pkg_resources
import tables
import logging
import indexedconv.utils as cvutils
from indexedconv.engine import IndexedConv, IndexedMaxPool2d, IndexedAveragePool2d, MaskedConv2d
import torch.nn.functional as F


class _IndexedConvLayer(nn.Sequential):
    def __init__(self, layer_id, index_matrix, num_input, num_output, non_linearity=nn.ReLU,
                 pooling=IndexedAveragePool2d, pooling_kernel='Hex', pooling_radius=1, pooling_stride=2,
                 pooling_dilation=1, pooling_retina=False,
                 batchnorm=True, drop_rate=0, bias=True,
                 kernel_type='Hex', radius=1, stride=1, dilation=1, retina=False):
        super(_IndexedConvLayer, self).__init__()
        self.drop_rate = drop_rate
        indices = cvutils.neighbours_extraction(index_matrix, kernel_type, radius, stride, dilation, retina)
        self.index_matrix = cvutils.pool_index_matrix(index_matrix, kernel_type=pooling_kernel, stride=1)
        self.add_module('cv'+layer_id, IndexedConv(num_input, num_output, indices, bias))
        if pooling is not None:
            p_indices = cvutils.neighbours_extraction(self.index_matrix, pooling_kernel, pooling_radius, pooling_stride,
                                                    pooling_dilation, pooling_retina)
            self.index_matrix = cvutils.pool_index_matrix(self.index_matrix, kernel_type=pooling_kernel,
                                                        stride=pooling_stride)
            self.add_module('pool'+layer_id, pooling(p_indices))
        if batchnorm:
            self.add_module('bn'+layer_id, nn.BatchNorm1d(num_output))
        if non_linearity is not None:
            self.add_module(non_linearity.__name__ + layer_id, non_linearity())

    def forward(self, x):
        new_features = super(_IndexedConvLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class MiniMultiNet(nn.Module):
    """
        Network with indexed convolutions and pooling. => Gamma-PHysNet
        Add missing ReLU in MT block
    """

    def __init__(self, net_parameters_dic, camera_parameters):
        """

        Parameters
        ----------
        net_parameters_dic (dict): a dictionary describing the parameters of the network
        camera_parameters (dict): a dictionary containing the parameters of the camera used with this network
        """
        super(MiniMultiNet, self).__init__()
        self.logger = logging.getLogger(__name__ + '.MiniMultiNet')
        camera_layout = camera_parameters['layout']
        pooling_kernel = camera_layout
        index_matrix = cvutils.create_index_matrix(camera_parameters['nbRow'],
                                                 camera_parameters['nbCol'],
                                                 camera_parameters['injTable'])

        # Options
        n_features = [2, 16, 16, 32, 32]
        init = 'kaiming'
        self.drop_rate = 0.1
        self.batchnorm = True

        self.feature = nn.Sequential()

        # Feature
        for i, (n_in, n_out) in enumerate(zip(n_features[:-1], n_features[1:])):
            if i == 0:
                cv_layer = _IndexedConvLayer(str(i), index_matrix, n_in, n_out, pooling=None,
                                             batchnorm=self.batchnorm, drop_rate=self.drop_rate)
            else:
                cv_layer = _IndexedConvLayer(str(i), index_matrix, n_in, n_out,
                                             batchnorm=self.batchnorm, drop_rate=self.drop_rate)
            self.feature.add_module('cv_layer' + str(i), cv_layer)

            index_matrix = cv_layer.index_matrix


        # Multitasking block
        self.energy = nn.Sequential(
            nn.Linear(n_features[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.fusion = nn.Linear(n_features[-1], 64)

        self.impact = nn.Linear(64, 2) #regressor['impact'])  # !! Define output size
        self.direction = nn.Linear(64, 2) #regressor['direction'])  # !! Define output size

        self.classifier = nn.Linear(n_features[-1], 2) #num_class)  # !! Define output size + !! num input channel


        for m in self.modules():
            if isinstance(m, IndexedConv):
                if init == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                elif init == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out')  # Recommended init when using ReLU
                else:
                    self.logger.warning('Unknown initialization, use default one')

    def forward(self, x):
        out = self.feature(x)
        out = torch.mean(out, 2)  # Global average pooling. After that, out dimension is (n_batch, n_canal, 1).
        out_tot = {}

        out_tot['energy'] = self.energy(out)
        out_f = F.relu(self.fusion(out))  # !! Add dropout ?
        out_tot['impact'] = self.impact(out_f)
        out_tot['direction'] = self.direction(out_f)
        out_tot['class'] = nn.LogSoftmax(1)(self.classifier(out))
        return out_tot
    
  

    
def load_camera_parameters(filepath='data/camera_parameters.h5', camera_type='LST_LSTCam'):
    """
    Load camera parameters : nbCol and injTable
    Parameters
    ----------
    datafiles (List) : files to load data from
    camera_type (str): the type of camera to load data for ; eg 'LST_LSTCam'

    Returns
    -------
    A dictionary
    """
    camera_parameters = {}
    if camera_type == 'LST':
        camera_type = 'LST_LSTCam'
    if camera_type in ['LST_LSTCam', 'MST_FlashCam', 'MST_NectarCam', 'CIFAR']:
        camera_parameters['layout'] = 'Hex'
    else:
        camera_parameters['layout'] = 'Square'
    camera_parameters_file = pkg_resources.resource_filename(__name__, filepath)
    with tables.File(camera_parameters_file, 'r') as hdf5_file:
        camera_parameters['nbRow'] = hdf5_file.root[camera_type]._v_attrs.nbRow
        camera_parameters['nbCol'] = hdf5_file.root[camera_type]._v_attrs.nbCol
        camera_parameters['injTable'] = hdf5_file.root[camera_type].injTable[()]
        camera_parameters['pixelsPosition'] = hdf5_file.root[camera_type].pixelsPosition[()]

    return camera_parameters

