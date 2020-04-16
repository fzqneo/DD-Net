from keras.optimizers import *
from keras.models import Model, load_model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
import tensorflow as tf

import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt 
from scipy.spatial.distance import cdist

#######################################################
## Public functions
#######################################################

#######################################################
## OpenPose data cleaning
#######################################################

OP_HAND_PICKED_GOOD_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16]


def nan_helper(y):
    """Helper function to handle real indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

class OpenPoseDataCleaner(object):

    def __init__(self, copy=True, filter_joint_idx=OP_HAND_PICKED_GOOD_JOINTS):
        super().__init__()
        self.copy = copy
        self.filter_joint_idx = filter_joint_idx

    def transform_point(self, p):
        """Clean a point output by OpenPose
        
        Arguments:
            p {ndarray} -- OpenPose output containing 0s representing unknown joints
        """
        p = self.make_nan(p, self.copy)
        if self.filter_joints:
            p = self.filter_joints(p, self.filter_joint_idx)
        p = self.temporal_interp(p, self.copy)
        p = self.per_video_normalize(p, self.copy)
        p = self.fill_nan_uniform(p, self.copy)
        return p

    def augment_XY(self, X, Y, factor=5):
        """Take a training set X, Y and augment it by factor of `factor`.
        Augmentation comes from the use of randomized functions like `fill_nan_uniform`
        
        Arguments:
            X {list of ndarray} -- [description]
            Y {ndarray} -- shape (num_points, num_classes)
        
        Keyword Arguments:
            factor {int} -- [description] (default: {5})
        """
        Xa = []
        Ya = []
        for p1, y1 in zip(X, Y):
            Xa.extend([self.transform_point(p1) for _ in range(factor)])
            Ya.extend([y1] * factor)

        Ya = np.stack(Ya)
        assert len(Xa) == Ya.shape[0]
        return Xa, Ya

    @staticmethod
    def make_nan(p, copy=True):
        """
        Convert 0 values to np.nan
        """
        assert isinstance(p, np.ndarray)
        q = p.copy() if copy else p
        q[q == 0] = np.nan
        return q

    @staticmethod
    def has_nan(p):
        assert isinstance(p, np.ndarray)
        return np.isnan(p).any()

    @staticmethod
    def count_nan(p):
        assert isinstance(p, np.ndarray)
        return np.isnan(p).sum()
        
    @staticmethod
    def filter_joints(p, good_joint_idx):
        """
        Filter a point by only keeping joints in good_joint_idx
        """
        return p[:, good_joint_idx, :]

    @staticmethod
    def temporal_interp(p, copy=True):
        """
        If a joint is detected in at least one frame in a video, 
        we interpolate the nan coordinates from other frames.
        This is done independently for each joint.
        Note: it can still leave some all-nan columns if a joint is never detected in any frame.
        """
        q = p.copy() if copy else p

        for j in range(q.shape[1]): # joint
            for coord in range(q.shape[2]): # x, y (,z)
                view = q[:, j, coord]
                if np.isnan(view).all() or  not np.isnan(view).any():
                    continue
                nans, idx = nan_helper(view)
                view[nans]= np.interp(idx(nans), idx(~nans), view[~nans])
        return q

    @staticmethod
    def per_video_normalize(p, copy=True):
        """
        For x,y[, z] independently:
            Normalize into between -0.5~0.5
        """
        q = p.copy() if copy else p
        for coord in range(p.shape[2]):
            view = q[:, :, coord]
            a, b = np.nanmin(view), np.nanmax(view)
            view[:] = ((view - a) / (b-a)) - 0.5

        return q

    @staticmethod
    def fill_nan_random(p, copy=True, sigma=.5):
        """
        Fill nan values with normal distribution
        """
        q = p.copy() if copy else p
        q[np.isnan(q)] = np.random.randn(np.count_nonzero(np.isnan(q))) * sigma
        return q

    @staticmethod
    def fill_nan_uniform(p, copy=True, a=-0.5, b=0.5):
        """
        Fill nan values with normal distribution
        """
        q = p.copy() if copy else p
        q[np.isnan(q)] = np.random.random((np.count_nonzero(np.isnan(q)),)) * (b-a) + a
        return q

    @staticmethod
    def augment_nan(X, Y, num=5):
        """
        Data augmentation.
        X is a list of arrays.
        """
        Xa = []
        Ya = []
        for p1, y1 in zip(X, Y):
            Xa.extend([fill_nan_uniform(p1) for _ in range(num)])
            Ya.extend([y1] * num)

        Ya = np.stack(Ya)
        assert len(Xa) == Ya.shape[0]
        return Xa, Ya


#######################################################
## DDNet preprocessing and helper function
#######################################################

class DDNetConfig():
    def __init__(self, frame_length=32, num_joints=15, joint_dim=2, num_classes=21, num_filters=16):
        """Stores configuration of DDNet
        
        Keyword Arguments:
            frame_length {int} -- Frame length of a data point (a clip) (default: {32})
            num_joints {int} -- Number of joints detected in each frame (default: {15})
            joint_dim {int} -- Joint coordinate dimensions, should be 2 or 3 (default: {2})
            num_classes {int} -- Number of activity classes to recognize (default: {21})
            num_filters {int} -- Controls the complexity of DDNet, higher is more accurate but more compute intensive (default: {16})
        """
        self.frame_l = frame_length
        self.joint_n = num_joints
        self.joint_d = joint_dim
        self.clc_num = num_classes
        self.feat_d = int(num_joints * (num_joints-1) / 2)  # the (flatten) diemsnion of JCD
        self.filters = num_filters

def infer_DDNet(net, C, batch, *args, **kwargs):
    """Infer on a batch of clips
    
    Arguments:
        net {Model} -- a DDNet instance created by create_DDNet
        C {DDNetConfig} -- a config object
        batch {list or array} -- Each element represents the joint coordinates of a clip
        args, kwargs -- will be passed to Modle.predict()
    """
    X0, X1 = preprocess_batch(batch, C)
    return net.predict([X0, X1], *args, **kwargs)

def fit_DDNet(net, C, X, Y, *args, **kwargs):
    if type(X) in (list, tuple):
        # assume preprocessed-input
        X0, X1 = X
    else:
        print(f"Preprocessing input {type(X)}")
        X0, X1 = preprocess_batch(X, C)
    net.fit([X0, X1], Y, *args, **kwargs)

def create_DDNet(C):
    assert isinstance(C, DDNetConfig)
    return build_DD_Net(C)

def save_DDNet(net, path):
    net.save(path)

def load_DDNet(path):
    return load_model(path, custom_objects=_custom_objs)    # custom_objects is necessary


def preprocess_batch(batch, C):
    """Preprocesss a batch of points (clips)
    
    Arguments:
        batch {ndarray or list or tuple} -- List of arrays as input to preprocess_point
        C {DDNetConfig} -- A DDNetConfig object
    
    Returns:
        ndarray, ndarray -- X0, X1 to input to the net
    """
    assert type(batch) in (np.ndarray, list, tuple)
    X0 = []
    X1 = []
    for p in batch:
        px0, px1 = preprocess_point(p, C)
        X0.append(px0)
        X1.append(px1)
    X0 = np.stack(X0)
    X1 = np.stack(X1)
    return X0, X1

#######################################################
## Private functions
#######################################################

#######################################################
### Preprocessing functions
#######################################################

# Interpolate the joint coordinates of a group of frames to be target_l frames
def zoom(p,target_l=64,joints_num=25,joints_dim=3):
    """Rescale and interploate the joint coordinates of a variable number of frames to be target_l frames.
    Used prepare a fixed-size input to the net.
    
    Arguments:
        p {ndarray} -- shape = (num_frames, num_joints, joints_dim)
    
    Keyword Arguments:
        target_l {int} -- [description] (default: {64})
        joints_num {int} -- [description] (default: {25})
        joints_dim {int} -- [description] (default: {3})
    
    Returns:
        ndarray -- Rescaled array of size (target_l, num_joints, joints_dim)
    """
    l = p.shape[0]
    if l == target_l: # need do nothing
        return p
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)
    return p_new

def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)

def get_CG(p,C):
    """Compute the Joint Collection Distances (JCD, refer to the paper) of a group of frames
    and normalize them to 0 mean.
    
    Arguments:
        p {ndarray} -- size = (C.frame_l, C.num_joints, C.joints_dim)
        C {Config} -- [description]
    
    Returns:
        ndarray -- shape = (C.frame_l, C.fead_d) 
    """
    # return JCD of a point, normalized to 0 mean
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l): 
        d_m = cdist(p[f],p[f],'euclidean')       
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M) 
    M = norm_scale(M)
    return M

def preprocess_point(p, C):
    """Preprocess a single point (a clip)
    
    Arguments:
        p {ndarray} -- shape = (variable, C.joint_n, C.joint_d)
        C {DDNetConfig} -- A Config object

    Returns:
        ndarray, ndarray -- X0, X1 to input to the net
    """
    assert p.shape[1:] == (C.joint_n, C.joint_d)
    p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)
    # interploate to the right number of frames
    assert p.shape == (C.frame_l, C.joint_n, C.joint_d)
    M = get_CG(p, C)

    return M, p


#######################################################
### Model architecture
#######################################################

def poses_diff(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.image.resize_nearest_neighbor(x,size=[H.value,W.value],align_corners=False) # should not alignment here
    return x

def pose_motion(P,frame_l):
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l,-1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:,::2,...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l/2),-1))(P_diff_fast)
    return P_diff_slow,P_diff_fast
    
def c1D(x,filters,kernel):
    x = Conv1D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def block(x,filters):
    x = c1D(x,filters,3)
    x = c1D(x,filters,3)
    return x
    
def d1D(x,filters):
    x = Dense(filters,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_FM(frame_l=32,joint_n=22,joint_d=2,feat_d=231,filters=16):   
    M = Input(shape=(frame_l,feat_d))
    P = Input(shape=(frame_l,joint_n,joint_d))
    
    diff_slow,diff_fast = pose_motion(P,frame_l)
    
    x = c1D(M,filters*2,1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,3)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x_d_slow = c1D(diff_slow,filters*2,1)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,3)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
        
    x_d_fast = c1D(diff_fast,filters*2,1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,3) 
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,1) 
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
   
    x = concatenate([x,x_d_slow,x_d_fast])
    x = block(x,filters*2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)
    
    x = block(x,filters*4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x,filters*8)
    x = SpatialDropout1D(0.1)(x)
    
    return Model(inputs=[M,P],outputs=x)

def build_DD_Net(C):
    M = Input(name='M', shape=(C.frame_l,C.feat_d))  # JCD
    P = Input(name='P', shape=(C.frame_l,C.joint_n,C.joint_d)) # Cartesian
    
    FM = build_FM(C.frame_l,C.joint_n,C.joint_d,C.feat_d,C.filters)
    
    x = FM([M,P])

    x = GlobalMaxPool1D()(x)
    
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = Dense(C.clc_num, activation='softmax')(x)
    
    ######################Self-supervised part
    model = Model(inputs=[M,P],outputs=x)
    return model

# used for Keras save/load model
_custom_objs = {
    'poses_diff': poses_diff,
    'pose_motion': pose_motion,
    'c1D': c1D,
    'block': block,
    'd1D': d1D,
    'build_FM': build_FM,
    'build_DD_Net': build_DD_Net
}