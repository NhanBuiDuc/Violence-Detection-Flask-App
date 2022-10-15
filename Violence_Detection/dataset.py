import torch.utils.data as data
import numpy as np

from Violence_Detection.utils import process_feat

r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
"""
class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False):

        # Detection mode
        self.modality = args.modality
        # test_mode decides either the dataset is train set or test set
        if test_mode:
            # Parse test list paths from args
            self.rgb_list_file = args.test_rgb_list
            self.flow_list_file = args.test_flow_list
            self.audio_list_file = args.test_audio_list
        else:
            # Parse train list paths from args
            self.rgb_list_file = args.rgb_list
            self.flow_list_file = args.flow_list
            self.audio_list_file = args.audio_list
        
        self.max_seqlen = args.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.normal_flag = '_label_A'
        self._parse_list()

    r""" _parse_list
    Parse the list with corresponding to the modality
    """
    def _parse_list(self):
        # Audio Only
        if self.modality == 'AUDIO':
            self.list = list(open(self.audio_list_file))
        # RGB Only
        elif self.modality == 'RGB':
            self.list = list(open(self.rgb_list_file))
        # Optical Flow only
        elif self.modality == 'FLOW':
            self.list = list(open(self.flow_list_file))
        #  RGB + Optical Flow
        elif self.modality == 'MIX':
            self.list = list(open(self.rgb_list_file))
            self.flow_list = list(open(self.flow_list_file))
        # RGB + Audio
        elif self.modality == 'MIX2':
            self.list = list(open(self.rgb_list_file))
            self.audio_list = list(open(self.audio_list_file))
        # Optical flow + Audio
        elif self.modality == 'MIX3':
            self.list = list(open(self.flow_list_file))
            self.audio_list = list(open(self.audio_list_file))
        # RGB + Optical flow + Audio
        elif self.modality == 'MIX_ALL':
            self.list = list(open(self.rgb_list_file))
            self.flow_list = list(open(self.flow_list_file))
            self.audio_list = list(open(self.audio_list_file))
        else:
            assert 1 > 2, 'Modality is wrong!'

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0

        if self.modality == 'AUDIO':
            features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        elif self.modality == 'RGB':
            features = np.array(np.load(self.list[index].strip('\n')),dtype=np.float32)
        elif self.modality == 'FLOW':
            features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        elif self.modality == 'MIX':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2),axis=1)
            else:# because the frames of flow is one less than that of rgb
                features = np.concatenate((features1[:-1], features2), axis=1)
        elif self.modality == 'MIX2':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2),axis=1)
            else:# because the frames of flow is one less than that of rgb
                features = np.concatenate((features1[:-1], features2), axis=1)

        elif self.modality == 'MIX3':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2),axis=1)
            else:# because the frames of flow is one less than that of rgb
                features = np.concatenate((features1[:-1], features2), axis=1)
        elif self.modality == 'MIX_ALL':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
            features3 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2, features3),axis=1)
            else:# because the frames of flow is one less than that of rgb
                features = np.concatenate((features1[:-1], features2, features3[:-1]), axis=1)
        else:
            assert 1>2, 'Modality is wrong!'
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features

        else:
          pass
          try:
            features = process_feat(features, self.max_seqlen, is_random=False)
            return features, label
          except Exception as e: print(e)

            
    def __len__(self):
        return len(self.list)
