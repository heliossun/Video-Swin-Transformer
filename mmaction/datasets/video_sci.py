import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from .builder import DATASETS
@DATASETS.register_module()
class VideoSCIDataset(Dataset):
    """Video dataset for reconstruction.

    The dataset loads mat file which include ground truth and measurements


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, VideoSCIDataset, test_mode=False):
        super().__init__()
        self.path = path
        self.test_mode = test_mode
        self.load_annotation()
    def load_annotation(self):
        self.data = []
        if os.path.exists(self.path):
            groung_truth_path = self.path + '/gt'
            measurement_path = self.path + '/measurement'

            if os.path.exists(groung_truth_path) and os.path.exists(measurement_path):
                groung_truth = os.listdir(groung_truth_path)
                measurement = os.listdir(measurement_path)
                self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i],
                              'measurement': measurement_path + '/' + measurement[i]} for i in range(len(groung_truth))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')
    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        groung_truth, measurement = self.data[idx]["groung_truth"], self.data[idx]["measurement"]
        gt = sio.loadmat(groung_truth)
        meas = sio.loadmat(measurement)
        if "patch_save" in gt:
            gt = torch.from_numpy(gt['patch_save'] / 255)
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)
        meas = torch.from_numpy(meas['meas'] / 255)
        gt = gt.permute(2, 0, 1)
        return gt, meas

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        groung_truth, measurement = self.data[idx]["groung_truth"], self.data[idx]["measurement"]
        gt = sio.loadmat(groung_truth)
        meas = sio.loadmat(measurement)
        if "patch_save" in gt:
            gt = torch.from_numpy(gt['patch_save'] / 255)
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)
        meas = torch.from_numpy(meas['meas'] / 255)
        gt = gt.permute(2, 0, 1)
        return gt, meas
    def __getitem__(self, idx):
        if not self.test_mode:
            return self.prepare_train_data(idx)
        return self.prepare_test_data(idx)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        imgs_root = self.path
        num_imgs = len(self)
        return (f'total {num_imgs} images in '
                f'imgs_root: {imgs_root}')