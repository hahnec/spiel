from torch.utils.data import Dataset
from pathlib import Path
from natsort import natsorted
import numpy as np


sample2dist = lambda x, c=345, fkHz=175.642, sample_rate=1: c/2 * x / sample_rate / fkHz
dist2sample = lambda d, c=345, fkHz=175.642, sample_rate=1: 2/c * d * fkHz * sample_rate


class SpielDataset(Dataset):
    
    def __init__(
            self, 
            dataset_path='', 
            transform=None, 
            flip=False, 
            bg_subtract=False, 
            blind_zone_idx: int = 0,
            skip_transducer_idcs = None,
            x_position_gap = 1,
            y_position_gap = 1,
            z_position_gap = 1,
            pow_law_opt = False,
            x_range = None,
            y_range = None,
            z_range = None,
            c = 343,
            fkHz = 22,
            ):

        self.dataset_path = Path(dataset_path)
        print(self.dataset_path.resolve())

        self.transform = transform
        self.flip = flip
        self.subtract = bg_subtract
        self.bidx = blind_zone_idx
        self.skip_transducer_idcs = skip_transducer_idcs
        self.x_position_gap = x_position_gap
        self.y_position_gap = y_position_gap
        self.z_position_gap = z_position_gap
        self.pow_law_opt = pow_law_opt
        self.c = c
        self.fkHz = fkHz

        # selected ground-truth position range
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.frame_filenames = []
        self.label_filenames = []

        self.piezo_positions = self.get_sensor_coords(radius=75)

        self.read_filenames()

    def read_filenames(self):

        self.frame_filenames = natsorted([str(fname.name) for fname in self.dataset_path.iterdir() if str(fname.name).lower().endswith('frames.csv')])
        self.label_filenames = natsorted([str(fname.name) for fname in self.dataset_path.iterdir() if str(fname.name).lower().endswith('labels.csv')])

        labels_list = [np.loadtxt(str(self.dataset_path / name), dtype=object, delimiter=' ') for name in self.label_filenames]
        if len(labels_list) > 0: gt_positions = np.array(labels_list)[:, 0, 3:6].astype(float)

        if self.x_range is not None and self.y_range is not None and self.z_range is not None:

            conditions = \
                (self.x_range[0] <= gt_positions[:, 0]) & (gt_positions[:, 0] < self.x_range[1]) & \
                (self.y_range[0] <= gt_positions[:, 1]) & (gt_positions[:, 1] < self.y_range[1]) & \
                (self.z_range[0] <= gt_positions[:, 2]) & (gt_positions[:, 2] < self.z_range[1]) & \
                (abs((gt_positions[:, 0]-self.x_range[0]) % self.x_position_gap) == 0) & \
                (abs((gt_positions[:, 1]-self.y_range[0]) % self.y_position_gap) == 0) & \
                (abs((gt_positions[:, 2]-self.z_range[0]) % self.z_position_gap) == 0)      # remove measurements inbetween position gap

            self.frame_filenames = [f for c, f in zip(conditions, self.frame_filenames) if c]
            self.label_filenames = [f for c, f in zip(conditions, self.label_filenames) if c]

    @staticmethod
    def get_eq_triangle_coords(centroid=(0, 0), radius: float = 1):

        side_length = radius * 3**.5
        a = [centroid[0], centroid[1] + (np.sqrt(3) / 3) * side_length]  # top vertex
        b = [centroid[0] - (side_length / 2), centroid[1] - (3**.5 / 6) * side_length]  # bottom left vertex
        c = [centroid[0] + (side_length / 2), centroid[1] - (3**.5 / 6) * side_length]  # bottom right vertex

        return a, b, c
    
    def get_sensor_coords(self, radius=75):
         # get triangle coords while adding another dimension
        sensors_xyz = np.hstack([np.array(self.get_eq_triangle_coords(radius=radius)), np.zeros([3, 1])])
        sensors_xyz[:, :2] = sensors_xyz[:, :2][:, ::-1]    # swap dimensions
        return np.vstack([np.zeros([1, 3]), sensors_xyz[1], sensors_xyz[0], sensors_xyz[2]])

    def rectify_power(self, frames, a=159.60594527030352, b=1.5636590485622333):

        x = np.arange(0, frames.shape[-1])+1
        g = a/x**b

        return frames/g[None, :]

    def __getitem__(self, idx):

        # load data frames
        frames_fname, labels_fname = (self.frame_filenames[idx], self.label_filenames[idx])
        frames = np.loadtxt(str(self.dataset_path / frames_fname), dtype=object , delimiter=' ')[:, :-2].astype(float)
        labels = np.loadtxt(str(self.dataset_path / labels_fname), dtype=object, delimiter=' ')

        # convert label data to ground-truth representation(s)
        gt_position = labels[0, 3:6].astype(float) if labels[0, 3:6].size > 0 else np.zeros(3)

        # compute reference distances (backward path + forward path)
        gt_distances = ((gt_position[:3]-self.piezo_positions)**2).sum(-1)**.5 + ((gt_position[:3])**2).sum(-1)**.5
        gt_samples = dist2sample(gt_distances, c=self.c, fkHz=self.fkHz)/2    # divide by 2 to account for factor 2 in round-trip distance equation
        tdk_distances = labels[:, 1].astype(float)
        tdk_samples = dist2sample(tdk_distances, c=self.c, fkHz=self.fkHz)/2

        # background subtraction
        if self.subtract != 0:
            bg_frames = np.loadtxt(str(Path(str(self.dataset_path)+'_bg') / frames_fname), dtype=object , delimiter=' ')[:, :-2].astype(float)
            frames -= bg_frames * self.subtract
            frames[frames<0] = 0

        # remove unwanted sensor (e.g. transmitter)
        if self.skip_transducer_idcs is not None:
            frames = np.delete(frames, self.skip_transducer_idcs, axis=0)
            gt_samples = np.delete(gt_samples, self.skip_transducer_idcs, axis=0)

        # blind zone rejection
        frames[..., :self.bidx] = 0

        # rectify power law
        frames = self.rectify_power(frames, a=159.60594527030352, b=1.5636590485622333) if self.pow_law_opt else frames

        if self.flip:
            frames = frames[:, ::-1]
            labels = len(frames[0, ...]) - labels
        
        return frames, gt_position, gt_samples, frames_fname
    
    def __len__(self):
        return len(self.frame_filenames)
