import glob
import os

import numpy as np
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of 0 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)      # if dim=2, (sidelen, sidelen) / dim=3, (sidelen, sidelen, sidelen)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)  # shape -> (1, sidelen, sidelen, 2)
        # Normalization
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)    # shape -> (1, sidelen, sidelen, 3)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords



class VideoTime(Dataset):
    def __init__(self, path_to_video, split_num=300):
        super().__init__()

        self.split_num = split_num
        print("[i] Video loading start......")
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            print(" videos")
            self.vid = skvideo.io.vread(path_to_video).astype(np.single)  # shape -> (frame #, y, x, channels)
            if "timelapse" in path_to_video:
                self.vid = self.vid[40:, :, :, :]
            if "GOPR" in path_to_video:
                self.vid = self.vid[:600]
                self.vid = self.vid[0::2]
        else:
            print(" imgs")
            video_path = os.path.join(path_to_video, "*.jpg")
            files = sorted(glob.glob(video_path))[:self.split_num]
            tmp_img = Image.open(files[0])
            tmp_img  = np.array(tmp_img)
            tmp_shape = tmp_img.shape

            self.vid = np.zeros((self.split_num, tmp_shape[0], tmp_shape[1], tmp_shape[2]), dtype=np.uint8)
            for idx, f in enumerate(files):
                img = Image.open(f)
                img = np.array(img)
                self.vid[idx] = img
                
        print("[i] Finished")

        self.shape = self.vid.shape[1:-1]
        
        self.nframes = self.vid.shape[0]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid



class VideoTimeWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None):

        self.dataset = dataset              # class 'dataio.VideoTime'
        nframes = self.dataset.nframes      # 300
        self.sidelength = sidelength        # (360, 640)
        
        self.mgrid = get_mgrid(sidelength, dim=2) # [w * h, 2] in range [0, 1] ((0, 0) to (1, 1))

        data = torch.from_numpy(self.dataset[0])
        self.data = data.view(self.dataset.nframes, -1, self.dataset.channels) # [f, w * h, 3]

        # batch 
        # self.N_samples = 1245184
        self.pixel_num = 360*640

        half_dt =  0.5 / nframes

        # modulation input
        self.temporal_steps = torch.linspace(half_dt, 1-half_dt, self.dataset.nframes)
        
        # temporal coords
        self.temporal_coords = torch.linspace(0, 1, nframes)
        
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch_frames = 5
        temporal_coord = []
        for frame in range(batch_frames):
            if self.epoch + frame >= self.dataset.nframes:
                temp = temp = torch.full((self.pixel_num,), self.epoch + frame - self.dataset.nframes, dtype=torch.int64)
            else: temp = torch.full((self.pixel_num,), self.epoch + frame, dtype=torch.int64)
            temporal_coord.append(temp)
        temporal_coord_idx = torch.cat(temporal_coord, dim=0)
        
        self.epoch += 1
        if self.epoch == self.dataset.nframes:
            self.epoch = 0
            
        spatial_coord = []
        for frame in range(batch_frames):
            temp = torch.arange(0, self.pixel_num)
            spatial_coord.append(temp)
        spatial_coord_idx = torch.cat(spatial_coord, dim=0) # spatial coordinate in index / (0, 1, ..., 360*640-1) * 5 times
        
        data = self.data[temporal_coord_idx, spatial_coord_idx, :] # [t, (x,y), rgb]
        
        spatial_coords = self.mgrid[spatial_coord_idx, :]  # spatial coordinates / ((0, 0), ..., (1, 1)) * 5 times
        temporal_coords = self.temporal_coords[temporal_coord_idx] # Frame index in [0, 1]
        
        temporal_steps = self.temporal_steps[temporal_coord_idx] # temporal steps in sec.

        all_coords = torch.cat((temporal_coords.unsqueeze(1), spatial_coords), dim=1) # (frame, (x, y)) * (360*640)
            
        in_dict = {'all_coords': all_coords, "temporal_steps": temporal_steps}
        gt_dict = {'img': data}

        return in_dict, gt_dict


'''
    def __getitem__(self, idx):

        temporal_coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,)) 
        spatial_coord_idx = torch.randint(0, self.data.shape[1], (self.N_samples,))
        data = self.data[temporal_coord_idx, spatial_coord_idx, :] 
        
        spatial_coords = self.mgrid[spatial_coord_idx, :] 
        temporal_coords = self.temporal_coords[temporal_coord_idx] 
        
        temporal_steps = self.temporal_steps[temporal_coord_idx]

        all_coords = torch.cat((temporal_coords.unsqueeze(1), spatial_coords), dim=1)
            
        in_dict = {'all_coords': all_coords, "temporal_steps": temporal_steps}
        gt_dict = {'img': data}

        return in_dict, gt_dict
'''