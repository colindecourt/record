import os
import numpy as np
from .dataloaders import CarradaDataset

class CarradaDatasetOnline(CarradaDataset):

    def __getitem__(self, idx):
        init_frame_name = self.dataset[idx][0]
        frame_id = int(init_frame_name)
        frame_names = [str(f_id).zfill(6) for f_id in range(frame_id-self.n_frames+1, frame_id+1)]
        rd_matrices = list()
        ra_matrices = list()
        ad_matrices = list()

        rd_masks = list()
        ra_masks = list()

        for frame_name in frame_names:
            if self.process_signal:
                rd_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_doppler_processed',
                                                 frame_name + '.npy'))
                ra_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_angle_processed',
                                                 frame_name + '.npy'))
                ad_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'angle_doppler_processed',
                                                 frame_name + '.npy'))
            else:
                rd_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_doppler_raw',
                                                 frame_name + '.npy'))
                ra_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_angle_raw',
                                                 frame_name + '.npy'))
                ad_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'angle_doppler_raw',
                                                 frame_name + '.npy'))

            rd_matrices.append(rd_matrix)
            ra_matrices.append(ra_matrix)
            ad_matrices.append(ad_matrix)

            rd_masks.append(np.load(os.path.join(self.path_to_annots, frame_name, 'range_doppler.npy')))
            ra_masks.append(np.load(os.path.join(self.path_to_annots, frame_name, 'range_angle.npy')))
            # Copy for augmentations
            #ad_masks.append(np.load(os.path.join(self.path_to_annots, frame_name, 'range_doppler.npy')))


        # Apply the same transfo to all representations
        if np.random.uniform(0, 1) > 0.5:
            is_vflip = True
        else:
            is_vflip = False
        if np.random.uniform(0, 1) > 0.5:
            is_hflip = True
        else:
            is_hflip = False

        rd_matrix = np.dstack(rd_matrices)
        rd_matrix = np.rollaxis(rd_matrix, axis=-1)
        rd_mask = np.stack(rd_masks, axis=0)
        rd_mask = np.swapaxes(rd_mask, 0, 1)
        rd_frame = {'matrix': rd_matrix, 'mask': rd_mask}
        rd_frame = self.transform(rd_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'],
                                                    axis=self.add_temp)

        ra_matrix = np.dstack(ra_matrices)
        ra_matrix = np.rollaxis(ra_matrix, axis=-1)
        ra_mask = np.stack(ra_masks, axis=0)
        ra_mask = np.swapaxes(ra_mask, 0, 1)
        ra_frame = {'matrix': ra_matrix, 'mask': ra_mask}
        ra_frame = self.transform(ra_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'],
                                                    axis=self.add_temp)

        ad_matrix = np.dstack(ad_matrices)
        ad_matrix = np.rollaxis(ad_matrix, axis=-1)
        # Fill fake mask just to apply transform
        ad_frame = {'matrix': ad_matrix, 'mask': rd_mask.copy()}
        ad_frame = self.transform(ad_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                ad_frame['matrix'] = np.expand_dims(ad_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                ad_frame['matrix'] = np.expand_dims(ad_frame['matrix'],
                                                    axis=self.add_temp)

        frame = {'rd_matrix': rd_frame['matrix'], 'rd_mask': rd_frame['mask'],
                 'ra_matrix': ra_frame['matrix'], 'ra_mask': ra_frame['mask'],
                 'ad_matrix': ad_frame['matrix']}

        return frame

class CarradaDatasetRangeDopplerOnline(CarradaDataset):

    def __getitem__(self, idx):
        init_frame_name = self.dataset[idx][0]
        frame_id = int(init_frame_name)
        frame_names = [str(f_id).zfill(6) for f_id in range(frame_id-self.n_frames+1, frame_id+1)]
        rd_matrices = list()
        rd_masks = list()

        for frame_name in frame_names:
            if self.process_signal:
                rd_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_doppler_processed',
                                                 frame_name + '.npy'))
            else:
                rd_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_doppler_raw',
                                                 frame_name + '.npy'))

            rd_matrices.append(rd_matrix)
            rd_masks.append(np.load(os.path.join(self.path_to_annots, frame_name, 'range_doppler.npy')))


        # Apply the same transfo to all representations
        if np.random.uniform(0, 1) > 0.5:
            is_vflip = True
        else:
            is_vflip = False
        if np.random.uniform(0, 1) > 0.5:
            is_hflip = True
        else:
            is_hflip = False

        rd_matrix = np.dstack(rd_matrices)
        rd_matrix = np.rollaxis(rd_matrix, axis=-1)
        rd_mask = np.stack(rd_masks, axis=0)
        rd_mask = np.swapaxes(rd_mask, 0, 1)
        rd_frame = {'matrix': rd_matrix, 'mask': rd_mask}
        rd_frame = self.transform(rd_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'],
                                                    axis=self.add_temp)

        frame = {'rd_matrix': rd_frame['matrix'], 'rd_mask': rd_frame['mask']}

        return frame

class CarradaDatasetRangeAngleOnline(CarradaDataset):

    def __getitem__(self, idx):
        init_frame_name = self.dataset[idx][0]
        frame_id = int(init_frame_name)
        frame_names = [str(f_id).zfill(6) for f_id in range(frame_id-self.n_frames+1, frame_id+1)]
        ra_matrices = list()
        ra_masks = list()

        for frame_name in frame_names:
            if self.process_signal:
                ra_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_angle_processed',
                                                 frame_name + '.npy'))
            else:
                ra_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_angle_raw',
                                                 frame_name + '.npy'))

            ra_matrices.append(ra_matrix)
            ra_masks.append(np.load(os.path.join(self.path_to_annots, frame_name, 'range_angle.npy')))

        # Apply the same transfo to all representations
        if np.random.uniform(0, 1) > 0.5:
            is_vflip = True
        else:
            is_vflip = False
        if np.random.uniform(0, 1) > 0.5:
            is_hflip = True
        else:
            is_hflip = False

        ra_matrix = np.dstack(ra_matrices)
        ra_matrix = np.rollaxis(ra_matrix, axis=-1)
        ra_mask = np.stack(ra_masks, axis=0)
        ra_mask = np.swapaxes(ra_mask, 0, 1)
        ra_frame = {'matrix': ra_matrix, 'mask': ra_mask}
        ra_frame = self.transform(ra_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'],
                                                    axis=self.add_temp)


        frame = {'ra_matrix': ra_frame['matrix'], 'ra_mask': ra_frame['mask']}

        return frame