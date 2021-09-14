import os
import cv2
import json
import pickle
import sqlite3

import pytorch_lightning as pl
import pandas as pd

from functools import reduce
from torch.utils.data import DataLoader

class SponsorNetVideoDatasetPreloaded():
    def __init__(self, samples_file_path):
        sample_file = open(samples_file_path, "rb")
        self.samples = pickle.load(sample_file)
        sample_file.close()
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        return self.samples[index]


class SponsorNetVideoDataset():
    def __init__(self, metadata_json, sponsortimes_db, data_dir, transform = None, seconds_per_sample = 10, limit_videos = None):
        self.data_dir = data_dir
        self.seconds_per_sample = seconds_per_sample
        self.transform = transform
        self.cache = {}

        con = sqlite3.connect(os.path.join(self.data_dir, sponsortimes_db))
        self.sponsortimes = pd.read_sql_query("SELECT * FROM sponsorTimes", con)
        con.close()
        
        metadata_file = open(os.path.join(self.data_dir, metadata_json), "r")
        self.metadata = json.loads(metadata_file.read())
        metadata_file.close()

        if limit_videos is not None:
            self.metadata = self.metadata[:limit_videos]

        sample_counts = [entry["duration"] // self.seconds_per_sample for entry in self.metadata]
        self.starting_counts = list(zip(
            reduce(lambda acc, val: acc + [acc[-1] + val], sample_counts, [0]),
            self.metadata + [None] # Since the upper list contains one additional element, we have to append one here as well
        ))
        self.num_samples = sum(sample_counts)
        print("num_samples", self.num_samples)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):
        if index in self.cache.keys():
            return self.cache[index]
        sample = self.get_item(index)
        self.cache[index] = sample
        return sample 
   

    def get_item(self, idx):
        for i in range(len(self.starting_counts)):
          if idx < self.starting_counts[i + 1][0]:
            sample_pos_sec = (idx - self.starting_counts[i][0]) * self.seconds_per_sample # e.g. sample occurs at second 3 in the video
            sample_pos_msec = sample_pos_sec * 1000
            sample_video_metadata = self.starting_counts[i][1]

            # extract one specific frame from this video
            cap = cv2.VideoCapture(os.path.join(self.data_dir, sample_video_metadata["file_path"]))
            cap.set(0, sample_pos_msec)
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not capture frame for index {}".format(idx))
            cap.release()

            # extract the label from the sponsortimes csv
            label = 0.0 # no_sponsor
            possible_categories = self.sponsortimes[
                (self.sponsortimes["videoID"] == sample_video_metadata["id"]) &
                (self.sponsortimes["startTime"] <= sample_pos_sec) &
                (self.sponsortimes["endTime"] >= sample_pos_sec) &
                (self.sponsortimes["category"] == "sponsor") # only use sponsor segments for now
            ]
            if len(possible_categories) > 0:
                label = 1.0 # sponsor (possible_categories.iloc[0]["category"])

            if self.transform:
                frame = self.transform(frame)
                frame = frame.half()

            return (frame, label)

        raise ValueError("Could not find video for sample with index {}".format(idx))


class SponsorNetVideoDataModule(pl.LightningDataModule):
    def __init__(self, training_sample_file_path, validation_sample_file_path, test_sample_file_path, batch_size = 32, num_workers = 1):
        super().__init__()
        self.training_sample_file_path = training_sample_file_path
        self.validation_sample_file_path = validation_sample_file_path
        self.test_sample_file_path = test_sample_file_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # TODO: evtl hier drin die datasets initialisieren?
        pass

    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = SponsorNetVideoDatasetPreloaded(self.training_sample_file_path)
            self.val_data = SponsorNetVideoDatasetPreloaded(self.validation_sample_file_path)
            self.dims = tuple(self.train_data[0][0].shape)
            pass
        
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = SponsorNetVideoDatasetPreloaded(self.test_sample_file_path)
            self.dims = tuple(self.test_data[0][0].shape)
            pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)