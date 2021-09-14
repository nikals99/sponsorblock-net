"""
Loads a video from disk, extracts one frame per second and returns and
array of samples. Each sample containing a video frame and a label (if this
frame is from sponsored content or not)
"""

import os
import cv2
import json
import numpy as np

from torchvision import transforms
from multiprocessing import Pool
from itertools import chain

NUM_CORES = 32


def load_chunk(video_path, frame_idxs):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_idx in frame_idxs:
        # seek to this frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # capture the frame
        ret, frame = cap.read()
        if not ret:
            break

        # apply the transform
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


class VideoFetcher:
    def __init__(self, metadata_json, data_dir):
        self.data_dir = data_dir
        
        # load the metadata file
        metadata_file = open(os.path.join(data_dir, metadata_json), "r")
        self.metadata = self._prepare_metadata(json.loads(metadata_file.read()))
        metadata_file.close()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def _prepare_metadata(self, raw_metadata):
        metadata = {}
        for entry in raw_metadata:
            metadata[entry["id"]] = entry
        return metadata


    def _get_video_metadata(self, videoId):
        return self.metadata.get(videoId, None)


    def get_video_data(self, videoId):
        video_metadata = self._get_video_metadata(videoId)
        if video_metadata is None:
            print("no video metadata found for {}".format(videoId))
            return []

        # Load the video
        video_path = os.path.join(self.data_dir, video_metadata["file_path"])
        cap = cv2.VideoCapture(video_path)

        # read some info about the current video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # compute the list of frame indices to be extraced
        frame_idxs = list(range(fps // 2, frame_count, fps))
        frame_idx_chunks = np.array_split(frame_idxs, NUM_CORES)

        # dispatch NUM_CORES processes to extract the frames
        with Pool(processes=NUM_CORES) as pool:
            results = pool.starmap(load_chunk, [(video_path, chunk) for chunk in frame_idx_chunks])
            results = list(chain.from_iterable(results))            
            results = list(map(lambda x: self.transform(x), results))
            return results
        