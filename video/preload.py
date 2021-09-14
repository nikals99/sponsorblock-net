import os
import sys
import cv2
import json
import pickle
import random
import sqlite3
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool
from torchvision import transforms
from itertools import chain

NUM_CORES = 32

def load_metadata(metadata_json, data_dir):
    metadata_file = open(os.path.join(data_dir, metadata_json), "r")
    metadata = json.loads(metadata_file.read())
    metadata_file.close()
    return metadata


def load_sponsortimes(sponsortimes_db, data_dir):
    con = sqlite3.connect(os.path.join(data_dir, sponsortimes_db))
    sponsortimes = pd.read_sql_query("SELECT * FROM sponsorTimes", con)
    con.close()
    return sponsortimes


def preload_lstm_multiprocess(metadata_json, sponsortimes_db, data_dir, transform=None, limit_videos=None):
    """
    Starts NUM_CORES lstm preload processes.
    """
    metadata = load_metadata(metadata_json, data_dir)

    if limit_videos is not None:
        metadata = random.sample(metadata, k=limit_videos)

    with Pool(processes=NUM_CORES) as pool:
        return pool.starmap(preload_lstm_one, [(entry, sponsortimes_db, data_dir, transform) for entry in metadata])


def preload_lstm_one(entry, sponsortimes_db, data_dir, transform=None):
    """
    Extracts one frame per second from a video. Returns a list of samples, each one
    containing a frame from the video and a label.
    """
    
    samples_for_video = []
    
    sponsortimes = load_sponsortimes(sponsortimes_db, data_dir)

    # Load the video
    cap = cv2.VideoCapture(os.path.join(data_dir, entry["file_path"]))
    
    # preload the metadata for this video
    sponsortimes_video_info = sponsortimes[sponsortimes["videoID"] == entry["id"]]

    # read some info about the current video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # extract the middle frame from each one second slice of the video
    for frame_idx in range(fps // 2, frame_count, fps):
        # seek to this frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # capture the frame
        ret, frame = cap.read()
        if not ret:
            break

        # apply the transform
        if transform:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frame = frame.half()

        # determine the label
        sample_pos_sec = frame_idx * (1/fps)
        label = 0.0 # no_sponsor
        possible_categories = sponsortimes_video_info[
            (sponsortimes_video_info["startTime"] <= sample_pos_sec) &
            (sponsortimes_video_info["endTime"] >= sample_pos_sec) &
            (sponsortimes_video_info["category"] == "sponsor")
        ]
        if len(possible_categories) > 0:
            label = 1.0 # sponsor (possible_categories.iloc[0]["category"])
        samples_for_video.append((frame, label))
    cap.release()
    return samples_for_video


def preload_lstm(metadata_json, sponsortimes_db, data_dir, transform = None, limit_videos = None):
    """
    Extracts frames from input videos. This is done by partitioning a video into one second
    slices and picking the middle frame of every slice. Each video will represent one sample in the final
    database, each sample containing a list of frames along with their associated ground truth label.
    """
    metadata = load_metadata(metadata_json, data_dir)

    sponsortimes = load_sponsortimes(sponsortimes_db, data_dir)

    if limit_videos is not None:
        metadata = random.sample(metadata, k=limit_videos)

    results = []
    for entry in tqdm(metadata):
        samples_for_video = []

        # Load the video
        cap = cv2.VideoCapture(os.path.join(data_dir, entry["file_path"]))
        
        # preload the metadata for this video
        sponsortimes_video_info = sponsortimes[sponsortimes["videoID"] == entry["id"]]

        # read some info about the current video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # extract the middle frame from each one second slice of the video
        for frame_idx in range(fps // 2, frame_count, fps):
            # seek to this frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # capture the frame
            ret, frame = cap.read()
            if not ret:
                break

            # apply the transform
            if transform:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frame = frame.half()

            # determine the label
            sample_pos_sec = frame_idx * (1/fps)
            label = 0.0 # no_sponsor
            possible_categories = sponsortimes_video_info[
                (sponsortimes_video_info["startTime"] <= sample_pos_sec) &
                (sponsortimes_video_info["endTime"] >= sample_pos_sec) &
                (sponsortimes_video_info["category"] == "sponsor")
            ]
            if len(possible_categories) > 0:
                label = 1.0 # sponsor (possible_categories.iloc[0]["category"])
            samples_for_video.append((frame, label))
        results.append(samples_for_video)
    return results


def preload_multiprocess(metadata_json, sponsortimes_db, data_dir, limit_videos=None, sponsor_stepsize_ms = 1000, non_sponsor_stepsize_ms = 1000):
    """
    Starts NUM_CORES preload processes.
    """
    
    metadata = load_metadata(metadata_json, data_dir)

    if limit_videos is not None:
        metadata = random.sample(metadata, k=limit_videos)

    with Pool(processes=NUM_CORES) as pool:
        results = pool.starmap(preload_one, [(entry, sponsortimes_db, data_dir, sponsor_stepsize_ms, non_sponsor_stepsize_ms) for entry in metadata])
        return list(chain.from_iterable(results))


def preload_one(entry, sponsortimes_db, data_dir, sponsor_stepsize_ms = 1000, non_sponsor_stepsize_ms = 1000):
    """
    Extracts the frames from one video using two different sample rates: If the current frame is withing
    a sponsored segment, sponsor_stepsize_ms will be used to forward seek the video, otherwise non_sponsor_stepsize_ms
    will be used.
    """

    # using the method in a multiprocessed environment hangs if the transform is
    # passed as a parameter, which is why it is hardcoded here.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    samples_for_video = []
    
    sponsortimes = load_sponsortimes(sponsortimes_db, data_dir)

    # Load the video
    cap = cv2.VideoCapture(os.path.join(data_dir, entry["file_path"]))
    
    # preload the metadata for this video
    sponsortimes_video_info = sponsortimes[sponsortimes["videoID"] == entry["id"]]

    current_time_ms = 0
    while True:
        # sample this frame
        ret, frame = cap.read()
        if not ret:
            break

        # determine the label
        sample_pos_sec = current_time_ms // 1000
        label = 0.0 # no_sponsor
        possible_categories = sponsortimes_video_info[
            (sponsortimes_video_info["startTime"] <= sample_pos_sec) &
            (sponsortimes_video_info["endTime"] >= sample_pos_sec) &
            (sponsortimes_video_info["category"] == "sponsor") # only use sponsor segments for now
        ]
        if len(possible_categories) > 0:
            label = 1.0 # sponsor (possible_categories.iloc[0]["category"])

        
        # apply the transform
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frame = frame.half()     

        samples_for_video.append((frame, label))

        # seek to the next frame
        current_time_ms += sponsor_stepsize_ms if label == 1.0 else non_sponsor_stepsize_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
    cap.release()
    return samples_for_video


def preload(metadata_json, sponsortimes_db, data_dir, transform = None, limit_videos = None, sponsor_stepsize_ms = 1000, non_sponsor_stepsize_ms = 1000):
    """
    Preloads and transforms all samples using opencv. The resulting samples array can be saved to disk using e.g. pickle.
    The metadata and sponsortimes database are assumed to be located in the root of data_dir. The two step sizes control
    the selection of the next frame. If the current sample is withing a sponsored segment, the sponsor_stepsize_ms will be used.
    Otherwise non_sponsor_stepsize_ms will be used. If limit_videos is provided, a random choice of videos is taken from the metadata file.
    """
    metadata = load_metadata(metadata_json, data_dir)

    sponsortimes = load_sponsortimes(sponsortimes_db, data_dir)

    if limit_videos is not None:
        metadata = random.sample(metadata, k=limit_videos)

    samples = []
    num_sponsor_samples = 0
    num_non_sponsor_samples = 0
    for entry in tqdm(metadata):
        # Load the video
        cap = cv2.VideoCapture(os.path.join(data_dir, entry["file_path"]))
        sponsortimes_video_info = sponsortimes[sponsortimes["videoID"] == entry["id"]]

        # sample frames until we reached the end of the video
        current_time_ms = 0
        while True:
            # sample this frame
            ret, frame = cap.read()
            if not ret:
                break

            # determine the label
            sample_pos_sec = current_time_ms // 1000
            label = 0.0 # no_sponsor
            possible_categories = sponsortimes_video_info[
                (sponsortimes_video_info["startTime"] <= sample_pos_sec) &
                (sponsortimes_video_info["endTime"] >= sample_pos_sec) &
                (sponsortimes_video_info["category"] == "sponsor") # only use sponsor segments for now
            ]
            if len(possible_categories) > 0:
                label = 1.0 # sponsor (possible_categories.iloc[0]["category"])
                num_sponsor_samples += 1
            else:
                num_non_sponsor_samples += 1
            
            # apply the transform
            if transform:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frame = frame.half() 

            samples.append((frame, label))

            # seek to the next frame
            current_time_ms += sponsor_stepsize_ms if label == 1.0 else non_sponsor_stepsize_ms
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
        cap.release()
    print("#samples: {}\t#sponsor: {}\t#non_sponsor: {}".format(num_sponsor_samples + num_non_sponsor_samples, num_sponsor_samples, num_non_sponsor_samples))
    return samples


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    training_samples = preload(
        "training.json",
        "database.db",
        sys.argv[1],
        transform=transform,
        limit_videos = 100,
        sponsor_stepsize_ms=2000,
        non_sponsor_stepsize_ms=30000
    )
    file = open("training_samples_baseline.pickle", "wb")
    pickle.dump(training_samples, file)
    file.close()

    validation_samples = preload(
        "validation.json",
        "database.db",
        sys.argv[1],
        transform=transform,
        limit_videos = 300,
        sponsor_stepsize_ms=2000,
        non_sponsor_stepsize_ms=30000
    )
    file = open("validation_samples_baseline.pickle", "wb")
    pickle.dump(validation_samples, file)
    file.close()

    test_samples = preload(
        "test.json",
        "database.db",
        sys.argv[1],
        transform=transform,
        limit_videos = 300,
        sponsor_stepsize_ms=2000,
        non_sponsor_stepsize_ms=30000
    )
    file = open("test_samples_baseline.pickle", "wb")
    pickle.dump(validation_samples, file)
    file.close()
