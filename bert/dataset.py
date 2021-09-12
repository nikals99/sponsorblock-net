import json
import random

import torch
from tqdm import tqdm
from bert.transcriptfetcher import TranscriptFetcher
from torch.utils.data import Dataset


class YoutubeDataSet(Dataset):

    def __init__(self, dataset, limit=10000):
        # Initialize the transcript fetcher
        fetcher = TranscriptFetcher()

        # load metadata from dataset
        with open(dataset) as jsonFile:
            videos = json.load(jsonFile)

        # shuffle all videos
        random.shuffle(videos)

        # select <limit> videos
        videos = videos[:limit]

        sequences = []
        # for each video
        for video in tqdm(videos):
            # extract sequences from transcriptfetcher
            seq = fetcher.extracttranscript(video)
            # add them to list of all sequences
            sequences.extend(seq)

        # set sequences for the current data set. when sequence has empty words array ignore it
        self.sequences = [seq for seq in sequences if len(seq['words']) > 0]

    def __getitem__(self, index):
        # get sequence for a given index
        sequence = self.sequences[index]

        # split sequence into its values and return
        return {
            'input_ids': sequence['input_ids'],
            'attention_mask': sequence['attention_mask'],
            'label': torch.tensor(sequence['token_labels'])
        }

    def __len__(self):
        # simply returns the amount of sequences
        return len(self.sequences)
