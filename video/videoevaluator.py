import torch

from video.model import SponsorNet
from video.videofetcher import VideoFetcher
from torch.utils.data import DataLoader

class VideoEvaluator:
    def __init__(self, checkpoint_path, metadata_json, data_dir):
        self.fetcher = VideoFetcher(metadata_json, data_dir)
        self.model = SponsorNet.load_from_checkpoint(checkpoint_path)
        dev = "cuda:0" if torch.cuda.is_available() else "cpu" 
        self.device = torch.device(dev)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, videoId):
        samples = self.fetcher.get_video_data(videoId)
        result = []
        loader = DataLoader(samples, batch_size=50)
        for batch in loader:
            out = self.model(batch.to(self.device))
            out = torch.sigmoid(out)
            for entry in out:
                result.append(entry)
        return result