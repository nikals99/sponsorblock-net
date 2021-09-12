import torch

from bert.classifier import BertClassifier
from bert.transcriptfetcher import TranscriptFetcher


class SliceEvaluatorBert:
    def __init__(self, model, datadir="../data/", snippet_len=400):
        self.datadir = datadir
        # Initialize the transcriptfetcher
        self.fetcher = TranscriptFetcher(datadir=datadir, snippet_len=snippet_len)

        # Initialize our Classifier
        self.model = BertClassifier(f"./{model}")

        # Check if we should run on cpu or gpu
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        # Move model to the selected device
        self.device = torch.device(dev)
        self.model.to(self.device)
        # Set model to eval mode
        self.model.eval()

    def evaluateOverlapping(self, video):
        # first do a normal prediction
        pred1 = torch.tensor(self.evaluate(video))
        # do a second prediction with a fixed offset
        pred2 = torch.tensor(self.evaluate(video, offset=200))

        # Maxpooling to get the final result
        return [max(p1, p2) for p1, p2 in zip(pred1, pred2)]

    def evaluate(self, video, offset=0):
        # Extract sequences for the given video and offset
        sequences = self.fetcher.extracttranscript(video, offset=offset)

        # prepare the result array
        sec_result = [-10] * video["duration"]

        # if no sequences could be extracted we return no sponsor segments
        if len(sequences) == 0:
            return [0] * video["duration"]

        # initialize the timestamp counter with a negative value
        last_timestamp = -1

        # iterate over each sequence
        for seq in sequences:
            # get variables from current sequence
            labels = seq['token_labels']
            times = seq['times']
            input_ids = seq['input_ids']
            attention_mask = seq['attention_mask']

            # unsqueeze the tensors so that it matches the format expexted by our model
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

            # get predictions from our model + move tensors to device first
            out = self.model.bert(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))

            # finally we need to do convert from words/tokens to seconds we start with second -1

            word_count = 0
            # Loop over all token/label pairs in the current sequence
            for i in range(0, len(labels)):
                # If the current word got filled during tokenization (e.g. BPE or Padding) skip it
                if labels[i] == -100:
                    continue

                # Get the actual probability of being sponsor by doing a softmax over both logits
                loga = torch.softmax(torch.tensor([out['logits'][0][i][0], out['logits'][0][i][1]]), dim=0)[1]

                # set the current timestamp to the timestamp reported by the current word
                current_timestamp = times[word_count]

                # if sponsorblock thinks that the video is short (this happens sometimes by 1-3 seconds) stop converting
                if current_timestamp >= video["duration"]:
                    break

                # fill all unset timestamps till the current timestamp with the current prediction
                while last_timestamp < current_timestamp:
                    last_timestamp += 1
                    sec_result[last_timestamp] = loga
                # increase wordcount
                word_count += 1

        # Usually the video is longer than something is spoken so we need to fill in the rest of the timestamps with the last prediction
        filler = sec_result[last_timestamp]
        while last_timestamp < video["duration"] - 1:
            last_timestamp += 1
            sec_result[last_timestamp] = filler

        return sec_result
