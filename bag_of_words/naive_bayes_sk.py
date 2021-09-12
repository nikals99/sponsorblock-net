import sys
# append the path of the parent directory
sys.path.append("..")

import argparse
import json
import pickle
from prettytable import PrettyTable

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm import tqdm

from bag_of_words.transcriptfetcher import TranscriptFetcher
import random

class NaiveBayesSKLearn:
    def __init__(self):
        self.datadir = "../data/"
        self.snippetextractor = TranscriptFetcher()
        self.target_names = ["sponsor", "non_sponsor"]
        self.text_clf:Pipeline = None

    def preparedataset(self, dataset):
        # Load metadata
        with open(f"{self.datadir}{dataset}.json") as jsonFile:
            videos = json.load(jsonFile)

        print(f"Extracting snippets from {dataset}")
        targets = []
        texts = []
        snippets = []
        # Randomize videos
        random.shuffle(videos)
        # Select x videos
        videos = videos[:10000]

        # Extract snippets for each video
        for video in tqdm(videos):
            s, tar, txt = self.snippetextractor.extracttranscript(video)
            texts.extend(txt)
            targets.extend(tar)
            snippets.extend(s)

        # Sanity check
        assert len(targets) == len(texts)

        # Log some information
        sponsors_count = sum(map(lambda x: x == 0, targets))
        print(f"Loaded {len(targets)} snippets which contain {sponsors_count} sponsor snippets")

        print("Done extracting text snippets")

        return texts, targets, snippets

    def train(self):
        # Load snippets from dataset
        texts, targets, snippets = self.preparedataset("training")

        # Create the classifier pipeline
        self.text_clf = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.0001))

        print("Training naive bayes classifier")
        # Actually train the pipeline
        self.text_clf.fit(texts, targets)

    def predict(self, video):
        print("Starting prediction")
        # Load snippets for a specific video
        snippets, targets, texts = self.snippetextractor.extracttranscript(video)

        # get predictions from model
        predictions = self.text_clf.predict_proba(texts)

        return predictions, targets, texts

    def validate(self):
        # Load all validation snippets
        texts, targets, snippets = self.preparedataset("validation")

        # Get predictions from model
        predicted = self.text_clf.predict_proba(texts)

        # Pretty print results sorted by sponsor probability
        results = zip(predicted, targets, texts, snippets)
        t = PrettyTable(['video', 'timestamp', 'target', '%sponsor', '%non_sponsor', 'text'])
        t.align = "l"
        results = sorted(results, key=lambda x: x[0][0])
        for result in results:
            t.add_row([result[3]['video_id'], f"{result[3]['start']}-{result[3]['end']}",  self.target_names[result[1]], f"{result[0][0]:.5f}", f"{result[0][1]:.5f}", result[2].replace("\n", " ")[:200]])

        print(t)

    def validate_auto(self):
        # Load all validation snippets
        texts, targets, snippets = self.preparedataset("validation")

        # Get predictions from model
        predicted_probs = self.text_clf.predict_proba(texts)

        # Set threshold and evaluate using sklearn metrics
        predicted = [0 if x[0] > 0.5 else 1 for x in predicted_probs]
        print(f"Threshold: {0.5}")
        print(metrics.classification_report(targets, predicted, target_names=self.target_names, zero_division=0))

        predicted = [0 if x[0] > 0.25 else 1 for x in predicted_probs]
        print(f"Threshold: {0.25}")
        print(metrics.classification_report(targets, predicted, target_names=self.target_names, zero_division=0))

        predicted = [0 if x[0] > 0.1 else 1 for x in predicted_probs]
        print(f"Threshold: {0.1}")
        print(metrics.classification_report(targets, predicted, target_names=self.target_names, zero_division=0))

        predicted = [0 if x[0] > 0.01 else 1 for x in predicted_probs]
        print(f"Threshold: {0.01}")
        print(metrics.classification_report(targets, predicted, target_names=self.target_names, zero_division=0))

        predicted = [0 if x[0] > 0.005 else 1 for x in predicted_probs]
        print(f"Threshold: {0.005}")
        print(metrics.classification_report(targets, predicted, target_names=self.target_names, zero_division=0))


if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(description='A video sponsor segment classifier.')
    parser.add_argument('--train', help="train the classifier", action='store_true')
    parser.add_argument('--validate_boxes', help="Validate the classifier on validation dataset and print out segments with the highest sponsor probability (you'll need to train or load a trained model first)", action='store_true')
    parser.add_argument('--validate_auto', help="Print metrics for various threshholds (you'll need to train or load a trained model first)", action='store_true')
    parser.add_argument('--apply', help="apply the classifier video by video (you'll need to train or load a trained model first)", action='store_true')

    args = parser.parse_args()
    filename = "naive_bayes.pickle"

    if args.train:
        # Initialize the classifier
        classifier = NaiveBayesSKLearn()

        # Train the classifier
        classifier.train()

        with open(filename, 'wb') as f:
            # Pickle the classifier using the highest protocol available.
            pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)

    if args.validate_boxes:
        with open(filename, 'rb') as f:
            # Load the classifier using pickle
            classifier: NaiveBayesSKLearn = pickle.load(f)

        classifier.validate()

    if args.validate_auto:
        with open(filename, 'rb') as f:
            # Load the classifier using pickle
            classifier: NaiveBayesSKLearn = pickle.load(f)

        classifier.validate_auto()

    if args.apply:
        with open(filename, 'rb') as f:
            # Load the classifier using pickle
            classifier: NaiveBayesSKLearn = pickle.load(f)

        # Load all validation videos
        with open(f"{classifier.datadir}validation.json") as jsonFile:
            videos = json.load(jsonFile)

        while 1:
            # Wait for input
            input("Hit any key to process random video from validation\n")
            print("====================================================")
            # Select a random video
            video = random.choice(videos)

            # If there are no sponsor segments: skip to next video
            while len(video["sponsorTimes"]) == 0:
                video = random.choice(videos)

            # Generate predictions from classifier
            predictions, targets, texts = classifier.predict(video)
            predicted_class = [0 if x[0] > 0.25 else 1 for x in predictions]

            # Pretty print using table
            t = PrettyTable(['index', 'target', 'prediction', '%sponsor', '%non_sponsor', 'text'])

            char = "\n"
            t.align = "l"
            for i in range(0, len(texts)):
                t.add_row([f'{i:04d}', classifier.target_names[targets[i]], classifier.target_names[predicted_class[i]], f'{predictions[i][0]:.5f}', f'{predictions[i][1]:.5f}', texts[i].replace(char, " ")])

            print(t)