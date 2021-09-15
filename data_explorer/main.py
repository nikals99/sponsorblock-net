import sys
import json
import random
import torch

# append the path of the parent directory
sys.path.append("..")

from bert.slice_evaluator import SliceEvaluatorBert
from video.videoevaluator import VideoEvaluator
from flask import Flask, render_template, redirect

app = Flask(__name__)


# checks if a timestamp is inside a sponsor segment or not
def slice_is_sponsor(time, sponsorTimes):
    # iterate over each sponsor segment
    for sponsorTime in sponsorTimes:
        # check if timestamp is between start and end of the current segment
        if sponsorTime["startTime"] <= time <= sponsorTime["endTime"]:
            return 1

    return 0


# creates ground truth for a given video
def create_ground_truth(video):
    ground_truth = []

    # for each second of the video check if it is sposnor or not
    for i in range(0, video["duration"]):
        ground_truth.append(slice_is_sponsor(i, video["sponsorTimes"]))

    return ground_truth


def get_video_metadata(videoId, videos):
    for entry in videos:
        if entry["id"] == videoId:
            return entry
    return None


# "main" method
with app.app_context():
    # load metadata for all validation videos
    with open("../data/validation.json") as jsonFile:
        videos = json.load(jsonFile)

    # initialize BERT based model
    bert_evaluator = SliceEvaluatorBert(model="../bert/checkpoint_005_finetuning_larger_token_body", snippet_len=400)

    # initialize video based model
    video_evaluator = VideoEvaluator(
        "auto_model_checkpoints/checkpoint.ckpt",
        "validation.json",
        "data_video"
    )

    print("DONE loading")


@app.route('/')
def home():
    return redirect("/prediction/random")


# prints out a prediction for a random video
@app.route('/prediction/random')
def predict_random():
    # pick a videoId from the smaller video dataset, this way we know that
    # it's present in the bigger bert dataset
    videoid = random.choice(list(video_evaluator.fetcher.metadata.values()))["id"]

    # get metadata for the current videoid
    video = get_video_metadata(videoid, videos)

    # create ground truth from sponsor segments
    ground_truth = create_ground_truth(video)

    # get predictions from both models
    predictions_bert = torch.tensor(bert_evaluator.evaluate(video))
    predictions_video = torch.tensor(video_evaluator.evaluate(videoid)[:len(ground_truth)])

    # weighted fusion of both models
    predictions_fusion = (0.9 * predictions_bert) + (0.1 * predictions_video)

    # pass to render engine
    return render_template(
        'predictions.html',
        videoId=videoid,
        videoDuration=video["duration"],
        ground_truth=ground_truth,
        predictions_bert=predictions_bert,
        predictions_video=predictions_video,
        predictions_fusion=predictions_fusion
    )


# prints out a prediction for a specific video
@app.route('/prediction/<videoId>')
def view_prediction(videoid):
    # get metadata for the current videoid
    video = get_video_metadata(videoid, videos)

    # create ground truth from sponsor segments
    ground_truth = create_ground_truth(video)

    # get predictions from both models
    predictions_bert = torch.tensor(bert_evaluator.evaluate(video))
    predictions_video = torch.tensor(video_evaluator.evaluate(videoid)[:len(ground_truth)])

    # weighted fusion of both models
    predictions_fusion = (0.9 * predictions_bert) + (0.1 * predictions_video)

    # pass to render engine
    return render_template(
        'predictions.html',
        videoId=videoid,
        videoDuration=video["duration"],
        ground_truth=ground_truth,
        predictions_bert=predictions_bert,
        predictions_video=predictions_video,
        predictions_fusion=predictions_fusion
    )
