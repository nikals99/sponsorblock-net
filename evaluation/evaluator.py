import json
import torch
import random
import pickle
import torchmetrics
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from bert.slice_evaluator import SliceEvaluatorBert
from video.videoevaluator import VideoEvaluator


def word_is_sponsor(time, sponsorTimes):
    for sponsorTime in sponsorTimes:
        if sponsorTime["startTime"] <= time <= sponsorTime["endTime"]:
            return 1
    return 0


def create_ground_truth(video):
    ground_truth = []
    for i in range(0, video["duration"]):
        ground_truth.append(word_is_sponsor(i, video["sponsorTimes"]))
    return ground_truth


def save_roc_fig(fprs, tprs, filename="roc.png"):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fprs, tprs, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(filename)


def save_roc_data(fprs, tprs, thresholds, filename="roc_data.pickle"):
    saveData = { "fprs": fprs, "tprs": tprs, "thresholds": thresholds }
    file = open(filename, "wb")
    pickle.dump(saveData, file)
    file.close()


def find_best_threshold(fprs, tprs, thresholds):
    entries = list(zip(fprs, tprs, thresholds, tprs-fprs))
    return max(entries, key=lambda x: x[3])


def evaluate(videos):
    print("Starting evaluation of small snippet len (150)")
    
    # initialize BERT model and metrics
    evaluator_bert = SliceEvaluatorBert(model="../bert/checkpoint_005_finetuning_larger_token_body", snippet_len=150)
    roc_bert = torchmetrics.ROC(num_classes=None, pos_label=1)
    precision_bert = torchmetrics.Precision(num_classes=1, average='none')
    recall_bert = torchmetrics.Recall(num_classes=1, average='none')
    f1_bert = torchmetrics.F1(num_classes=1, average='none')

    # initialize video model and metrics
    evaluator_video = VideoEvaluator("auto_model_checkpoints/checkpoint.ckpt", "meta.json", "data_video")
    roc_video = torchmetrics.ROC(num_classes=None, pos_label=1)
    precision_video = torchmetrics.Precision(num_classes=1, average='none')
    recall_video = torchmetrics.Recall(num_classes=1, average='none')
    f1_video = torchmetrics.F1(num_classes=1, average='none')

    # initialize fusion metrics
    roc_fusion = torchmetrics.ROC(num_classes=None, pos_label=1)
    precision_fusion = torchmetrics.Precision(num_classes=1, average='none')
    recall_fusion = torchmetrics.Recall(num_classes=1, average='none')
    f1_fusion = torchmetrics.F1(num_classes=1, average='none')

    # Create the list of ground truth tensors and remove unusable entries
    evaluation_entries = []
    for video in videos:
        # skip videos if there are no sponsored segments
        if len(video["sponsorTimes"]) == 0:
            continue

        # make sure this video is also present in the video dataset
        if evaluator_video.fetcher._get_video_metadata(videoId=video["id"]) is None:
            continue

        # make sure the ground truth does not consist of only negative samples
        ground_truth = torch.tensor(create_ground_truth(video))
        if (torch.equal(ground_truth, torch.zeros(ground_truth.shape, dtype=torch.long))):
            continue
        
        evaluation_entries.append((video, ground_truth))

    # Evaluate each of the remaining videos
    num_evaluated_videos = 0
    for (video, ground_truth) in tqdm(evaluation_entries[:1000]):

        # evaluate BERT classifier
        pred_bert = torch.tensor(evaluator_bert.evaluate(video))

        # evaluate the video classifier
        # remove the trailing bits of the video prediction, this is due to the unprecise duration in the sponsortimes db
        pred_video = torch.tensor(evaluator_video.evaluate(video["id"])[:ground_truth.shape[0]])
        if pred_video.shape[0] != ground_truth.shape[0]:
            continue
    
        # create the fusioned prediction
        pred_fusion = (0.9 * pred_bert) + (0.1 * pred_video)

        # update BERT metrics
        roc_bert(pred_bert, ground_truth)
        precision_bert(pred_bert, ground_truth)
        recall_bert(pred_bert, ground_truth)
        f1_bert(pred_bert, ground_truth)

        # update video metrics
        roc_video(pred_video, ground_truth)
        precision_video(pred_video, ground_truth)
        recall_video(pred_video, ground_truth)
        f1_video(pred_video, ground_truth)
        
        # update fusioned metrics
        roc_fusion(pred_fusion, ground_truth)
        precision_fusion(pred_fusion, ground_truth)
        recall_fusion(pred_fusion, ground_truth)
        f1_fusion(pred_fusion, ground_truth)

        num_evaluated_videos += 1


    print(f"Evaluated {num_evaluated_videos} videos")
    print()
    
    print("Bert")
    print(f"Precision: {precision_bert.compute()}")
    print(f"Recall: {recall_bert.compute()}")
    print(f"F1: {f1_bert.compute()}")
    fprs_bert, tprs_bert, thresholds_bert = roc_bert.compute()
    save_roc_data(fprs_bert, tprs_bert, thresholds_bert, filename="roc_data_bert.pickle")
    save_roc_fig(fprs_bert, tprs_bert, filename="roc_bert.png")
    best_result_bert = find_best_threshold(np.array(fprs_bert), np.array(tprs_bert), np.array(thresholds_bert))
    print(f"fpr: {best_result_bert[0]} tpr: {best_result_bert[1]} threshold: {best_result_bert[2]}")
    print()

    print("Video")
    print(f"Precision: {precision_video.compute()}")
    print(f"Recall: {recall_video.compute()}")
    print(f"F1: {f1_video.compute()}")
    fprs_video, tprs_video, thresholds_video = roc_video.compute()
    save_roc_data(fprs_video, tprs_video, thresholds_video, filename="roc_data_video.pickle")
    save_roc_fig(fprs_video, tprs_video, filename="roc_video.png")
    best_result_video = find_best_threshold(np.array(fprs_video), np.array(tprs_video), np.array(thresholds_video))
    print(f"fpr: {best_result_video[0]} tpr: {best_result_video[1]} threshold: {best_result_video[2]}")
    print()

    print("Fusion")
    print(f"Precision: {precision_fusion.compute()}")
    print(f"Recall: {recall_fusion.compute()}")
    print(f"F1: {f1_fusion.compute()}")
    fprs_fusion, tprs_fusion, thresholds_fusion = roc_fusion.compute()
    save_roc_data(fprs_fusion, tprs_fusion, thresholds_fusion, filename="roc_data_fusion.pickle")
    save_roc_fig(fprs_fusion, tprs_fusion, filename="roc_fusion.png")
    best_result_fusion = find_best_threshold(np.array(fprs_fusion), np.array(tprs_fusion), np.array(thresholds_fusion))
    print(f"fpr: {best_result_fusion[0]} tpr: {best_result_fusion[1]} threshold: {best_result_fusion[2]}")


if __name__ == '__main__':
    with open("../data/test_enriched.json") as jsonFile:
        videos = json.load(jsonFile)

    random.shuffle(videos)
    evaluate(videos)
