# SponsorBlock Net
This is the project repository of the SoSe 21 Machine Learning Project which aims to automatically
recognize sponsored segments in YouTube videos.

# Attribution
Uses SponsorBlock data licensed used under CC BY-NC-SA 4.0. More details: https://sponsor.ajay.app/

# Running the Project
## Directory structure
## Setup
## Data preparation
### Transcript Fetcher
Transcriptfetcher downloads english transcripts and metadata for all video-urls provided in a ````videoids.txt````. The file should have 
the following format
```
https://www.youtube.com/watch?v=ATMi30bGEVc
https://www.youtube.com/watch?v=0d0R-pwnrYA
https://www.youtube.com/watch?v=Kh7z0pGwAqY
https://www.youtube.com/watch?v=kPhfMUqzAqM
https://www.youtube.com/watch?v=UTUMCJomz7U
https://www.youtube.com/watch?v=RT2SWdw7L1g
https://www.youtube.com/watch?v=jDJSUclgDG8
https://www.youtube.com/watch?v=LRfP8r9u3W0
https://www.youtube.com/watch?v=xLDBnytcdDw
https://www.youtube.com/watch?v=Yn55w-8bfTY
```
and needs to be placed inside the ````transcript_fetcher```` directory. Inside the directory simply call ```python main.py``` and the script will download transcripts and metadata for all provided videos at a rate of ~1 video per second. 
Transcript and metadata are will be placed in the ```data``` subdirectory. To not overload the os filesystem, all data is arranged in a tree like structure.
Furthermore a logfile will be written in which all videos + paths are persisted. To simplify dataloading a metadata file containing all videos is helpful. By running 
````python generate_meta.py```` such a file can be generated from the logfile generated in the step before.

An single entry looks like this contains general information about a video as well as sponsortimes from sponsorblock:
```json
{
    "id": "MMD0pBs4DlU",
    "directory": "out/000/024/029/MMD0pBs4DlU",
    "channel": "Stephen Hawes",
    "channel_id": "UCMf49SMPnhxdLormhEpfyfg",
    "duration": 629,
    "title": "Running Multi-Feeder Jobs!",
    "subtitles_en": false,
    "captions_en": true,
    "sponsorTimes": [
      {
        "startTime": 535,
        "endTime": 567
      },
      {
        "startTime": 537,
        "endTime": 567
      },
      {
        "startTime": 534,
        "endTime": 567
      }
    ]
  }
```

### Data split
The data was split into training, validation and test sets (distribution: 70%, 20%, 10%). The splits respects channels so that channels only occur in one dataset to avoid overfitting. 
To split the data yourself you first need to generate a list of all channels and their amount of videos. This can be done by first executing the following command inside of the ```data``` directory:

```cat meta.json | jq ". [] | .channel_id" | sort | uniq -c | sort -nr > channels.txt```

and then running ```python data_split.py``` inside the ```data_split``` directory. To ceck the distribution of the split execute:

```cat <dataset_name>.json | jq  ". [] | .channel_id" | sort | uniq -c | sort -nr | wc -l```

## Training
### Bag of words
There are two bag of words based classifiers available (naive_bayes and logistic_regression). Both classifiers are trained on the training dataset by simply calling 
```python naive_bayes_sk.py --train``` or ```python logistic_regression_sk.py --train```. Training generates a dump of the model which later can be used for evaluation. 

### BERT
The BERT based model gets trained by pytorch lightning so running it is simple as well. To make sure you are running on the correct gpu (and only on one gpu) set the ```export CUDA_VISIBLE_DEVICES=<DEVICE_ID>``` environment variable.
Then you can start training by running ```python classifier.py``` inside the ```bert``` directory. This trains your model and it in your current directory. When training you can choose the length of snippets given to BERT and if you want to finetune the Transformer layer.

## Evaluation and Demo

### Bag of Words
Both bag of words model provide an easy to use command line interface for evaluation. Available options (replace model.py with ```naive_bayes_sk.py``` or ```logistic_regression_sk.py``` :

- ```python model.py --apply```: Apply the model to a single, randomly chosen video and print each snippet with ground truth + prediction. Pressing any key restarts the process with a new video
- ```python model.py --validate_boxes```: Applies the model to all videos from the validation dataset and reports snippets sorted by sponsor probability
- ```python model.py --validate_auto```: Applies the model to all videos from the validation dataset and reports precision, recall and f1 for various thresholds

### BERT
There are two different ways to evaluate your BERT based model. To view current metrics (precision, recall, f1) simply run the training and open tensorboard (```tensorboard --logdir ./lightning_logs```). In tensorboard the current metrics on the validation set are shown.
The other way of evaluating BERT is through the ```evaluate.py``` script inside the ```bert``` directory. It behaves similar to the ```--apply``` function from the bag of words based models. 