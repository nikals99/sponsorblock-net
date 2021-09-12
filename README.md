# SponsorBlock Net
This is the project repository of the SoSe 21 Machine Learning Project which aims to automatically
recognize sponsored segments in YouTube videos.

# Attribution
Uses SponsorBlock data licensed used under CC BY-NC-SA 4.0. More details: https://sponsor.ajay.app/

# Running the Project
## Setup
## Fetching Data
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


## Training
## Evaluation and Demo