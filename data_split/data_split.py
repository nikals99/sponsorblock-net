import json

# Setup an array and a counter for each split
channels_training = []
count_training = 0
channels_validation = []
count_validation = 0
channels_test = []
count_test = 0

# Setup a total counter
count_total = 1

# load a list of all channels + the amount of videos they have (can be generated with: ```cat meta.json | jq ". [] | .channel_id" | sort | uniq -c | sort -nr > channels.txt```)
with open("../data/channels.txt") as channels:
    # For each channel
    for line in channels:
        # remove the trailing newline
        line = line.strip("\n")
        # Split into count and channel_id (format: \t<count>\t"<channel_id>")
        # 0 => count
        # 1 => channel_id
        # 2 => garbage
        split = line.split("\"")

        # Something is wrong with the split... (e.g. " inside of channel_id)
        if len(split) != 3:
            continue
        # convert the video count of the current channel to an integer
        count_local = int(split[0])

        # Check if the training data set is under represented
        if count_training / count_total < 0.7:
            # increase the total videos count
            count_total += count_local
            # Increase the training videos count
            count_training += count_local
            # append channel_id to validation channels
            channels_training.append(split[1])
            continue
        # Check if the validation data set is under represented
        if count_validation / count_total < 0.2:
            # increase the total videos count
            count_total += count_local
            # Increase the test videos count
            count_validation += count_local
            # append channel_id to validation channels
            channels_validation.append(split[1])
            continue

        # Since the other data sets are fine add to test data
        # increase the total videos count
        count_total += count_local
        # Increase the training videos count
        count_test += count_local
        # append channel_id to validation channels
        channels_test.append(split[1])

    # Print out some stats
    print(f"total: {count_total}")
    print(f"train: {count_training / count_total:.2f} => {count_training}")
    print(f"valid: {count_validation / count_total:.2f} => {count_validation}")
    print(f"test: {count_test / count_total:.2f} => {count_test}")

# get metadata for every video
meta_json = json.load(open(f"../data/meta.json"))

# get all videos with a training channel_id
training = [video for video in meta_json if ("channel_id" in video and video["channel_id"] in channels_training) and (video["captions_en"] or video["subtitles_en"])]
print(f"Saving training videos: {len(training)}")
# save metadata for training videos
with open("../data/training.json", "w") as out:
    json.dump(training, out)

# get all videos with a validation channel_id
validation = [video for video in meta_json if ("channel_id" in video and video["channel_id"] in channels_validation) and (video["captions_en"] or video["subtitles_en"])]
print(f"Saving validation videos: {len(validation)}")
# save metadata for training videos
with open("../data/validation.json", "w") as out:
    json.dump(validation, out)

# get all videos with a test channel_id
test = [video for video in meta_json if ("channel_id" in video and video["channel_id"] in channels_test) and (video["captions_en"] or video["subtitles_en"])]
print(f"Saving test videos: {len(test)}")
# save metadata for training videos
with open("../data/test.json", "w") as out:
    json.dump(test, out)