from __future__ import unicode_literals
import os
import youtube_dl

# If you don't want to start at the first video set this value
start_at = 0
log_file = object


def download(video_url, index, level_0, level_1, level_2):

    # Base path e.g. /000/000/000
    base_path = f"out/{f'{level_0:03d}'}/{f'{level_1:03d}'}/{f'{level_2:03d}'}"

    # Set youtube dl options see: https://github.com/ytdl-org/youtube-dl/blob/3e4cedf9e8cd3157df2457df7274d0c842421945/youtube_dl/YoutubeDL.py#L137-L312
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'writeinfojson': True,
        'subtitlesformat': 'srv3',
        'outtmpl': f"{base_path}/%(id)s.%(ext)s",
        'ignoreerrors': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        # Extract video url + video_id
        video_url = video_url.strip("\n")
        video_id = video_url.split('?v=')[-1]

        # Actually download it
        print(f"Downloading transcript for video {index}, url: {video_url}, id: {video_id}\n")
        ydl.download([video_url])

        # Check if english subtitles were downloaded else remove metadata
        if not os.path.isfile(f"{base_path}/{video_id}.en.srv3"):
            print("Could not download transcript....")
            try:
                os.remove(f"{base_path}/{video_id}.info.json")
                log_file.write(f"{index} {video_id} no_transcript\n")
            except Exception:
                print("Could not delete file")
                log_file.write(f"{index} {video_id} no_infojson\n")

        else:
            log_file.write(f"{index} {video_id} {base_path}/{video_id}\n")


if __name__ == '__main__':

    # open logfile for persistence
    log_file = open("logfile.txt", "a")

    # initialize variables for directory tree
    counter = start_at
    level_0 = 0
    level_1 = 0
    level_2 = 0
    level_3 = 0

    with open('videoids.txt') as f:

        # skip start_at videos
        for i in range(start_at):
            next(f)

        # download each video and place it inside of the directory structure
        for line in f:
            download(line, counter, level_0, level_1, level_2)
            counter += 1
            level_3 += 1
            if level_3 > 100:
                level_3 = 0
                level_2 += 1
                if level_2 > 100:
                    level_2 = 0
                    level_1 += 1
                    if level_1 > 100:
                        level_1 = 0
                        level_0 += 1
