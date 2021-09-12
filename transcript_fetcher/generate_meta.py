import json
import sqlite3
from urllib import parse

# Setup Database connection
db = sqlite3.connect("../data/database.db")

db.row_factory = sqlite3.Row

# Simple method for querying the database
def query_db(query, args=(), one=False):
    cur = db.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

# Extract sponsortimes from database
def extractsponsortimes(videoid):
    sponsorTimes = query_db(f"SELECT * FROM sponsorTimes where videoID='{videoid}' AND category='sponsor' AND votes >= 0")
    return [{"startTime": int(x["startTime"]), "endTime": int(x["endTime"])} for x in sponsorTimes]


if __name__ == '__main__':

    # read each logfile entry an write it to meta.json
    with open('logfile.txt') as f:
        videos = []
        for line in f:
            output = {}

            # Split the logfile line into values
            obj = line.split(" ")

            # extract id
            output["id"] = obj[1]

            # extract directory
            directory = obj[2].strip("\n")
            output["directory"] = directory

            # fetch sponsortimes
            output["sponsorTimes"] = extractsponsortimes(output["id"])

            # Fetch data from info_json downloaded by youtube-dl
            if directory != "no_transcript" and directory != "no_infojson":
                info_json = json.load(open(f"{directory}.info.json"))
                try:
                    output["channel"] = info_json["channel"]
                    output["channel_id"] = info_json["channel_id"]
                    output["duration"] = info_json["duration"]
                    output["title"] = info_json["title"]
                    output["subtitles_en"] = False
                    output["captions_en"] = False
                except Exception as e:
                    print(f"{directory} => {e}")

                if "en" in info_json["subtitles"]:
                    output["subtitles_en"] = True

                if "automatic_captions" in info_json and "en" in info_json["automatic_captions"] and dict(parse.parse_qsl(parse.urlsplit(info_json["automatic_captions"]["en"][0]["url"]).query))["lang"] == "en":
                    output["captions_en"] = True

                # Append the video to the list of videos
                videos.append(output)

        # Write meta.json to disk
        with open("meta.json", "w") as out:
            json.dump(videos, out)
