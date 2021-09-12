from xml.dom import minidom

"""
Returns an array of snippets:
snippet["start"]
snippet["end"]
snippet["text"]
snippet["sponsor"]
"""


class TranscriptFetcher:
    def __init__(self):
        self.datadir = "../data/"

    def isSponsor(self, start, end, sponsorTimes):
        start = int(start)
        end = int(end)
        for time in sponsorTimes:
            if int(time["startTime"]) >= start and int(time["endTime"]) <= end:
                return True

            if int(time["startTime"]) <= end <= int(time["endTime"]):
                return True

            if int(time["startTime"]) <= start and int(time["endTime"]) >= end:
                return True

        return False

    def getText(self, nodelist):
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        return ''.join(rc)

    def extracttranscript(self, video):
        doc = minidom.parse(f'{self.datadir}{video["directory"]}.en.srv3')

        items = doc.getElementsByTagName('p')
        snippets = []
        targets = []
        texts = []

        for item in items:
            snippet = {}
            text = ""

            if item.hasAttribute("a"):
                continue

            if not (item.hasAttribute("d") and item.hasAttribute("t")):
                continue

            snippet["start"] = int(int(item.getAttribute("t")) / 1000)

            snippet["end"] = snippet["start"] + int(int(item.getAttribute("d")) / 1000)
            # Chcek which format we have to parse
            if item.hasAttribute("w"):
                words = item.getElementsByTagName('s')

                if len(words) > 1:
                    if not words[-1].hasAttribute("t"):
                        continue
                    snippet["end"] = snippet["start"] + int(int(words[-1].getAttribute("t")) / 1000)
                else:
                    snippet["end"] = snippet["start"] + 1  # estimate 1 second for the one word spoken

                for word in words:
                    text += f" {self.getText(word.childNodes)}"

            else:
                text = self.getText(item.childNodes)

            snippet["text"] = text.lower()
            snippet["sponsor"] = self.isSponsor(snippet["start"], snippet["end"], video["sponsorTimes"])
            snippet["video_id"] = video["id"]
            snippets.append(snippet)
            targets.append(0 if snippet["sponsor"] else 1)
            texts.append(snippet["text"])

        return snippets, targets, texts
