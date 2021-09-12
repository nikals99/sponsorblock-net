from xml.dom import minidom

from transformers import BertTokenizerFast

"""
Returns an array of snippets:
snippet["start"]
snippet["end"]
snippet["text"]
snippet["sponsor"]
"""

# Define our base model, this should match with the model user for training your classifier
BERT_MODEL_NAME = "bert-base-uncased"


class TranscriptFetcher:
    def __init__(self, datadir="../data/", snippet_len=400):
        self.datadir = datadir

        # Initialize the tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
        self.snippet_len = snippet_len

    def wordIsSponsor(self, time, sponsorTimes):
        # iterate over each sponsor segment
        for sponsorTime in sponsorTimes:
            # if time is inbetween start and end the current word is inside of a sponsor segment
            if sponsorTime["startTime"] <= time <= sponsorTime["endTime"]:
                return True

        # if no segment matches we return False
        return False

    def getText(self, nodelist):
        # extract all text from a given xml node
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        return ''.join(rc)

    def convert(self, times, word_targets, words):

        # use the pretrained tokenizer to encode our words into tokens
        inputs = self.tokenizer.encode_plus(
            words,
            None,
            add_special_tokens=False,  # Add [CLS] [SEP]
            max_length=self.snippet_len + 50, # +50 for additional tokens generated during BPE
            padding='max_length',
            return_token_type_ids=False,
            is_split_into_words=True,
            return_attention_mask=True,  # Differentiates padded vs normal token
            return_offsets_mapping=True,
            truncation=True,  # Truncate data beyond max length
            return_tensors='pt'  # PyTorch Tensor format
        )

        labels = []
        counter = 0
        # set labels for each token
        for t in inputs.word_ids(0):
            if t is None:
                # if t is none it was padded so we set a negative label
                labels.append(-100)
                continue

            if t == counter:
                labels.append(word_targets[counter])
                counter += 1
                continue
            else:
                # word was generated during BPE so we set a negative label
                labels.append(-100)

        # Bring arrays into the correct format and return as a map so we dont have to deal with arrays of different sizes
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        seq = {
            "times": times,
            "words": words,
            "word_target": word_targets,
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'token_labels': labels
        }
        return seq

    def extracttranscript(self, video, offset=0):
        # Load the transcript file and parse it into a minidom doc
        doc = minidom.parse(f'{self.datadir}{video["directory"]}.en.srv3')

        # get items of interest
        items = doc.getElementsByTagName('p')

        sequences = []
        times = []
        word_targets = []
        words = []

        # sanity check
        assert offset < self.snippet_len

        for item in items:

            # some header foo we don't actually need
            if item.hasAttribute("a"):
                continue

            # We cannot deal with a missing duration or missing start time
            if not (item.hasAttribute("d") and item.hasAttribute("t")):
                continue

            # get the star time of the current segment
            start = int(int(item.getAttribute("t")) / 1000)

            # Chcek which format we have to parse (auto generated (item.hasAttribute("w")) vs user generated (else))
            if item.hasAttribute("w"):
                # get all word objects
                source_words = item.getElementsByTagName('s')

                # iterate over each word object
                for word in source_words:
                    # if word has a start time we use it. else it's the first word and we can use the snippets start time
                    if word.hasAttribute("t"):
                        time = start + int(int(word.getAttribute("t")) / 1000)
                    else:
                        time = start

                    # Append the current timestamp
                    times.append(time)
                    # Append the class of the current word
                    word_targets.append(1 if self.wordIsSponsor(time, video["sponsorTimes"]) else 0)
                    # Append the current word
                    words.append(self.getText(word.childNodes).replace(" ", ""))

            else:
                # get the complete text of the current segment
                text = self.getText(item.childNodes)

                # Split it into words
                source_words = text.split(" ")

                # Sanity check
                if len(source_words) <= 0:
                    continue

                # Get the duration for the whole snippet
                duration = int(item.getAttribute("d")) / 1000

                # set the current time to star time
                time = start
                # Interpolate for duration so each word gets a new timestamp between start and end
                time_increment = duration / len(source_words)

                # iterate over each word
                for word in source_words:
                    # Append the interpolated timestamp
                    times.append(int(time))
                    # Increase the timer
                    time += time_increment
                    # Append the class of the current word
                    word_targets.append(1 if self.wordIsSponsor(time, video["sponsorTimes"]) else 0)
                    # Append the current word
                    words.append(word)

            # Check when we need to split into a new sequence or if we skip because of offset
            if len(times) >= self.snippet_len or (offset != 0 and len(sequences) == 0 and len(times) >= offset):
                seq = self.convert(times, word_targets, words)
                sequences.append(seq)
                times = []
                word_targets = []
                words = []

        # Tokenize and append
        sequences.append(self.convert(times, word_targets, words))

        return sequences
