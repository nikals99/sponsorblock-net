from prettytable import PrettyTable
from transformers import BertTokenizerFast

from classifier import BertClassifier
from dataset import YoutubeDataSet

BERT_MODEL_NAME = "bert-base-uncased"
CHECK_POINT_NAME = "./checkpoint_005_finetuning_larger_token_body"

label_to_text = ["no_sponsor", "sponsor"]


def evaluate_random_video():
    # Load a random video from dataset
    dataset = YoutubeDataSet("../data/validation.json", limit=1)

    # if dataset is empty try another one (e.g. parsing failure)
    if len(dataset) == 0:
        evaluate_random_video()

    # setup pretty table headers
    t = PrettyTable(['word', "token", 'target', 'logits'])
    t.align = "l"

    # Load classifier from checkpoint
    model = BertClassifier(pretrained=CHECK_POINT_NAME)
    # Set model to eval mode
    model.eval()

    # initialize a tokenizer for reversing the tokenization
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    # iterate over each sequence of the dataset
    for data in dataset:
        # extract variables from map
        labels = data['label']
        words = data['words']
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']

        # convert the token_ids back to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # unsqueeze the tensors so that it matches the format expexted by our model
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        # get the prediction from our model
        out = model.bert(input_ids, attention_mask=attention_mask)

        wordcounter = 0
        # iterate over each prediction
        for i in range(0, len(labels)):
            # Set default values for filled tokens
            loga = "---"
            word = ""
            label = "---"

            # if the current token is a regular (not filled) token
            if labels[i] != -100:
                # Assign sponsor or no_sponsor class
                if out['logits'][0][i][0] > out['logits'][0][i][1]:
                    loga = "no_sponsor"
                else:
                    loga = "sponsor"
                # set the tables values
                word = words[wordcounter]
                label = label_to_text[labels[i].item()]
                wordcounter += 1

            # write to the table
            t.add_row([word, tokens[i], label, loga])

        # Add a filler to separate different sequences
        t.add_row(["######", "######", "######", "######"])

    # finally print the table
    print(t)


if __name__ == '__main__':
    while 1:
        # Wait for input
        input("Hit any key to process random video from validation\n")
        print("====================================================")
        # Evaluate a random video
        evaluate_random_video()
