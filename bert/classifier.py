from collections import OrderedDict

import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, AdamW
import torch

from bert.dataset import YoutubeDataSet


# Set our base model, when using your own pretrained model set this to a file system path
BERT_MODEL_NAME = "bert-base-uncased"

# Sometimes the tokenizer has weird problems... this seems to fix it..
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class BertClassifier(LightningModule):
    def __init__(self, pretrained=BERT_MODEL_NAME, finetuning=True, **kwargs):
        super().__init__()
        # initialize validation metrics via torchmetrics
        self.valid_precision = torchmetrics.Precision(num_classes=2, average='none')
        self.valid_recall = torchmetrics.Recall(num_classes=2, average='none')
        self.valid_f1 = torchmetrics.F1(num_classes=2, average='none')

        # initialize BertForTokenClassification from pretrained huggingface bert
        self.bert = BertForTokenClassification.from_pretrained(pretrained, num_labels=2)

        if not finetuning:
            for param in self.bert.bert.parameters():
                param.requires_grad = False

    def forward(self, batch):
        # The actual forward pass for our model
        # Parse elements from batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        # Simply path to pretrained bert model
        return self.bert(input_ids, attention_mask=attention_mask, labels=labels)

    def train_dataloader(self):
        # Load training data via DataLoader + Dataset
        data = YoutubeDataSet("../data/training.json", limit=10000)
        return DataLoader(data, batch_size=10, num_workers=1)

    def val_dataloader(self):
        # Load validation data via DataLoader + Dataset
        data = YoutubeDataSet("../data/validation_enriched.json", limit=4000)
        return DataLoader(data, batch_size=10, num_workers=1)

    def training_step(self, batch, batch_idx):
        # perform one training step

        # get the loss from one batched forward pass
        loss = self(batch)['loss']

        # log the current training loss
        self.log('training_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # prepare return value
        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        # perform one validation step

        # do one batched forward pass
        result = self(batch)

        # extract loss and log it
        loss = result['loss']
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # for each prediction do some metric calculations
        for i, x in enumerate(result["logits"]):
            # fetch all logits and labels by ignoring negative values (those that are filled when doing BPE)
            m = batch["label"][i] >= 0
            logits = x[m]
            labels = batch["label"][i][m]

            # apply softmax to our logits
            pred = torch.softmax(logits, 1)

            # pass predictions to metric calculators
            self.valid_recall(pred, labels)
            self.valid_precision(pred, labels)
            self.valid_f1(pred, labels)

        # prepare return value
        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        })
        return output

    def on_validation_epoch_end(self):
        # After each validation epoch compute the current metrics and log them

        precision = self.valid_precision.compute()
        self.log('valid_precision_no_sponsor', precision[0])
        self.log('valid_precision_sponsor', precision[1])

        recall = self.valid_recall.compute()
        self.log('valid_recall_no_sponsor', recall[0])
        self.log('valid_recall_sponsor', recall[1])

        f1 = self.valid_f1.compute()
        self.log('valid_f1_no_sponsor', f1[0])
        self.log('valid_f1_sponsor', f1[1])

    def configure_optimizers(self):
        # Configure optimizers for BERT (taken from: huggingface)
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=2e-5,
        )
        return optimizer


if __name__ == '__main__':
    # Name of the output model
    NAME = "checkpoint_005_finetuning_larger_token_body"

    # Configure logging for use with tensorboard
    logger = TensorBoardLogger("lightning_logs", default_hp_metric=False, log_graph=True, name=NAME)

    # initialize our classifier
    classifier = BertClassifier()

    # initialize the trainer
    trainer = Trainer(max_epochs=10, logger=logger, gpus=1)

    # start training
    trainer.fit(classifier)

    # Save our model
    trainer.save_checkpoint(f"{NAME}.ckpt")

    # save the finetuned bert model since it isn't saved via save checkpoint
    classifier.bert.save_pretrained(NAME)