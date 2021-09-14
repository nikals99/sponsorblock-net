import math
import torch
import torchmetrics

from torch import nn, sigmoid
from torch.nn import functional as F
from torchvision import transforms

import pytorch_lightning as pl
import torchvision.models as models

def get_loggable_images(inputs, preds):
	combined = list(zip(inputs, preds))
	filtered = list(filter(lambda x: not math.isnan(x[1][0]) and (int(x[1][0].round()) == 1), combined))
	if len(filtered) > 0:
		# revert the normalization and log these images
		filtered_inputs, _ = list(zip(*filtered))
		filtered_inputs = torch.stack(filtered_inputs)
		inv_normalize = transforms.Normalize(
			mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
			std=[1/0.229, 1/0.224, 1/0.225]
		)
		filtered_inputs = inv_normalize(filtered_inputs)
		return filtered_inputs
	return None


class SponsorNet(pl.LightningModule):
	def __init__(self, config):
		super().__init__()
		self.save_hyperparameters()
		
		self.learning_rate = config["lrate"]
		self.weight_decay = config["weight_decay"]

		self.val_recall = torchmetrics.Recall(num_classes=1, average='none')
		self.val_precision = torchmetrics.Precision(num_classes=1, average='none')
		self.val_f1 = torchmetrics.F1(num_classes=1, average='none')

		# source: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#inception-v3
		self.model = models.inception_v3(pretrained=True)
		for param in self.model.parameters():
			param.requires_grad = config["finetune"]

		num_classes = 1
		self.model.AuxLogits.fc = nn.Linear(768, num_classes)
		self.model.fc = nn.Linear(2048, num_classes)

	def forward(self, x):
		x = self.model(x)
		return x

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(
			self.parameters(),
			lr=(self.learning_rate or self.lr),
			weight_decay=self.weight_decay
		)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		inputs, labels = train_batch

		# source: https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
		outputs, aux_outputs = self.model(inputs)
		loss1 = F.binary_cross_entropy_with_logits(torch.squeeze(outputs), labels)
		loss2 = F.binary_cross_entropy_with_logits(torch.squeeze(aux_outputs), labels)
		loss = loss1 + .4 * loss2

		self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)		
		return loss


	def validation_step(self, val_batch, batch_idx):
		inputs, labels = val_batch # B x 3 x 299 x 299
		outputs = self.forward(inputs) # B x 1

		loss = F.binary_cross_entropy_with_logits(torch.squeeze(outputs), labels) # scalar value
		self.log("val_loss", loss)

		preds = sigmoid(outputs)
		labels = labels.int()
		
		recall = self.val_recall(preds, labels)
		self.log("val_recall", recall)
		
		precision = self.val_precision(preds, labels)
		self.log("val_precision", precision)
		
		f1 = self.val_f1(preds, labels)
		self.log("val_f1", f1)

		images_to_log = get_loggable_images(inputs, preds)
		if images_to_log is not None:
			self.logger.experiment.add_images("val_images", images_to_log, self.global_step)
		
		return loss
