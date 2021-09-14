import pytorch_lightning as pl

from video.model import SponsorNet
from video.dataset import SponsorNetVideoDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == "__main__":
    # Name of the output model
    name = "sponsornet_video"

    # Define hyper parameter
    config = {
        "lrate": 0.04061086404168325,
        "weight_decay": 8.208955573020693e-06,
        "finetune": False,
        "batch_size": 600,
    }

    # Configure the DataModule to use preloaded samples
    datamodule = SponsorNetVideoDataModule(
        "training_samples_baseline.pickle",
        "validation_samples_baseline.pickle",
        "test_samples_baseline.pickle",
        num_workers=32,
        batch_size=config["batch_size"]
    )

    # initialize the model
    model = SponsorNet(config)

    # configure logging for use with tensorboard
    logger = TensorBoardLogger("lightning_logs", name=name)
    
    # configure callbacks
    callbacks = [
        # This will automatically save the top 3 models
        ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=3,
            filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
            dirpath="auto_model_checkpoints/",
        ),
        # This will stop the training early if val_f1 does not improve for 5 epochs
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            min_delta=0.00,
            patience=5,
            verbose=False,
        )
    ]

    # initialize the trainer
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        logger=logger,
        max_epochs=10,
        gradient_clip_algorithm='value',
        callbacks=callbacks,
    )

    # start training
    trainer.fit(model, datamodule)
