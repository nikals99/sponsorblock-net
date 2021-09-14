import pytorch_lightning as pl

from video.model import SponsorNet
from video.dataset import SponsorNetVideoDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

def ray_train_model(config):
    """
    This method is used by ray tune for the hyperparameter tuning
    """
    # data
    datamodule = SponsorNetVideoDataModule(
        "training_samples_bgr2rgb.pickle",
        "validation_samples_bgr2rgb.pickle",
        "test_samples.pickle",
        num_workers=32,
        batch_size=config["batch_size"]
    )

    # model
    model = SponsorNet(config)

    # training
    logger = TensorBoardLogger("tensorboard_runs")
    metrics = {"loss": "val_loss", "acc": "val_f1"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        logger=logger,
        max_epochs=10,
        gradient_clip_algorithm='value',
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model, datamodule)


def tune_model():
    search_space = {
        "lrate": tune.loguniform(1e-3, 1e-0),
        "weight_decay": tune.loguniform(1e-7, 1e-3),
        "finetune": tune.grid_search([True, False]),
        "batch_size": 600
    }

    trainable = tune.with_parameters(ray_train_model)
    analysis = tune.run(
        trainable,
        num_samples=10,
        scheduler=ASHAScheduler(metric="acc", mode="max"),
        config=search_space,
        name="sponsornet_tune",
        resources_per_trial={
            "gpu": 1
        },
    )

    print(analysis.get_best_config(metric="acc", mode="max"))


if __name__ == '__main__':
    tune_model()