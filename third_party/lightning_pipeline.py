import torch
from torch.nn import functional as F
from torch.utils import data
import torchvision
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from .dataloader import BricksDataloader


class LitBrickDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset_instance: BricksDataloader,
                 val_dataset_instance: BricksDataloader,
                 batch_size: int = 32):
        super().__init__()
        self.train_dataset_instance = train_dataset_instance
        self.val_dataset_instance = val_dataset_instance
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset_instance,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset_instance,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )


class LitBrickClassifier(pl.LightningModule):
    def __init__(self, classes_count: int):
        super().__init__()
        self.model = torchvision.models.resnet50(False)
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            classes_count
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        acc = torch.eq(y, out.argmax(dim=1)).sum() / y.size(0)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def trainLitBrickClassifier(self, base_folder, epochs, gpus_count):

        train_dataset_path = os.path.join(base_folder, 'partno/')
        val_dataset_path = os.path.join(base_folder, 'partno_val/')
        test_dataset_path = ''
        experiments_folder = os.path.join(base_folder, 'experiments/')
        batch_size = 32

        os.makedirs(experiments_folder, exist_ok=True)

        train_dataset = BricksDataloader(
            train_dataset_path,
            (224, 224),
            True,
            os.path.join(experiments_folder, 'train_data.csv'),
            os.path.join(experiments_folder, 'classes.txt')
        )
        val_dataset = BricksDataloader(
            val_dataset_path,
            (224, 224),
            False,
            os.path.join(experiments_folder, 'val_data.csv'),
            os.path.join(experiments_folder, 'classes.txt')
        )
        train_dataset.updated_classes(val_dataset)

        print('Train dataset size: {}'.format(len(train_dataset)))
        print('Validation dataset size: {}'.format(len(val_dataset)))
        print('Classes count: {}'.format(train_dataset.num_classes))

        pl_model = LitBrickClassifier(
            train_dataset.num_classes
        )
        pl_dataloader = LitBrickDataModule(train_dataset, val_dataset)
        tensorboard_logger = TensorBoardLogger(
            save_dir=os.path.join(experiments_folder, 'logs/'),
            name='LegoBrickClassification'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(experiments_folder, 'checkpoints/'),
            filename="resnet50-{epoch:03d}-{val_loss:.2f}-{version:03d}.pt",
            save_top_k=3,
            mode="min",
        )

        trainer = pl.Trainer(
            default_root_dir=experiments_folder,
            logger=tensorboard_logger,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
            max_epochs=epochs,
            gpus=gpus_count
        )
        trainer.fit(pl_model, pl_dataloader)

        # ch = '/home/robert/LegoImageCropper/output/training_data/trainingexport_/20211004_200630/experiments/checkpoints/resnet50-epoch=006-val_loss=5.84.pt.ckpt'
        pl_model.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            #  ch,
            classes_count=train_dataset.num_classes
        )
        pl_model.eval()
        torch.jit.save(
            pl_model.to_torchscript(),
            os.path.join(experiments_folder, 'traced_best_model.pt')
        )
