from argparse import ArgumentParser, Namespace
from typing import Tuple
import tqdm
import torch
from torch.utils import data
import torchvision
import os

from third_party.dataloader import BricksDataloader
from third_party.callbacks import VisImagesGrid, VisPlot
from third_party.optimizers import RangerAdam as RAdam


class CustomTrainingPipeline(object):
    def __init__(self,
                 train_data_path: str,
                 val_data_path: str,
                 experiment_folder: str,
                 model: torch.nn.Module = torchvision.models.resnet50(False),
                 load_path: str = None,
                 visdom_port: int = 9000,
                 batch_size: int = 32,
                 epochs: int = 200,
                 stop_criteria: float = 1E-7,
                 device: str = 'cuda'):
        """
        Train model
        Args:
            train_data_path: Path to training data
            val_data_path: Path to testing data
            experiment_folder: Path to folder with checkpoints and experiments data
            load_path: Path to model weights to load
            visdom_port: Port of visualization
            batch_size: Training batch size
            epochs: Count of epoch
            stop_criteria: criteria to stop of training process
            device: Target device to train
        """
        self.model = model
        self.device = device
        self.train_dataset_path = train_data_path
        self.val_dataset_path = val_data_path
        self.experiment_folder = args.experiment_folder
        self.checkpoints_dir = os.path.join(experiment_folder, 'checkpoints/')

        self.load_path = args.pretrain_weights
        self.visdom_port = args.visdom_port  # Set None to disable
        self.batch_size = args.batch_size
        self.epochs = epochs
        self.stop_criteria = stop_criteria
        self.best_test_score = 0

        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.train_dataset = BricksDataloader(
            train_data_path,
            (224, 224),
            True,
            os.path.join(experiment_folder, 'train_data.csv'),
            os.path.join(experiment_folder, 'classes.txt')
        )
        self.val_dataset = BricksDataloader(
            val_data_path,
            (224, 224),
            False,
            os.path.join(experiment_folder, 'val_data.csv'),
            os.path.join(experiment_folder, 'classes.txt')
        )
        self.train_dataset.updated_classes(self.train_dataset)

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        self.batch_visualizer = None if visdom_port is None else VisImagesGrid(
            title='Bricks',
            port=visdom_port,
            vis_step=250,
            scale=1,
            grid_size=8
        )

        self.plot_visualizer = None if visdom_port is None else VisPlot(
            title='Training curves',
            port=visdom_port
        )

        if self.plot_visualizer is not None:
            self.plot_visualizer.register_scatterplot(
                name='train validation loss per_epoch',
                xlabel='Epoch',
                ylabel='CrossEntropy'
            )

            self.plot_visualizer.register_scatterplot(
                name='validation acc per_epoch',
                xlabel='Epoch',
                ylabel='Accuracy'
            )

        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features, self.train_dataset.num_classes)

        if load_path is not None:
            self.model.load_state_dict(
                torch.load(load_path, map_location='cpu'))
            print(
                'Model has been loaded by path: {}'.format(load_path)
            )
        self.model = self.model.to(device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = RAdam(params=model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, verbose=True, factor=0.1
        )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_step(self, epoch) -> float:
        self.model.train()
        avg_epoch_loss = 0

        with tqdm.tqdm(total=len(self.train_dataloader)) as pbar:
            for i, (_x, _y) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x = _x.to(self.device)
                y_truth = _y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_truth)

                loss.backward()
                self.optimizer.step()

                pbar.postfix = \
                    'Epoch: {}/{}, loss: {:.8f}'.format(
                        epoch,
                        self.epochs,
                        loss.item() / self.train_dataloader.batch_size
                    )
                avg_epoch_loss += \
                    loss.item() / len(self.train_dataloader)

                if self.batch_visualizer is not None:
                    self.batch_visualizer.per_batch(
                        {
                            'img': x
                        }
                    )

                pbar.update(1)

        return avg_epoch_loss

    def _validation_step(self) -> Tuple[float, float]:
        self.model.eval()
        avg_acc_rate = 0
        avg_loss_rate = 0
        test_len = 0

        if self.val_dataloader is not None:
            for _x, _y in tqdm.tqdm(self.val_dataloader):
                x = _x.to(self.device)
                y_truth = _y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_truth)
                avg_loss_rate += loss.item()
                acc_rate = torch.eq(y_truth, y_pred.argmax(
                    dim=1)).sum() / y_truth.size(0)
                acc_rate = float(acc_rate.to('cpu').numpy())

                avg_acc_rate += acc_rate
                test_len += 1

        if test_len > 0:
            avg_acc_rate /= test_len
            avg_loss_rate /= test_len

        self.scheduler.step(1.0 - avg_acc_rate)

        return avg_loss_rate, avg_acc_rate

    def _plot_values(self, epoch, avg_train_loss, avg_val_loss, avg_val_acc):
        if self.plot_visualizer is not None:
            self.plot_visualizer.per_epoch(
                {
                    'n': epoch,
                    'val loss': avg_val_loss,
                    'loss': avg_train_loss,
                    'val acc': avg_val_acc
                }
            )

    def _save_best_traced_model(self, save_path: str):
        traced_model = torch.jit.trace(self.model, torch.rand(1, 3, 224, 224))
        torch.jit.save(traced_model, save_path)

    def _save_best_checkpoint(self, epoch, avg_acc_rate):
        model_save_path = os.path.join(
            self.checkpoints_dir,
            'resnet_epoch_{}_loss_{:.2f}.trh'.format(epoch, avg_acc_rate)
        )
        best_model_path = os.path.join(
            self.checkpoints_dir,
            'best.trh'
        )
        best_traced_model_path = os.path.join(
            self.experiment_folder,
            'traced_best_model.pt'
        )

        if self.best_test_score - avg_acc_rate < -1E-5:
            self.best_test_score = avg_acc_rate

            self.model = self.model.to('cpu')
            self.model.eval()
            torch.save(
                self.model.state_dict(),
                model_save_path
            )
            torch.save(
                self.model.state_dict(),
                best_model_path
            )
            self._save_best_traced_model(best_traced_model_path)
            self.model = self.model.to(self.device)

    def check_stop_criteria(self):
        return self.get_lr() - self.stop_criteria < -1E-9

    def fit(self):
        for epoch_num in range(1, self.epochs + 1):
            epoch_train_loss = self._train_step(epoch_num)
            val_loss, val_acc = self._validation_step()
            self._plot_values(epoch_num, epoch_train_loss, val_loss, val_acc)
            self._save_best_checkpoint(epoch_num, val_acc)

            if self.check_stop_criteria():
                break


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Training pipeline')
    parser.add_argument(
        '--train_data', type=str, required=True,
        help='Path to training data.'
    )
    parser.add_argument(
        '--test_data', type=str, required=True,
        help='Path to testing data.'
    )
    parser.add_argument(
        '--experiment_folder', type=str, required=True,
        help='Path to folder with checkpoints and experiments data.'
    )
    parser.add_argument(
        '--epochs', type=int, required=False, default=200
    ),
    parser.add_argument(
        '--pretrain_weights', type=str, required=False,
        help='Path to model weights to load.'
    )
    parser.add_argument(
        '--visdom_port', type=int, required=False, default=9000,
        help='Port of visualization.'
    )
    parser.add_argument(
        '--batch_size', type=int, required=False, default=32,
        help='Training batch size.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    CustomTrainingPipeline(
        train_data_path=args.train_data,
        val_data_path=args.test_data,
        experiment_folder=args.experiment_folder,
        load_path=args.pretrain_weights,
        visdom_port=args.visdom_port,
        epochs=args.epochs,
        batch_size=args.batch_size
    ).fit()

    exit(0)
