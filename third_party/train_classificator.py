import tqdm
import torch
from torch.utils import data
import torchvision
from torch.nn import functional
import os

from dataloader import BricksDataloader
from callbacks import VisImagesGrid, VisPlot
from optimizers import RangerAdam as RAdam


if __name__ == '__main__':
    dataset_path = '/media/alexey/SSDDataDisk/datasets/lego/bricks_classification/'
    test_dataset_path = '/media/alexey/SSDDataDisk/datasets/lego/bricks_classification/'
    experiments_folder = '/media/alexey/SSDDataDisk/experiments/bricks_classification/'

    load_path = None
    visdom_port = 9000  # Set None to disable
    batch_size = 16
    epochs = 200
    device = 'cuda'
    save_best = False
    best_test_score = 0

    os.makedirs(experiments_folder, exist_ok=True)

    train_dataset = BricksDataloader(
        dataset_path,
        (224, 224),
        True,
        os.path.join(experiments_folder, 'train_data.csv'),
        os.path.join(experiments_folder, 'classes.txt')
    )

    print('Train dataset size: {}'.format(len(train_dataset)))
    print('Classes count: {}'.format(train_dataset.num_classes))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    test_dataset = BricksDataloader(
        dataset_path,
        (224, 224),
        False,
        os.path.join(experiments_folder, 'val_data.csv'),
        os.path.join(experiments_folder, 'classes.txt')
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    visdom_visualizer = None if visdom_port is None else VisImagesGrid(
        title='Bricks',
        port=visdom_port,
        vis_step=100,
        scale=2,
        grid_size=4
    )

    plot_visualizer = None if visdom_port is None else VisPlot(
        title='Training curves',
        port=visdom_port
    )

    if plot_visualizer is not None:
        plot_visualizer.register_scatterplot(
            name='per_epoch loss',
            xlabel='Epoch',
            ylabel='CrossEntropy'
        )

        plot_visualizer.register_scatterplot(
            name='per_epoch acc',
            xlabel='Epoch',
            ylabel='Accuracy'
        )

    # model = torchvision.models.resnet18(False)
    # model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model = torchvision.models.resnet50(False)
    model.fc = torch.nn.Linear(model.fc.in_features, train_dataset.num_classes)

    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
        print(
            'Model has been loaded by path: {}'.format(load_path)
        )
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = RAdam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, factor=0.1
    )

    for epoch in range(1, epochs + 1):
        model.train()
        avg_epoch_loss = 0

        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            for i, (_x, _y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                x = _x.to(device)
                y_truth = _y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y_truth)

                loss.backward()
                optimizer.step()

                pbar.postfix = \
                    'Epoch: {}/{}, loss: {:.8f}'.format(
                        epoch,
                        epochs,
                        loss.item() / train_dataloader.batch_size
                    )
                avg_epoch_loss += \
                    loss.item() / train_dataloader.batch_size / len(
                        train_dataloader
                    )

                if visdom_visualizer is not None:
                    visdom_visualizer.per_batch(
                        {
                            'img': x
                        }
                    )

                pbar.update(1)

        model.eval()
        avg_acc_rate = 0
        test_len = 0
        if test_dataloader is not None:
            for _x, _y in tqdm.tqdm(test_dataloader):
                x = _x.to(device)
                y_truth = _y.to(device)
                y_pred = model(x)
                acc_rate = torch.eq(y_truth, y_pred.argmax(dim=1)).sum() / y_truth.size(0)
                acc_rate = float(acc_rate.to('cpu').numpy())

                avg_acc_rate += acc_rate
                test_len += 1

        if test_len > 0:
            avg_acc_rate /= test_len

        if plot_visualizer is not None:
            plot_visualizer.per_epoch(
                {
                    'n': epoch,
                    'loss': avg_epoch_loss,
                    'acc': avg_acc_rate
                }
            )

        scheduler.step(1.0 - avg_acc_rate)
        print('Test accuracy rate: {:.3f}'.format(avg_acc_rate))

        model_save_path = os.path.join(
            experiments_folder,
            'resnet_epoch_{}_loss_{:.2f}.trh'.format(epoch, avg_acc_rate)
        )

        if save_best:
            if best_test_score - avg_acc_rate < -1E-5:
                best_test_score = avg_acc_rate

                model = model.to('cpu')
                torch.save(
                    model.state_dict(),
                    model_save_path
                )
                model = model.to(device)
        else:
            model = model.to('cpu')
            torch.save(
                model.state_dict(),
                model_save_path
            )
            model = model.to(device)

    exit(0)
