import os
import pickle
from time import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam

from capsule_net.loss import capsule_loss_fn
from .tester import tester


def save_model(model, file_dir="model/"):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    torch.save(model.state_dict(), f"{file_dir}model.pkl")


def save_log_file(log_file, id, file_dir="log/"):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(f"{file_dir}log{id}.pickle", "wb") as f:
        pickle.dump(log_file, f)


def trainer(
    model, train_loader, test_loader, epochs: int, id: int, device=None
) -> None:
    """Training a CapsuleNet

    Args:
        model: the CapsuleNet model
        train_loader: torch.utils.data.DataLoader for training data
        test_loader: torch.utils.data.DataLoader for test data
        epoch:

    Returns:
        The trained model
    """
    print("Begin Training" + "-" * 70)

    start_time = time()
    optimizer = Adam(model.parameters(), lr=3e-4)
    best_val_acc = 0
    train_loss = 0.0
    train_accuracy = 0.0

    # log writer
    log_file = {
        "train_loss": np.zeros(epochs),
        "test_loss": np.zeros(epochs),
        "train_accuracy": np.zeros(epochs),
        "test_accuracy": np.zeros(epochs),
    }

    for epoch in range(epochs):
        model.train()  # set to training mode
        ti = time()
        for x, y in train_loader:
            x, y = (
                Variable(x.to(device)),
                Variable(y.to(device)),
            )
            y_pred = model(x)
            loss = capsule_loss_fn(y_pred, y)
            # record
            train_loss += loss.item() * x.shape[0]
            literal_y_pred = y_pred.data.max(1)[1]
            literal_y_true = y.data.max(1)[1]
            train_accuracy += literal_y_pred.eq(literal_y_true).cpu().sum()
            # update net
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = torch.true_divide(train_loss, len(train_loader.dataset))
        train_accuracy = torch.true_divide(train_accuracy, len(train_loader.dataset))
        log_file["train_loss"][epoch] = train_loss
        log_file["train_accuracy"][epoch] = train_accuracy
        # compute validation loss and acc
        val_loss, val_acc = tester(model, test_loader, device)
        # record
        log_file["test_loss"][epoch] = val_loss
        log_file["test_accuracy"][epoch] = val_acc
        print(
            "==> Epoch %02d: train_loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
            % (
                epoch,
                train_loss,
                val_loss,
                val_acc,
                time() - ti,
            )
        )
        # update best validation acc and save model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print("best val_acc increased to %.4f" % best_val_acc)
    print("Total time = %ds" % (time() - start_time))
    save_log_file(log_file, id)
