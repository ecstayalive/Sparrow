import torch
from torch.autograd import Variable

from capsule_net.loss import capsule_loss_fn


def tester(model, test_loader, device):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    for x, y in test_loader:
        with torch.no_grad():
            x, y = Variable(x.to(device)), Variable(y.to(device))

        y_pred = model(x)
        test_loss += capsule_loss_fn(y_pred, y).item() * x.shape[0]
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        test_accuracy += y_pred.eq(y_true).cpu().sum()

    test_loss = torch.true_divide(test_loss, len(test_loader.dataset))
    test_accuracy = torch.true_divide(test_accuracy, len(test_loader.dataset))
    return test_loss.item(), test_accuracy.item()
