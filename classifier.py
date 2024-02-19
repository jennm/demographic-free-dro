import torch
import torch.nn as nn
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return nn.functional.softmax(self.linear(x), dim=1)


def loss_fn(outputs, labels):
    ones = -1 *labels*torch.log(outputs)
    zeros = (1-labels)*torch.log(1 - outputs)
    print(f"1: {ones}, 0: {zeros}")
    return 10 * ones - zeros
    # torch.sump