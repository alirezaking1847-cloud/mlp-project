
import torch
import torch.nn as nn

from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE


class MLPModel(nn.Module):

    def __init__(self):
        super(MLPModel, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),

            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),

            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )

    def forward(self, x):

        return self.network(x)


def create_model():

    model = MLPModel()

    return model

