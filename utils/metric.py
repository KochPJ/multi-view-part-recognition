"""Computes the TopK Accuracy of a given model.
"""
import torch
import numpy as np


class TopKAccuracy(object):
    def __init__(self, k: int):
        self.res = []
        self.i = []
        self.k = k

    def add(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k.

        Arguments:
            output (tensor): prediction of a classifier
            target (tensor): labels corresponding to input of classifier
        """
        with torch.no_grad():
            batch_size = target.size(0)

            # for one hot encoding take the maximum argument of the target
            if target.dim() == 2:
                batch_size = target.size(0)
                conf, pred = output.topk(self.k, 1, True, True)
                pred = pred.t()

                # convert one hot to a prediction
                target = torch.argmax(target, dim=1)
            else:
                _, pred = output.topk(self.k, 1, True, True)
                pred = pred.t()

            correct = pred.eq(target[None])

            correct_k = correct[:self.k].flatten().sum(dtype=torch.float32).cpu().numpy()
            self.res.append(correct_k * (100.0 / batch_size))
            self.i.append(batch_size)

    def reset(self):
        """Resets the memory of the metric.
        """
        self.res = []
        self.i = []

    def result(self):
        """Calculates the resulting accuracy.

        Returns:
            float: resulting accuracy
        """
        n_images = np.sum(self.i)
        result = 0
        for i, res in enumerate(self.res):
            result += (self.i[i]/n_images)*res
        return float(result)
