from .EdgeHandler.tf_handler import TensorFlowHandler
from .EdgeHandler.torch_handler import TorchHandler
import tensorflow as tf
import torch


class EdgeOptimizer:
    def __init__(self, model):
        self.model = model
        self.handler = self._get_handler()

    def _get_handler(self):
        if isinstance(self.model, tf.keras.Model):
            return TensorFlowHandler(self.model)
        elif isinstance(self.model, torch.nn.Module):
            return TorchHandler(self.model)
        else:
            raise TypeError("Unsupported model type")

    def optimize(self, techniques):
        return self.handler.optimize(techniques)

    def convert(self, target_format):
        return self.handler.convert(target_format)

    def evaluate(self, test_data):
        return self.handler.evaluate(test_data)