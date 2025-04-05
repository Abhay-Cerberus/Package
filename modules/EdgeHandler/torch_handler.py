import torch
import torch.nn.utils.prune as prune
from .base_handler import BaseModelHandler

class TorchHandler(BaseModelHandler):
    def optimize_model(self, model, techniques):
        if 'pruning' in techniques:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.5)

        if 'quantization' in techniques:
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            dummy_input = torch.rand(1, 28, 28)
            model(dummy_input)
            torch.quantization.convert(model, inplace=True)

        return model

    def convert_model(self, model, target_format):
        if target_format == 'torchscript':
            return torch.jit.script(model)
        return model

    def evaluate_model(self, model, test_data):
        model.eval()
        x, y = test_data
        with torch.no_grad():
            out = model(x)
            preds = torch.argmax(out, dim=1)
            acc = (preds == y).float().mean().item()
        return {"accuracy": acc}
    