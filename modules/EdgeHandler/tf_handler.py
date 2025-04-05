import tensorflow as tf
import tensorflow_model_optimization as tfmot
from .base_handler import BaseModelHandler

class TensorFlowHandler(BaseModelHandler):
    def optimize_model(self, model, techniques):
        if 'pruning' in techniques:
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.2, final_sparsity=0.5,
                begin_step=0, end_step=1000
            )
            model = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

        if 'quantization' in techniques:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            model = converter.convert()

        return model

    def convert_model(self, model, target_format):
        if target_format == 'tflite':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open("converted_tf_model.tflite", "wb") as f:
                f.write(tflite_model)
        return model

    def evaluate_model(self, model, test_data):
        loss, acc = model.evaluate(*test_data, verbose=0)
        return {"loss": loss, "accuracy": acc}