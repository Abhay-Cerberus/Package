from modules import EdgeOptimizer
import tensorflow as tf
import torch
import torch.nn as nn

def test_tensorflow_model():
    print("\nTesting TensorFlow Model")

    # Dummy TF model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dummy_data = tf.random.normal([5, 28, 28])
    dummy_labels = tf.random.uniform([5], maxval=10, dtype=tf.int32)

    optimizer = EdgeOptimizer(model)

    print("\nEvaluating before optimization:")
    pre_metrics = optimizer.evaluate((dummy_data, dummy_labels))
    print("Pre-optimization metrics:", pre_metrics)

    print("\nRunning optimization...")
    optimizer.optimize(['pruning', 'quantization'])

    print("\nConverting model...")
    optimizer.convert('tflite')

    print("\nEvaluating after optimization:")
    post_metrics = optimizer.evaluate((dummy_data, dummy_labels))
    print("Post-optimization metrics:", post_metrics)


def test_pytorch_model():
    print("\nTesting PyTorch Model")

    # Dummy Torch model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    model = SimpleNet()
    dummy_data = torch.randn(5, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (5,))

    optimizer = EdgeOptimizer(model)

    print("\nEvaluating before optimization:")
    pre_metrics = optimizer.evaluate((dummy_data, dummy_labels))
    print("Pre-optimization metrics:", pre_metrics)

    print("\nRunning optimization...")
    optimizer.optimize(['pruning', 'quantization'])

    print("\nConverting model...")
    optimizer.convert('torchscript')

    print("\nEvaluating after optimization:")
    post_metrics = optimizer.evaluate((dummy_data, dummy_labels))
    print("Post-optimization metrics:", post_metrics)


if __name__ == "__main__":
    test_tensorflow_model()
    test_pytorch_model()
