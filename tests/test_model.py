import unittest
import torch
from src.model import MNISTClassifier

class TestMNISTClassifier(unittest.TestCase):
    def setUp(self):
        self.model = MNISTClassifier()

    def test_output_shape(self):
        dummy_input = torch.randn(4, 1, 28, 28)
        output = self.model(dummy_input)

        self.assertEqual(output.shape, (4, 10), "Size of output tensor should be 4x10")

    def test_device_movement(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_on_device = self.model.to(device)
        dummy_input = torch.randn(4, 1, 28, 28).to(device)

        try:
            output = model_on_device(dummy_input)
            self.assertEqual(output.device.type, device.type)
        except RuntimeError as e:
            self.fail(f"Model failed on {device}: {e}")