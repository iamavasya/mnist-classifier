import unittest
import torch
from torchvision import transforms
from PIL import Image


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def test_image_transformation(self):
        dummy_image = Image.new('RGB', (500, 500), color='white')

        tensor_out = self.transform(dummy_image)

        self.assertEqual(tensor_out.shape, (1, 28, 28), "Transformation should return a tensor of shape (1, 28, 28).")

        self.assertTrue(isinstance(tensor_out, torch.FloatTensor) or isinstance(tensor_out, torch.Tensor))