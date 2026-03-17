import unittest
import os
import io
import torch
from unittest.mock import patch
from PIL import Image
from src.model import MNISTClassifier
from src.predict import predict_digit


class TestIntegrationPredict(unittest.TestCase):
    def setUp(self):
        self.test_img_path = "test_dummy_digit.png"
        img = Image.new('RGB', (28, 28), color='black')
        img.save(self.test_img_path)

        self.dummy_model = MNISTClassifier()
        self.dummy_state_dict = self.dummy_model.state_dict()

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    @patch('torch.load')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_predict_digit_output(self, mock_stdout, mock_torch_load):
        mock_torch_load.return_value = self.dummy_state_dict

        predict_digit(self.test_img_path, correctNumber=None)

        output = mock_stdout.getvalue()

        self.assertIn("--- RESULT ---", output)
        self.assertIn("Digit on image:", output)
        self.assertIn("Model confidence:", output)

    @patch('torch.save')
    @patch('torch.load')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_fix_model_training(self, mock_stdout, mock_torch_load, mock_torch_save):
        mock_torch_load.return_value = self.dummy_state_dict

        predict_digit(self.test_img_path, correctNumber=5)

        output = mock_stdout.getvalue()

        self.assertIn("Model trained on this image! ✅", output)

        mock_torch_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()