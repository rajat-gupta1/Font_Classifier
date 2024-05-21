from torch import nn
from torchvision import transforms, datasets
from PIL import Image

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) with two convolutional blocks
    and a fully connected classifier.

    Args:
        input_shape (int): Number of input channels (e.g., 3 for RGB images).
        hidden_units (int): Number of hidden units in the convolutional layers.
        output_shape (int): Number of output units (e.g., number of classes for classification).
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()

        # First convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2)
        )

        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2)
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units * 13 * 13,
                      out_features = output_shape)
        )

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

class CustomResizeAndSplit(transforms.Resize):
    """
    A custom transformation that resizes an image to a specified size
    and splits it into equal parts based on the target aspect ratio.

    Args:
        size (tuple): The desired output size (height, width).
        interpolation (int, optional): The interpolation method to use
                                       for resizing. Default is Image.BILINEAR.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        Initializes the CustomResizeAndSplit transformation.
        """
        super().__init__(size, interpolation)
        self.target_ratio = size[1] / size[0]   # Here, size[0] is the height

    def __call__(self, img, cnt):
        """
        Resizes and splits the image.

        Args:
            img (PIL.Image): The input image to be transformed.
            cnt (int): The index of the split part to return.

        Returns:
            PIL.Image: The resized and split image part.
        """
        width, height = img.size
        ratio = width / height

        # Number of splits to create for the image
        num_splits = int(ratio // self.target_ratio)

        # Width of each split
        split_width = width // num_splits
        left = cnt * split_width
        right = (cnt + 1) * split_width
        split_img = img.crop((left, 0, right, height))
        return super().__call__(split_img)


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths.

    Extends the ImageFolder dataset to also return the file path of each image.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # Override the __getitem__ method. This is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path