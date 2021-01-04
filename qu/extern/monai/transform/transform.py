from monai.transforms import Transform, torch
import numpy as np

from monai.data import itk

from qu.transform.transform import label_image_to_onehot_stack


class ToOneHot(Transform):
    """Converts a label image into a stack of binary images (one-hot).

    The binary image at index 'i' in the stack contains the pixels from the
    class i in the label image.

    It is recommended to set the background to class 0 in the label image.

    The OneHot stack is in the form [CxHxW]

    Args:

        num_classes: int
            Expected number of classes. If omitted, the number of classes will be
            extracted from the image. When processing a series of images, it is
            best to explicitly set num_classes, otherwise some image stacks may
            have less planes if a class happens to be missing from those images.
    """

    def __init__(self, num_classes: int) -> None:
        """Constructor"""
        self.num_classes = num_classes

    def __call__(self, label: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `label`.

        @return a stack of images with the same width and height as `label` and
            with `num_classes` planes.
        """

        # Make sure 'label' is a numpy array
        label = np.array(label)

        return label_image_to_onehot_stack(
            label,
            num_classes=self.num_classes,
            channels_first=True,
            dtype=np.float32
        )


class Informer(Transform):
    """
    Simple reporter to be added to a Composed list of Transforms
    to return some information.
    """
    def __init__(self, *args, **kwargs):
        """Constructor.

        @param name: str
            Name of the Informer. It is printed as a prefix to the output.
        """
        super().__init__()
        self.name = ""
        if "name" in kwargs:
            self.name = kwargs['name']

    def __call__(self, data):
        """Call the Transform."""
        prefix = f"{self.name} :: " if self.name != "" else ""
        if type(data) == torch.Tensor:
            print(f"{prefix}Torch tensor: Size = {data.size()}, type = {data.dtype}, min = {data.min()}, max = {data.max()}")
        elif type(data) == np.ndarray:
            print(f"{prefix}Numpy array: Size = {data.shape}, type = {data.dtype}, min = {data.min()}, max = {data.max()}")
        elif type(data) == str:
            print(f"{prefix}String: value = '{str}'")
        elif type(data) == itk.itkPyBufferPython.NDArrayITKBase:
            print(f"{prefix}ITK array: Size = {data.shape}, type = {data.dtype}, min = {data.min().item()}, max = {data.max().item()}")
        else:
            print(f"{prefix}{type(data)}: {str(data)}")
        return data
