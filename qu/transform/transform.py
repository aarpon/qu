import numpy as np


def label_image_to_binary_mask(
        img: np.ndarray,
        background_label: int = 0
) -> np.ndarray:
    """Converts a label image to a black-and-white binary mask.

    :param img: np.ndarray
        Label image.

    :param background_label: int
        Label of the background (default is 0).

    The background label will be set to 0 (if it is not), and all other labels
    are converted to 1.

    :return: black (0) and white (1) mask (np.ndarray) with the same data type as img.
    """

    # Make a copy of the image
    mask = img.copy()

    # Set background label to 0 (if it is not)
    if background_label != 0:
        mask[mask == background_label] = 0

    # Set all other labels to 1
    mask[mask > background_label] = 1

    # Return
    return mask


def binary_masks_to_label_image(masks: tuple) -> np.ndarray:
    """Convert a tuple of mask images to a single label image where each pixel
    corresponds to the index of the mask that contains it as a foreground pixel.

    @param masks: Tuple
        Tuple of images to combined into a label image.

    @return class_image: np.ndarray
        Class image.
    """

    # Initialize output image
    labels = np.zeros(masks[0].shape, dtype=np.int32)

    # Process the images
    for i in range(len(masks)):
        labels[masks[i] > 0] = i + 1

    # Return the class image
    return labels


def label_image_to_one_hot_stack(
        label: np.ndarray,
        num_classes: int = -1,
        channels_first: bool = False,
        dtype: np.dtype = np.uint8
) -> np.ndarray:
    """Converts a label image into a stack of binary images (one-hot).

    The binary image at index 'i' in the stack contains the pixels from the
    class i in the label image. The background class is also considered.

    It is recommended to set the background to class 0 in the label image.

    @param label: np.ndarray
        Label image. Each pixel intensity corresponds to one class.

    @param num_classes: int
        Expected number of classes. If omitted, the number of classes will be
        extracted from the image. When processing a series of images, it is
        best to explicitly set num_classes, otherwise some image stacks may
        have less planes if a class happens to be missing from those images.

    @param channels_first: bool
        Toggles between [C, H, W] and [H, W, C] geometries. The default geometry
        is [H, W, C].

    @param dtype: np.dtype
        By default the output stack will have dtype np.uint8.

    @return a stack of images with the same width and height as label and
    with num_classes planes.
    """

    label = label.astype(np.int32)

    min_class = int(label.min(initial=None))
    max_class = int(label.max(initial=None))

    if num_classes == -1:
        num_classes = max_class - min_class + 1

    # Allocate the stack
    if channels_first:
        stack = np.zeros(
            (
                num_classes,
                label.shape[0],
                label.shape[1]
            ),
            dtype=dtype
        )
    else:
        stack = np.zeros(
            (
                label.shape[0],
                label.shape[1],
                num_classes),
            dtype=dtype
        )

    # Now process the classes
    for i in range(num_classes):

        if channels_first:
            tmp = stack[i, :, :]
        else:
            tmp = stack[:, :, i]

        # By reference, the data is changed in 'stack'
        tmp[label == i] = 1

    return stack


def one_hot_stack_to_label_image(
        stack: np.ndarray,
        first_index_is_background: bool = True,
        channels_first: bool = False,
        dtype: np.dtype = np.uint16
) -> np.ndarray:
    """Converts a stack of binary images (one-hot) into a label image.

    The binary image at index 'i' in the stack contains the pixels from the
    class i in the label image.

    It is recommended to set the background to class 0 in the label image.

    @param stack: np.ndarray
        Stack of images with the same width and height as label and either first or
        last dimension equal to the number of classes.

    @param first_index_is_background: bool
        If True, the first image is the background and will get class value 0 in
        the label image. If False, it will get class value 1.

    @param channels_first: bool
        Toggles between [C, H, W] and [H, W, C] geometries. The default geometry
        is [H, W, C].

    @param dtype: np.dtype
        By default the output stack will have dtype np.uint16.

    @return a label image where each pixel intensity corresponds to one class.
    """

    if channels_first:
        num_classes = stack.shape[0]
        height = stack.shape[1]
        width = stack.shape[2]
    else:
        num_classes = stack.shape[2]
        height = stack.shape[0]
        width = stack.shape[1]

    # Initialize the label image
    label = np.zeros((height, width), dtype=dtype)

    # Not process the stack
    c = 1
    for i in range(num_classes):

        # Current class
        if channels_first:
            tmp = stack[i, :, :]
        else:
            tmp = stack[:, :, i]

        if i == 0 and first_index_is_background is True:
            continue

        label[tmp > 0] = c

        # Update current class
        c += 1

    return label
