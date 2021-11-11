import numpy as np
import torch
import torchvision
from PIL import Image


def create_square_crop_by_detection(
        frame: np.ndarray,
        box: list,
        return_shifts: bool = False,
        zero_pad: bool = True):
    """
    Rebuild detection box to square shape
    Args:
        frame: rgb image in np.uint8 format
        box: list with follow structure: [x1, y1, x2, y2]
        return_shifts: if set True then function return tuple of image crop
           and (x, y) tuple of shift coordinates
        zero_pad: pad result image by zeros values

    Returns:
        Image crop by box with square shape or tuple of crop and shifted coords
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w // 2
    cy = box[1] + h // 2
    radius = max(w, h) // 2
    exist_box = []
    pads = []

    # y top
    if cy - radius >= 0:
        exist_box.append(cy - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cy - radius))

    # y bottom
    if cy + radius >= frame.shape[0]:
        exist_box.append(frame.shape[0] - 1)
        pads.append(cy + radius - frame.shape[0] + 1)
    else:
        exist_box.append(cy + radius)
        pads.append(0)
    # x left
    if cx - radius >= 0:
        exist_box.append(cx - radius)
        pads.append(0)

    else:
        exist_box.append(0)
        pads.append(-(cx - radius))

    # x right
    if cx + radius >= frame.shape[1]:
        exist_box.append(frame.shape[1] - 1)
        pads.append(cx + radius - frame.shape[1] + 1)
    else:
        exist_box.append(cx + radius)
        pads.append(0)

    exist_crop = frame[
                 exist_box[0]:exist_box[1],
                 exist_box[2]:exist_box[3]
                 ].copy()

    croped = np.pad(
        exist_crop,
        (
            (pads[0], pads[1]),
            (pads[2], pads[3]),
            (0, 0)
        ),
        'reflect' if not zero_pad else 'constant',
        constant_values=0
    )

    if not return_shifts:
        return croped

    shift_x = exist_box[2] - pads[2]
    shift_y = exist_box[0] - pads[0]

    return croped, (shift_x, shift_y)


class StandardClassification(object):
    def __init__(self,
                 traced_model_file: str,
                 file_names_file: str,
                 device: str = 'cpu'):
        self.model = torch.jit.load(traced_model_file)
        self.model.eval()
        self.model = self.model.to(device)
        self.input_shape = (224, 224)
        self.device = device

        with open(file_names_file, 'r') as f:
            self.classes_names = [line.rstrip() for line in f]

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.input_shape),
                torchvision.transforms.ToTensor()
            ]
        )

    def _preprocess(self, _input_img: Image) -> torch.Tensor:
        _image = np.array(_input_img.convert('RGB'))
        _image = create_square_crop_by_detection(
            _image,
            [0, 0, _image.shape[1], _image.shape[0]]
        )
        _image = Image.fromarray(_image)
        inp_tensor = self.transform(_image).unsqueeze(0).to(self.device)
        return inp_tensor

    def __call__(self, _img: Image) -> tuple:
        out = self.model(self._preprocess(_img)).detach().to('cpu')[0]
        return self.classes_names[out.argmax()], \
               float('{:.3f}'.format(torch.sigmoid(out).max().numpy()))

    def get_prediction(self, _img: Image) -> tuple:
        out = self.model(self._preprocess(_img)).detach().to('cpu')[0]
        return out.numpy(), self.classes_names[out.argmax()]
