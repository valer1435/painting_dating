from PIL import Image
import torch

import torchvision.models as models
from torchvision import transforms

import joblib
import numpy as np


class PeriodPredictor:
    PATH_TO_VGG = 'cut_vgg19.pth'
    PATH_TO_FINAL_MODEL = 'linear_svc_balanced.pkl'
    PATH_TO_FEATURE_MASK = 'features8000.npy'
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    INPUT_SIZE = (224, 224)

    NUM_CHANNELS = 3
    INPUT_IMAGE_WIDTH = 224
    OUTPUT_GRAM_VECTOR_LENGTH = 512 * 512
    target_dict = {
        0: "до 1300",
        1: "1300-1350",
        2: "1351-1400",
        3: "1401-1450",
        4: "1451-1500",
        5: "1501-1550",
        6: "1551-1600",
        7: "1601-1650",
        8: "1651-1700",
        9: "1701-1750",
        10: "1751-1800",
        11: "1801-1850",
        12: "1851-1900",
    }
    OFFSET = 16

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        self.cut_cnn = torch.nn.Sequential(
            *(list(models.vgg19(pretrained=False).features.to(self.device).children())[:26]))
        self.cut_cnn.load_state_dict(torch.load(PeriodPredictor.PATH_TO_VGG))
        self.cut_cnn.eval()

        self.final_model = joblib.load(PeriodPredictor.PATH_TO_FINAL_MODEL)

        self.features_mask = np.load(PeriodPredictor.PATH_TO_FEATURE_MASK)

    @staticmethod
    def resize_and_convert_if_need(image):
        image = image.convert('RGB')
        if image.size != PeriodPredictor.INPUT_SIZE:
            image = image.resize(PeriodPredictor.INPUT_SIZE)
        return image

    def gram_matrix(self, inp):
        a, b, c, d = inp.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = inp.view(a * b, c * d).to(self.device)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def predict(self, x):
        x = PeriodPredictor.transformations(x)

        g_matrices_tensor = self.gram_matrix(self.cut_cnn(
            torch.reshape(x, (-1,
                              PeriodPredictor.NUM_CHANNELS,
                              PeriodPredictor.INPUT_IMAGE_WIDTH,
                              PeriodPredictor.INPUT_IMAGE_WIDTH)
                          ).to(self.device)
            )
        )

        g_vectors_numpy = g_matrices_tensor.to(self.cpu).detach().numpy().reshape((-1,
                                                                                   PeriodPredictor.OUTPUT_GRAM_VECTOR_LENGTH)
                                                                                  )[0][self.features_mask]
        res = self.final_model.predict(g_vectors_numpy.reshape(1, -1)).item() - PeriodPredictor.OFFSET

        return PeriodPredictor.target_dict[res]


if __name__ == '__main__':
    img = PeriodPredictor.resize_and_convert_if_need(Image.open("1898.jpg"))
    predictor = PeriodPredictor()
    print(predictor.predict(img))
