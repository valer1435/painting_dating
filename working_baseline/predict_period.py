from PIL import Image
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

from sklearn.linear_model import SGDClassifier
import joblib


random_seed = 4545435
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StyleMatrix(nn.Module):

    def __init__(self):
        super(StyleMatrix, self).__init__()

    def forward(self, inp):
        G = gram_matrix(inp)
        return G


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    
    
def gram_matrix(inp):
    a, b, c, d = inp.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = inp.view(a * b, c * d).to(device)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


from numpy import load

mask = load('features8000.npy')
len(mask)


PATH_TO_VGG = "cutted_vgg19.pth"
cnn = models.vgg19(pretrained=False).features.to(device)
cut_vgg19 = torch.nn.Sequential(*(list(cnn.children())[:26])+[StyleMatrix()])

cut_vgg19.load_state_dict(torch.load(PATH_TO_VGG))
cut_vgg19.eval()

cpu = torch.device("cpu")



final_model = joblib.load('sgd.pkl')

target_dict= {
    0: "до 1300",
    1:"1300-1350",
    2:"1351-1400",
    3:"1401-1450",
    4:"1451-1500",
    5:"1501-1550",
    6:"1551-1600",
    7:"1601-1650",
    8:"1651-1700",
    9:"1701-1750",
    10:"1751-1800",
    11:"1801-1850",
    12:"1851-1900",
}

OFFSET = 16



transformations = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def make_prediction(img, device):
    if img.size != (224, 224):
        img = img.resize((224, 244))
    img_tensor = transformations(img)
    G_matrix = cut_vgg19(torch.reshape(img_tensor, (1, 3, 224, 224)).to(device))
    G_vector_numpy = G_matrix.to(cpu).detach().numpy().reshape((1,512*512))[0][mask]
    res = final_model.predict(G_vector_numpy.reshape(1, -1)).item() - OFFSET

    predicted_period = target_dict[res]
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Наиболее вероятный период написания картины: {predicted_period}")
    return res, predicted_period, img


trans = transforms.ToPILImage()

img = Image.open("../resized_images/1751-1800/3211.jpg")

make_prediction(img, device)
