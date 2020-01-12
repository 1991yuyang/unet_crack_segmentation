import torch as t
from torch import nn
import os
from model_define import Unet
import cv2
import numpy as np
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def predict(image_path):
    img = Image.open(image_path)
    img.show()
    image_size = img.size
    image = cv2.imread(image_path).astype(np.int16)
    image = cv2.resize(image, (388, 388))
    image = np.transpose(cv2.copyMakeBorder(image, 92, 92, 92, 92, cv2.BORDER_REFLECT), [2, 0, 1])
    image = t.tensor(image).type(t.FloatTensor).unsqueeze(0).cuda(0)
    softmax_op = nn.Softmax(dim=1)
    with t.no_grad():
        output = model(image)
    softmax_result = softmax_op(output)
    predict_result = t.argmax(softmax_result, dim=1).squeeze().cpu().detach().numpy() * 255
    predict_result = cv2.resize(predict_result, image_size, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(predict_result.astype(np.uint8)).show()


if __name__ == "__main__":
    model = Unet(3, 2, True)  #
    model = nn.DataParallel(module=model, device_ids=[0])
    model = model.cuda(0)
    model.load_state_dict(t.load("./model_save/best_model.pth"))
    model.eval()
    image_path = "./dataset/image_test/320.jpg"
    predict(image_path)