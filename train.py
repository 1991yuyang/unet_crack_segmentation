from torch import nn, optim
import torch as t
from model_define import Unet
import os
from dataPrepare import make_loader
from numpy import random as rd
import numpy as np
with open("./train_conf.json", "r", encoding="utf-8") as file:
    train_conf = eval(file.read())
cuda_visible_devices = train_conf["cuda_visible_devices"]
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def train():
    valid_good_perform_times = train_conf["valid_good_perform_times"]
    valid_loss_lst = [float("inf")] * valid_good_perform_times
    mini_valid_loss = np.sum(valid_loss_lst) / valid_good_perform_times
    train_image_path = train_conf["train_image_path"]
    train_mask_path = train_conf["train_mask_path"]
    valid_image_path = train_conf["valid_image_path"]
    valid_mask_path = train_conf["valid_mask_path"]
    model_save_path = train_conf["model_save_path"]
    positive_weight = train_conf["positive_weight"]
    negative_weight = train_conf["negative_weight"]
    weight = t.tensor([negative_weight, positive_weight]).type(t.FloatTensor)
    batch_size = train_conf["batch_size"]
    epoch = train_conf["epoch"]
    lr = train_conf["lr"]
    device_ids = train_conf["device_ids"]
    model = Unet(3, 2)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    if os.path.exists(model_save_path):
        print("load model......")
        model.load_state_dict(t.load(model_save_path))
        with open(os.path.join(model_save_path.rstrip("best_model.pth"), "mini_valid_loss.txt"), "r", encoding="utf-8") as file:
            mini_valid_loss = float(file.read())
            valid_loss_lst = [mini_valid_loss] * valid_good_perform_times
    criterion = nn.CrossEntropyLoss(weight=weight).cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    for e in range(1, 1 + epoch):
        train_loader = make_loader(train_image_path, train_mask_path, batch_size)
        valid_loader = make_loader(valid_image_path, valid_mask_path, batch_size)
        step = 0
        for img_train, msk_train in train_loader:
            step += 1
            model.train()
            img_train_cuda = img_train.cuda(device_ids[0])
            msk_train_cuda = msk_train.cuda(device_ids[0])
            train_output = model(img_train_cuda)
            train_loss = criterion(train_output, msk_train_cuda)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            try:
                img_valid, msk_valid = next(valid_loader)
            except:
                valid_loader = make_loader(valid_image_path, valid_mask_path, batch_size)
                img_valid, msk_valid = next(valid_loader)
            img_valid_cuda = img_valid.cuda(device_ids[0])
            msk_valid_cuda = msk_valid.cuda(device_ids[0])
            model.eval()
            with t.no_grad():
                valid_output = model(img_valid_cuda)
                valid_loss = criterion(valid_output, msk_valid_cuda).item()
            print("epoch %d step %d valid loss: %f" % (e, step, valid_loss))
            valid_loss_lst.pop(0)
            valid_loss_lst.append(valid_loss)
            if np.sum(valid_loss_lst) / valid_good_perform_times < mini_valid_loss:
                mini_valid_loss = np.sum(valid_loss_lst) / valid_good_perform_times
                print("saving model......")
                t.save(model.state_dict(), model_save_path)
                with open(os.path.join(model_save_path.rstrip("best_model.pth"), "mini_valid_loss.txt"), "w", encoding="utf-8") as file:
                    file.write(str(mini_valid_loss))
        if e % int(0.2 * epoch) == 0:
            lr = 0.1 * lr
            if e < int(0.5 * epoch):
                optimizer = optim.Adam(params=model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=rd.choice([0.7, 0.8, 0.9]))


if __name__ == "__main__":
    train()
