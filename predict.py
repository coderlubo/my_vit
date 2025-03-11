import json
import os
from matplotlib import pyplot as plt
import torch

from PIL import Image

from utils.settings import DEVICE

from torchvision import transforms

from model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # load image
    img_path = "/home/llb/workspace/my_vit/predict_image/image.jpg"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    img = Image.open(img_path)

    img = data_transform(img) # [N, C, H, W]
    img = torch.unsqueeze(img, dim=0) # expand batch dimension

    # read class_indict
    json_path = '/home/llb/workspace/my_vit/utils/class_indices.json'

    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=5, has_logits=False).to(device)

    # load model weights
    model_weight_path = "/home/llb/workspace/my_vit/output/2025-03-10_10:05:05/weights/model-2025-03-10_10:05:05.pth"

    weights_dict = torch.load(model_weight_path, map_location=device)

    # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    
    for k in del_keys:
        del weights_dict[k]

    
    model.load_state_dict(weights_dict, strict=False)

    model.eval()

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()


    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))


    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    print("\nResult: {}".format(print_res))


if __name__ == '__main__':
    main()
        