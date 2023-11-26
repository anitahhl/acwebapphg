import torch
import torch.onnx
from PIL import Image

import utils
from transformer_net import StyleTransferNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    ckpt = torch.load('./static/model.ckpt', map_location=device)
    model = StyleTransferNetwork()
    print('torch load')
    model.load_state_dict(ckpt['state_dict'])
    print('load_state_dict')
    model.eval()
    print('eval')
    return model


def stylize(_style_model, style_index, content_image, output_image):
    content_image = utils.imload(content_image, imsize=512)
    style_code = torch.zeros(1, 16, 1)
    style_code[:, style_index, :] = 1
    
    result = _style_model(content_image, style_code)
    utils.save_image(output_image, result)


def get_result(input_image, style_index):
    model_path = "./static/model.ckpt"
    output_image = "./static/images/output-images/result.jpg"

    model = load_model(model_path)
    stylize(model, style_index, input_image, output_image)

    return output_image
