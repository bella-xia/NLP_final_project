import torch
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder


if __name__ == "__main__":
    PATH_TO_FORWARD = "params/opengpt2_pyopengpt2_pytorch_backward"
    PATH_TO_BACKWARD = "params/opengpt2_pytorch_forward"
    device_forward = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_backward = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_forward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FORWARD).to(device_forward)
    model_backward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_BACKWARD).to(device_backward)
    encoder = Encoder().to(device_forward)

    input_text = ' And that was the last I heard from her.'
    input_tokens = encoder.encode(input_text)[::-1]

    output = model_backward.generate(torch.tensor([input_tokens]).to(device_backward))
    output_tokens = output.tolist()[0][::-1]
    print(encoder.decode(output_tokens))