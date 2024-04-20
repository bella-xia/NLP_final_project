import torch, subprocess
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder

def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))

def get_forward_backward_preds(model, encoder, input_text, is_forward, device):
    input_tokens = encoder.encode(input_text) if is_forward else encoder.encode(input_text)[::-1]
    pred = model.generate(torch.tensor([input_tokens]).to(device))
    # print(pred)
    pred_tokens = pred.tolist()[0] if is_forward else pred.tolist()[0][::-1]
    print("model prediction: ", encoder.decode(pred_tokens))

if __name__ == "__main__":
    #PATH_TO_FORWARD = "/home/cs601-zxia15/NLP_final_project/params/opengpt2_pytorch_forward"
    # PATH_TO_BACKWARD = "/home/zxia15/NLP_final_project/params/opengpt2_pytorch_backward"
    PATH_TO_FINE_TUNED_FORWARD = "/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_forward"
    PATH_TO_FINE_TUNED_BACKWARD = "/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_backward"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_forward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FINE_TUNED_FORWARD).to(device)
    model_backward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FINE_TUNED_BACKWARD).to(device)
    encoder = Encoder()

    RESPONSE = "It's hard to say how long the leftovers will keep."
    QUERY = "How long will my leftovers keep refrigerated?"

    input_text = ' not very interesting paper by Peter West.'

    # print("forward: ")
    # get_forward_backward_preds(model_forward, encoder, input_text, True, device)
    # print("backward")
    get_forward_backward_preds(model_backward, encoder, RESPONSE, False, device)
    # print_gpu_memory()
    