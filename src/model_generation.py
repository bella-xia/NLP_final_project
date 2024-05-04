import torch, json, random
import torch.nn.functional as F
from tqdm import tqdm
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder
from gpt2_finetune import read_json

def model_generate(
        model, idx, max_new_tokens=100, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # predictable_tokens = max_new_tokens - idx.size()[1]
        predictable_tokens = 60
        model.eval()
        with torch.no_grad():
            for _ in range(predictable_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx
                # forward the model to get the logits for the index in the sequence
                logits = model(idx_cond)['logits']
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # either sample from the distribution or take the most likely element
                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                 _, idx_next = torch.topk(probs, k=1, dim=-1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
                # last_token += 1

        return idx

def get_generation(input_tokens, expected_output, model, encoder, is_backward=False):
    # input_tokens = encoder.encode(input_text) if is_forward else encoder.encode(input_text)[::-1]
    temp_choice = [0.5, 1.0, 1.5]
    top_k = [None, 5, 15, 30, 100]
    preds_dict = {}
    preds_dict['expected output'] = expected_output
    preds_dict['input'] = encoder.decode(torch.flip(input_tokens, [0]))
    pred= model_generate(model, torch.unsqueeze(input_tokens, dim=0).to(device))
    # pred_tokens = .tolist()[0][::-1] if is_backward else pred.tolist()[0]
    preds_dict['default'] = encoder.decode(torch.flip(pred[0], [0])) if is_backward else encoder.decode(pred[0])
    # print(f"model prediction with default first choice:\n {decoded_pred}\n", )
    for temp in temp_choice:
        for k in top_k:
            pred= model_generate(model, torch.unsqueeze(input_tokens, dim=0).to(device), temperature=temp, do_sample=True, top_k=k)
            # pred_tokens = pred.tolist()[0][::-1] if is_backward else pred.tolist()[0]
            preds_dict[f'temp {temp} top_k {k}'] = encoder.decode(torch.flip(pred[0], [0])) if is_backward else encoder.decode(pred[0])
            # print(f"model prediction with temperature {temp} top_k {k} multinomial sampling: {encoder.decode(pred_tokens)}")
    return preds_dict

def get_generation_with_temp_one_and_half_k_thirty(input_tokens, model, encoder):
    TEMP = 1.5
    TOP_K = 30
    preds_dict = {}
    preds_dict['input'] = encoder.decode(input_tokens)
    pred= model_generate(model, torch.unsqueeze(input_tokens, dim=0).to(device), temperature=TEMP, do_sample=True, top_k=TOP_K)
    pred_tokens = pred.tolist()[0]
    preds_dict[f'temp 1.5 top_k 30'] = encoder.decode(pred_tokens)
    # print(f"model prediction with temperature {temp} top_k {k} multinomial sampling: {encoder.decode(pred_tokens)}")
    return preds_dict

def process_data(instruction_text, output_text, encoder, is_backward=False, max_length=100):
    if is_backward:
        encoded_instruction_text = torch.flip(torch.tensor(encoder.encode('[SEP]' + output_text)), [0])
        encoded_output_text = torch.flip(torch.tensor(encoder.encode(instruction_text)), [0])
    else:
        encoded_instruction_text = torch.tensor(encoder.encode(instruction_text + '[SEP]'))
        encoded_output_text = torch.tensor(encoder.encode(output_text))
    encoded_instruction_length = len(encoded_instruction_text)
    encoded_output_length = len(encoded_output_text)
     # ...(padded / truncated)... x x x x (prompt) x x x x x x (ouput) ...(padded / truncated)...

    # step for truncation
    if (encoded_output_length + encoded_instruction_length > max_length):
        # print("truncating!")
        truncated_length = encoded_output_length + encoded_instruction_length - max_length
        if (encoded_instruction_length > encoded_output_length * 2 and max_length > encoded_output_length * 2):
            output_truncated_length = 0
            instruction_truncated_length = truncated_length
        elif (encoded_output_length > encoded_instruction_length * 2 and max_length > encoded_instruction_length * 2):
            output_truncated_length = truncated_length
            instruction_truncated_length = 0
        else:
            truncation_ratio = float(encoded_output_length) / float(encoded_output_length + encoded_instruction_length)
            output_truncated_length = int(truncated_length * truncation_ratio)
            instruction_truncated_length = truncated_length - output_truncated_length
        # print(io_encoded_length, instruction_encoded_length, io_truncated_length, instruction_truncated_length)
        return encoded_instruction_text[instruction_truncated_length:].to(torch.int64)
    else:
        return encoded_instruction_text.to(torch.int64)


if __name__ == "__main__":
    PATH_TO_MODEL = "/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_backward_alcapa"
    # PATH_TO_MODEL = "/home/zxia15/NLP_final_project/params/opengpt2_pytorch_backward"
    PATH_TO_DATASET = "/home/zxia15/NLP_final_project/data/alpaca-clean/no_input_alphaca_data_cleaned.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_forward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FINE_TUNED_FORWARD).to(device)
    model_finetuned_backward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_MODEL).to(device)
    encoder = Encoder()
    encoder.add_special_tokens(["[SEP]"])

    data = read_json(PATH_TO_DATASET)
    print(len(data))

    START_POINT = int(len(data) * 0.8)
    END_POINT = int(len(data))

    random_idx = [random.randint(0, START_POINT) for _ in range(10)]
    random_idx.extend([random.randint(START_POINT , END_POINT) for _ in range(10)])

    print(random_idx)
    output_arr = []
    for idx in tqdm(random_idx):
        input_tokens = process_data(data[idx]['instruction'], data[idx]['output'], encoder, is_backward=True)
        output_dict = get_generation(input_tokens, data[idx]['instruction'], model_finetuned_backward, encoder, is_backward=True)
        output_arr.append(output_dict)
    json_string = json.dumps(output_arr, indent=4)
    with open(f"backawrd_generation_samples_uncalibrated.json", "w") as json_file:
        json_file.write(json_string)

    '''
    TOTAL_ROUND = 9
    PER_FILE = 960
    START_POINT = PER_FILE * TOTAL_ROUND * 3

    for round in tqdm(range(7)):
        output_arr = []
        for idx in tqdm(range(PER_FILE) if round != 6 else range(len(data) -START_POINT - 6 * PER_FILE)):
            input_tokens = process_data(data[START_POINT + idx + round * PER_FILE]['instruction'], data[START_POINT + idx + round * PER_FILE]['output'], encoder)
            output_dict = get_generation_with_temp_one_and_half_k_thirty(input_tokens, model_finetuned_forward, encoder)
            output_arr.append(output_dict)
        json_string = json.dumps(output_arr, indent=4)

        # Write the JSON string to a file
        with open(f"forward_generation_round_{round +1}.json", "w") as json_file:
            json_file.write(json_string)
    '''