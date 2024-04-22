import torch, json, random
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder
from transformers import Trainer, TrainingArguments, GPT2Tokenizer


def process_backward_data(io_text, instruction, encoder, max_length):
    encoded_io_text = torch.flip(torch.tensor(encoder.encode(io_text)), [0])
    instruction_text = torch.flip(torch.tensor(encoder.encode(instruction + " ")), [0])
    encoded_attention_mask = torch.ones(max_length)
    io_encoded_length = len(encoded_io_text)
    instruction_encoded_length = len(instruction_text)
     # ...(padded / truncated)... x x x x (prompt) x x x x x x (ouput) ...(padded / truncated)...

    # step for truncation
    if (io_encoded_length + instruction_encoded_length > max_length):
        truncated_length = io_encoded_length + instruction_encoded_length - max_length
        io_truncated_length = truncated_length / 2
        instruction_truncated_length = truncated_length - io_truncated_length
        full_encoded_text = torch.cat([encoded_io_text[io_truncated_length:], instruction_text[:instruction_encoded_length-instruction_truncated_length]])
        instruction_encoded_text = torch.cat([torch.ones(io_encoded_length - io_truncated_length - 1) * -100, instruction_text[:instruction_encoded_length-instruction_truncated_length], torch.tensor([-100])])
    else:
        padded_length = max_length - io_encoded_length - instruction_encoded_length
        full_encoded_text = torch.cat([encoded_io_text, instruction_text, torch.zeros(padded_length)])
        instruction_encoded_text = torch.cat([torch.ones(io_encoded_length - 1) * -100, instruction_text, torch.ones(padded_length + 1) * -100])
        encoded_attention_mask = torch.cat([torch.ones(io_encoded_length + instruction_encoded_length), torch.zeros(padded_length)])


    assert(len(full_encoded_text) == max_length)
    assert(len(instruction_encoded_text) == max_length)
    assert(len(encoded_attention_mask) == max_length)

    return {"input_ids": full_encoded_text.to(torch.int64), "attention_mask": encoded_attention_mask.to(torch.int64),
            "labels": instruction_encoded_text}
    

class OpenGPT2Dataset(Dataset):
    
    def __init__(self, dataset, device, encoder, is_backward, max_length, random_seed=42):
        self.dataset = dataset
        self.dataset_length = len(dataset)
        self.device = device
        self.encoder = encoder
        self.max_length = max_length
        self.is_backward = is_backward
        random.seed(random_seed)
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        data = self.dataset[idx]

        if self.is_backward:
            return process_backward_data("Provided: " + data["instances"][0]["input"] + "; Result: " + data["instances"][0]["output"],
                                         data['instruction'], self.encoder, max_length=self.max_length)
        else:
            raise Exception("Unimplemented forward fine tuning. Please ask Jiefu.")


class OpenGPT2Trainer(Trainer):
    def __init__(self, *args, device=torch.device("cpu"), is_core=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.is_core = is_core

    def get_train_dataloader(self):
        return load_datasets(tokenizer=self.tokenizer, 
                             device=self.device,
                             is_core= self.is_core,
                             batch_size=self.args.train_batch_size)


def load_models(device=torch.device("cpu"), is_backward=False) -> Tuple[torch.nn.Module, Encoder]:
    # PATH_TO_FORWARD = "/home/cs601-zxia15/NLP_final_project/params/opengpt2_pytorch_forward"
    PATH_TO_FORWARD = "/home/zxia15/NLP_final_project/params/opengpt2_pytorch_forward"
    PATH_TO_BACKWARD = "/home/zxia15/NLP_final_project/params/opengpt2_pytorch_backward"
    # model_forward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FORWARD).to(device)
    model = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_BACKWARD).to(device) if is_backward else OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FORWARD).to(device)
    encoder = Encoder()
    return model, encoder

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_datasets(is_core, encoder, is_backward, batch_size,
                  max_length, device=torch.device("cpu"), train_val_ratios=[0.8, 0.1]):
    
    PATH_TO_CORE_DATASET = "/home/zxia15/NLP_final_project/data/unnatural-instructions/core_data.jsonl" 
    PATH_TO_FULL_DATASET = "/home/zxia15/NLP_final_project/data/unnatural-instructions/full_data.jsonl" 

    data_path = PATH_TO_CORE_DATASET if is_core else PATH_TO_FULL_DATASET
    data = read_jsonl(data_path)
    train_len, val_len = len(data) * train_val_ratios[0], len(data) * train_val_ratios[1]

    train_dataset = OpenGPT2Dataset(data[:train_len], device, encoder=encoder, is_backward=is_backward,
                                    max_length=max_length)
    val_dataset = OpenGPT2Dataset(data[train_len:train_len + val_len], device, encoder=encoder, is_backward=is_backward,
                                    max_length=max_length)
    test_dataset = OpenGPT2Dataset(data[train_len + val_len:], device, encoder=encoder, is_backward=is_backward,
                                    max_length=max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    return train_dataloader, val_dataloader, test_dataloader

def fine_tune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("loading models and tokenizers!")
    model, _ = load_models(device=device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define training arguments
    print("set training args")
    training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_dir='./logs',
    output_dir = "/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_forward"
    )

    # Initialize Trainer
    print("trainer!")
    trainer = OpenGPT2Trainer(model=model, tokenizer=tokenizer, 
                              args=training_args, device=device, is_core=True)

    # Fine-tune the model
    print("trainer starts training")
    trainer.train()

    # Save the fine-tuned model
    print("save model!")
    model.save_pretrained("/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_forward_02")

if __name__ == "__main__":
    fine_tune_model()


