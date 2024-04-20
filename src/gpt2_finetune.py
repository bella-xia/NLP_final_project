import torch, json, random
from torch.utils.data import Dataset
from typing import Tuple
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder
from transformers import Trainer, TrainingArguments, GPT2Tokenizer


def process_data(text, encoder, max_length, is_input=True, is_backward=False):
    encoded_text = torch.tensor(encoder.encode(text))
    encoded_attention_mask = torch.ones(max_length)
    encoded_length = len(encoded_text)
     # ...(padded / truncated)... x x x x (prompt) x x x x x x (ouput) ...(padded / truncated)...

    # step for truncation
    if is_input and (encoded_length > max_length):
        encoded_text = encoded_text[encoded_length - max_length:]
    elif (not is_input) and (encoded_length > max_length):
        encoded_text = encoded_text[:max_length]
    # step for padding
    elif is_input and (encoded_length < max_length):
        encoded_text = torch.cat((torch.zeros(max_length - encoded_length), encoded_text))
        encoded_attention_mask = torch.cat((torch.zeros(max_length - encoded_length), torch.ones(encoded_length)))
    elif (not is_input) and (encoded_length < max_length):
        encoded_text = torch.cat((encoded_text, torch.zeros(max_length - encoded_length)))
        encoded_attention_mask = torch.cat((torch.ones(encoded_length), torch.zeros(max_length - encoded_length)))
    
    assert(len(encoded_text) == max_length)

    if is_backward:
        encoded_text = torch.flip(encoded_text, [0])
        encoded_attention_mask = torch.flip(encoded_attention_mask, [0])

    return {"input_ids": encoded_text.to(torch.int64), "attention_mask": encoded_attention_mask.to(torch.int64)}
    

class OpenGPT2Dataset(Dataset):
    
    def __init__(self, dataset, device, encoder, is_backward, input_size, output_size, random_seed=42):
        self.dataset = dataset
        self.dataset_length = len(dataset)
        self.device = device
        self.encoder = encoder
        self.input_output_size, self.instruction_size = (input_size, output_size) if is_backward else (output_size, input_size) 
        self.is_backward = is_backward
        random.seed(random_seed)
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        data = self.dataset[idx]

        instruction_is_input = False if self.is_backward else True
        
        instruction_data = process_data(data['instruction'], 
                                        self.encoder, max_length=self.instruction_size, 
                                        is_input= instruction_is_input, is_backward=self.is_backward)
        output_and_input_data = process_data("Provided: " + data["instances"][0]["input"] + "; Result: " + data["instances"][0]["output"], 
                                             self.encoder, max_length=self.input_output_size, 
                                             is_input=~instruction_is_input, is_backward=self.is_backward)
        
        data = {"input": output_and_input_data, "output": instruction_data} if self.is_backward else {"input": instruction_data,"output": output_and_input_data}
        
        return {"input_ids": data["input"]["input_ids"].to(self.device),
                "attention_mask": data["input"]["attention_mask"].to(self.device),
                "labels": data["output"]["input_ids"].to(self.device),
               }

    def get_random_instance(self):
        random_idx = random.randrange(0, self.dataset_length)
        return ("Provided: " + self.dataset[random_idx]["instances"][0]["input"] + "; Result: " + self.dataset[random_idx]["instances"][0]["output"]) if self.is_backward else (self.dataset[random_idx]['instruction'])


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


def load_datasets(is_core, encoder, is_backward,
                  input_size, output_size, device=torch.device("cpu")):
    
    PATH_TO_CORE_DATASET = "/home/zxia15/NLP_final_project/data/unnatural-instructions/core_data.jsonl" 
    PATH_TO_FULL_DATASET = "/home/zxia15/NLP_final_project/data/unnatural-instructions/full_data.jsonl" 

    data_path = PATH_TO_CORE_DATASET if is_core else PATH_TO_FULL_DATASET
    dataset = OpenGPT2Dataset(read_jsonl(data_path), device, encoder=encoder, is_backward=is_backward,
                              input_size=input_size, output_size=output_size)
 
    return dataset

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


