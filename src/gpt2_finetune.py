import torch, json
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder
from transformers import Trainer, TrainingArguments, GPT2Tokenizer

class OpenGPT2Dataset(Dataset):
    
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]

        data = {"input": self.tokenizer(data["instances"][0]["instruction_with_input"], 
                                          padding="max_length", truncation=True, max_length=1024, 
                                          return_tensors="pt"),
                "output": self.tokenizer(data["instances"][0]["output"], 
                                         padding="max_length", truncation=True, max_length=1024, 
                                         return_tensors="pt")}
        
        return {"input_ids": data["input"]["input_ids"].to(self.device),
                "attention_mask": data["input"]["attention_mask"].to(self.device),
                "labels": data["output"]["input_ids"].to(self.device),
                "output_attentions": data["output"]["attention_mask"].to(self.device)}

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


def load_models(device=torch.device("cpu")) -> Tuple[torch.nn.Module, Encoder]:
    # PATH_TO_FORWARD = "/home/cs601-zxia15/NLP_final_project/params/opengpt2_pytorch_forward"
    PATH_TO_FORWARD = "/home/cs601-zxia15/NLP_final_project/params/opengpt2_pytorch_forward"
    # model_forward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FORWARD).to(device)
    model_forward = OpenGPT2LMHeadModel.from_pretrained(PATH_TO_FORWARD).to(device)
    encoder = Encoder()
    return model_forward, encoder

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_datasets(tokenizer, batch_size, device=torch.device("cpu"), is_core=True):
    
    PATH_TO_CORE_DATASET = "/home/cs601-zxia15/NLP_final_project/unnatural-instructions/data/core_data.jsonl" 
    PATH_TO_FULL_DATASET = "/home/cs601-zxia15/NLP_final_project/unnatural-instructions/data/full_data.jsonl" 

    data_path = PATH_TO_CORE_DATASET if is_core else PATH_TO_FULL_DATASET
    dataset = OpenGPT2Dataset(read_jsonl(data_path), tokenizer, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
    return dataloader

def fine_tune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("loading models and tokenizers!")
    model, _ = load_models(device=device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define training arguments
    print("set training args")
    training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    output_dir = "/home/cs601-zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_forward"
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
    model.save_pretrained("/home/cs601-zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_forward_02")

if __name__ == "__main__":
    fine_tune_model()


