import torch, argparse, subprocess, time
from transformers import get_scheduler
from gpt2_finetune import load_datasets, load_models
from load_model import get_forward_backward_preds
from torch.utils.data import DataLoader

import evaluate as evaluate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomModelforSequenceGeneration(torch.nn.Module):

    def __init__(self, model, device):
        super(CustomModelforSequenceGeneration, self).__init__()
        self.device = device
        self.gpt2_model = model

    def forward(self, input_ids, attention_mask, labels):
        
        output = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return output
    
    def get_model(self):
        return self.gpt2_model
    
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


def evaluate_model(model, encoder, dataset, is_backward, device):
    """
    Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    # dev_accuracy = evaluate.load('accuracy')
    input_text = dataset.get_random_instance()
    print("prompt give: ")
    print(input_text)
    is_forward = ~is_backward
    get_forward_backward_preds(model, encoder, input_text, is_forward, device)
    return

    raise "Unimplemented"

    # turn model into evaluation mode
    model.eval()

    # iterate over the dataloader
    for batch in dataloader:
        # TODO: implement the evaluation function
        # raise NotImplementedError("You need to implement the evaluation function")
        # get the input_ids, attention_mask from the batch and put them on the device
        # Hints:
        # - see the getitem function in the BoolQADataset class for how to access the input_ids and attention_mask
        # - use to() to move the tensors to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        

        # forward pass
        # name the output as `output`
        output = model(input_ids, attention_mask)

        # your code ends here

        predictions = output['labels']
        # predictions = torch.argmax(predictions, dim=1)
        # dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    # return dev_accuracy.compute()

def train(mymodel, num_epochs, batch_size, train_dataset, device, lr, encoder, is_backward):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :param string model_name: the name of the model
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.AdamW
    print(" >>>>>>>>  Loading dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(" >>>>>>>>  Initializing optimizer")
    
    weight_decay = 0.01
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in mymodel.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
        {'params': [p for n, p in mymodel.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    # loss_fn = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_loss_list = []

    prev_loss = 100000


    for epoch in range(num_epochs):

        epoch_start_time = time.time()

        cur_epoch_train_loss = []


        print(f"Epoch {epoch + 1} training:")

        for index, batch in tqdm(enumerate(train_dataloader)):

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, depending on model.type, you may want to use different optimizers
            Then, call loss.backward() to compute the gradients.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.step()  to update the model parameters.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """

            # TODO: implement the training loop
            # raise NotImplementedError("You need to implement this function")
            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            # Hints: similar to the evaluate_model function
  
           
            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            # Hints: similar to the evaluate_model function

            mymodel.train()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)


            predictions = mymodel(input_ids, attention_mask, labels)
            
            loss = predictions['loss']
            cur_epoch_train_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            
            optimizer.zero_grad()

            mymodel.eval()

            if (index % 500 == 0):
                evaluate_model(mymodel.get_model(), encoder, train_dataset, is_backward, device)
        
        # print loss metrics
        avg_loss =  np.array(cur_epoch_train_loss).sum() / len(cur_epoch_train_loss)
        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average loss metrics: {avg_loss}")
        train_loss_list.append(avg_loss)
        
        epoch_list.append(epoch)

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

        lr_scheduler.step()

        if (epoch == 0) or (epoch != 0 and prev_loss > avg_loss):
            print(f"save model for epoch {epoch + 1}!")
            model = mymodel.get_model()
            model.save_pretrained("/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_backward")
            prev_loss = avg_loss

def plot(train_list, valid_list, name, finetune_method):
    
    plt.figure()
    plt.plot(train_list, label='Train')
    plt.plot(valid_list, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.savefig(f'{name}_{finetune_method}.png')


def pre_process(device, small_subset, is_backward, input_size, output_size):

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    model, encoder = load_models(device=device, is_backward=is_backward)
    pretrained_model = CustomModelforSequenceGeneration(model=model, device=device)

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataset = load_datasets(is_core=small_subset, encoder=encoder, 
                                     is_backward=is_backward,
                                     input_size=input_size, output_size=output_size,
                                     device=device)


    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataset, encoder

# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset",action='store_true', default=True)
    parser.add_argument("--is_backward",action='store_true', default=False)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=30)
    parser.add_argument("--output_size", type=int, default=30)

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    # global prefix_length
    # prefix_length = args.prefix_length
    #load the data and models
    pretrained_model, train_dataset, encoder = pre_process(device, args.small_subset, args.is_backward,
                                                    args.input_size, args.output_size)
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, args.batch_size, train_dataset,
          device, args.lr, encoder, args.is_backward)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()