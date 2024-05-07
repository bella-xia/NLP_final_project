import torch, argparse, subprocess, time
from transformers import get_scheduler
from gpt2_finetune import load_datasets, load_models
from load_model import get_forward_backward_preds
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel

import evaluate as evaluate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomModelforSequenceGeneration(torch.nn.Module):

    def __init__(self, model, device):
        super(CustomModelforSequenceGeneration, self).__init__()
        self.device = device
        self.gpt2_model = model

    def forward(self, input_ids, attention_mask, labels, position_reverse):
        
        output = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                position_reverse=position_reverse)

        return output
    
    def get_model(self):
        return self.gpt2_model

    def change_model(self, path):
        del self.gpt2_model
        self.gpt2_model = OpenGPT2LMHeadModel.from_pretrained(path).to(self.device) 
    
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

    raise Exception("Unimplemented. If you get there you are DOOMED")

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

def trainer(mymodel, num_epochs, device, lr, encoder, train_forward_dataloader, train_backward_dataloader, val_forward_dataloader, val_backward_dataloader, test_dataloader, is_backward, position_reverse):
    
    weight_decay = 0.01
    no_decay = ['bias', 'LayerNorm.weight']

    # loss_fn = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_loss_list = []
    train_ppl_list = []
    val_loss_list = []
    val_ppl_list = []
    all_train_loss_list = []
    all_train_ppl_list = []
    all_val_loss_list = []
    all_val_ppl_list = []

    prev_loss = 100000

    with torch.cuda.device(0):
        mymodel.to(device)
        optimizer_grouped_parameters = [
        {'params': [p for n, p in mymodel.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
        {'params': [p for n, p in mymodel.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
        ]
        forward_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        forward_lr_scheduler = get_scheduler(
            "linear",
            optimizer=forward_optimizer,
            num_warmup_steps=50,
            num_training_steps=(len(train_forward_dataloader) + len(train_backward_dataloader))* num_epochs
            )
        for epoch in range(num_epochs):

            epoch_start_time = time.time()

            mymodel.train()
            trainer_helper(mymodel, train_forward_dataloader, forward_optimizer, forward_lr_scheduler,
                        all_train_loss_list, all_train_ppl_list, train_loss_list, train_ppl_list,
                        epoch, is_train=True, is_backward=False, position_reverse=False)
        
            mymodel.eval()
            avg_val_loss = trainer_helper(mymodel, val_forward_dataloader, None, None,
                        all_val_loss_list, all_val_ppl_list, val_loss_list, val_ppl_list,
                        epoch, is_train=False, is_backward=False, position_reverse=False)

            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

            if (epoch == 0) or (epoch != 0 and prev_loss > avg_val_loss):
                print(f"save model for epoch {epoch + 1}!")
                model = mymodel.get_model()
                model.save_pretrained("/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_backward_alcapa")
                prev_loss = avg_val_loss
    
    torch.cuda.empty_cache()
    with torch.cuda.device(0):
        mymodel.change_model("/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_backward_alcapa")
        mymodel.to(device)

        backward_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        
        backward_lr_scheduler = get_scheduler("linear",
                                optimizer=backward_optimizer, num_warmup_steps=50,
                                num_training_steps=len(train_backward_dataloader)* num_epochs)

        for epoch in range(num_epochs):

            epoch_start_time = time.time()
            mymodel.train()
            trainer_helper(mymodel, train_backward_dataloader, backward_optimizer, backward_lr_scheduler,
                        all_train_loss_list, all_train_ppl_list, train_loss_list, train_ppl_list,
                        epoch, is_train=True, is_backward=True, position_reverse=position_reverse)
        
            mymodel.eval()
            avg_val_loss = trainer_helper(mymodel, val_backward_dataloader, None, None,
                        all_val_loss_list, all_val_ppl_list, val_loss_list, val_ppl_list,
                        epoch, is_train=False, is_backward=True, position_reverse=position_reverse)

            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

            if (epoch == 0) or (epoch != 0 and prev_loss > avg_val_loss):
                print(f"save model for epoch {epoch + 1}!")
                model = mymodel.get_model()
                model.save_pretrained("/home/zxia15/NLP_final_project/params/fine_tuned_opengpt2_model_backward_alcapa")
                prev_loss = avg_val_loss

    torch.cuda.empty_cache()
    with torch.cuda.device(0):
        mymodel.to(device)
        mymodel.eval()
        trainer_helper(mymodel, test_dataloader, None, None, [], [], [], [], 10082, 
        is_train=False, is_backward=True, position_reverse=position_reverse)
    
    torch.cuda.empty_cache()

    plot(train_loss_list, val_loss_list, 'Train vs Validation Loss Graph Alpaca per Epoch For And Back', False)
    plot(train_ppl_list, val_ppl_list, 'Train vs Validation Perpexity Graph Alpaca per Epoch For And Back', True)
    plot(all_train_loss_list, all_val_loss_list, 'Train vs Validation Loss Graph Alpaca per Batch For And Back', False, False)
    plot(all_train_ppl_list, all_val_ppl_list, 'Train vs Validation Perpexity Graph Alpaca per Batch For And Back', True, False)

def trainer_helper(mymodel, dataloader, optimizer, lr_scheduler, all_loss_list, all_ppl_list,
                   loss_list, ppl_list, epoch, is_train, is_backward, position_reverse):

    cur_epoch_ppl = []
    cur_epoch_loss = []
    print(f" ===> Begin training Epoch {epoch + 1}" if is_train else f" ===> Begin validation Epoch {epoch + 1}")

    for index, batch in tqdm(enumerate(dataloader)):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        predictions = mymodel(input_ids, attention_mask, labels, position_reverse=position_reverse)

        if index == 0:
                if is_backward:
                    print(f"\n---> evaluating backward train instance" if is_train else f"\n---> evaluating backward validation instance" )
                    input_text = encoder.decode(torch.flip(input_ids[0], [0]))
                    pred_text = encoder.decode(torch.flip(torch.argmax(predictions['logits'][0], dim=1), [0]))
                    mod_labels = torch.where(labels[0] == -100, torch.tensor(0), labels[0])
                    expected_pred = encoder.decode(torch.flip(mod_labels, [0]))
                else:
                    print(f"\n---> evaluating forward train instance" if is_train else f"\n---> evaluating forward validation instance")
                    input_text = encoder.decode(input_ids[0])
                    pred_text = encoder.decode(torch.argmax(predictions['logits'][0], dim=1))
                    mod_labels = torch.where(labels[0] == -100, torch.tensor(0), labels[0])
                    expected_pred = encoder.decode(mod_labels)
                print(f"current input : {input_text}\n")
                print(f"predicted text : {pred_text}\n")
                print(f"expected prediction : {expected_pred}\n")
            
        loss = predictions['loss']
        cur_epoch_ppl.append(torch.exp(loss).item())
        cur_epoch_loss.append(loss.item())
        all_loss_list.append(loss.item())
        all_ppl_list.append(torch.exp(loss).item())

        if is_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        # print loss metrics
    avg_loss =  np.array(cur_epoch_loss).sum() / len(cur_epoch_loss)
    avg_ppl = np.array(cur_epoch_ppl).sum() / len(cur_epoch_loss)
    print(f" ===> Train Epoch {epoch + 1}" if is_train else f" ===> Val Epoch {epoch + 1}")
    print(f" - Average Train loss metrics: {avg_loss}" if is_train else f" - Average Val loss metrics: {avg_loss}")
    print(f" - Average Train perplexity metrics: {avg_ppl}" if is_train else f" - Average Val perplexity metrics: {avg_ppl}")
    loss_list.append(avg_loss)
    ppl_list.append(avg_ppl)
    return avg_loss

def plot(train_list, valid_list, name, is_ppl, same_length=True):
    
    plt.figure()
    if same_length:
        plt.plot(train_list, label='Train')
        plt.plot(valid_list, label='Validation')
    else:
        train_x = range(len(train_list))
        valid_x = np.linspace(0, len(train_list)-1, num=len(valid_list))  # Correctly spaced x values for validation data
        plt.plot(train_x, train_list, label='Train')
        plt.plot(valid_x, valid_list, label='Validation')
        
    plt.xlabel('Epochs' if same_length else 'Batches')
    plt.ylabel('Perplexity' if is_ppl else 'Loss')
    plt.title('Train vs Validation')
    plt.legend()
    plt.savefig(f'{name}.png')


def pre_process(device, small_subset, is_backward, max_length, batch_size):

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    model, encoder = load_models(device=device, is_backward=False)
    pretrained_model = CustomModelforSequenceGeneration(model=model, device=device)

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_forward_dataloader, train_backward_dataloader, val_forward_dataloader, val_backward_dataloader, test_dataloader = load_datasets(is_core=small_subset, encoder=encoder, 
                                                                      is_backward=is_backward, batch_size=batch_size,
                                                                        max_length=max_length, device=device)


    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, encoder, train_forward_dataloader, train_backward_dataloader, val_forward_dataloader, val_backward_dataloader, test_dataloader

# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset",action='store_true', default=False)
    parser.add_argument("--is_backward",action='store_true', default=False)
    parser.add_argument("--position_reverse", action='store_true', default=False)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=100)

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    # global prefix_length
    # prefix_length = args.prefix_length
    #load the data and models
    pretrained_model, encoder, train_forward_dataloader, train_backward_dataloader, val_forward_dataloader, val_backward_dataloader, test_dataloader = pre_process(
        device, args.small_subset, args.is_backward,
        args.max_length, args.batch_size)
    print(" >>>>>>>>  Starting training ... ")
    trainer(pretrained_model, args.num_epochs,
          device, args.lr, encoder, 
          train_forward_dataloader, train_backward_dataloader, val_forward_dataloader, val_backward_dataloader, test_dataloader, args.is_backward, args.position_reverse)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()