import torch, argparse, subprocess, time
from transformers import get_scheduler
from gpt2_finetune import load_datasets, load_models
from load_model import get_forward_backward_preds

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

def backward_train(mymodel, num_epochs, device, lr, encoder, train_dataloader, val_dataloader, test_dataloader):
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
    all_train_loss_list = []
    all_train_ppl_list = []
    all_val_loss_list = []
    all_val_ppl_list = []
    train_ppl_list = []
    val_loss_list = []
    val_ppl_list = []

    prev_loss = 100000


    for epoch in range(num_epochs):

        epoch_start_time = time.time()

        cur_epoch_train_loss = []
        cur_epoch_val_loss = []
        cur_epoch_train_ppl = []
        cur_epoch_val_ppl = []


        print(f"Epoch {epoch + 1} training:")

        mymodel.train()

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

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)


            predictions = mymodel(input_ids, attention_mask, labels)

            if index == 0:
                non_input_length = attention_mask.size()[0] - torch.sum(attention_mask)
                input_output_text = encoder.decode(torch.flip(input_ids[0], [0]))
                pred_in_text = encoder.decode(torch.flip(torch.argmax(predictions['logits'][0], dim=1), [0]))
                mod_labels = torch.where(labels[0] == -100, torch.tensor(0), labels[0])
                expected_pred = encoder.decode(torch.flip(mod_labels, [0]))
                print(f"current input : {input_output_text}\n")
                print(f"predicted text : {pred_in_text}\n")
                print(f"expected prediction : {expected_pred}\n")
            
            loss = predictions['loss']
            cur_epoch_train_ppl.append(torch.exp(loss).item())
            all_train_ppl_list.append(torch.exp(loss).item())
            all_train_loss_list.append(loss.item())
            cur_epoch_train_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            
            optimizer.zero_grad()
        
            lr_scheduler.step()
        # print loss metrics
        avg_loss =  np.array(cur_epoch_train_loss).sum() / len(cur_epoch_train_loss)
        avg_ppl = np.array(cur_epoch_train_ppl).sum() / len(cur_epoch_train_loss)
        print(f" ===> Train Epoch {epoch + 1}")
        print(f" - Average Train loss metrics: {avg_loss}")
        print(f" - Average Train perplexity metrics: {avg_ppl}")
        train_loss_list.append(avg_loss)
        train_ppl_list.append(avg_ppl)

        print(f"Epoch {epoch + 1} validation:")
        
        mymodel.eval()

        for index, batch in tqdm(enumerate(val_dataloader)):

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

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)


            predictions = mymodel(input_ids, attention_mask, labels)

            if index == 0:
                non_input_length = attention_mask.size()[0] - torch.sum(attention_mask)
                input_output_text = encoder.decode(torch.flip(input_ids[0], [0]))
                pred_in_text = encoder.decode(torch.flip(torch.argmax(predictions['logits'][0], dim=1), [0]))
                mod_labels = torch.where(labels[0] == -100, torch.tensor(0), labels[0])
                expected_pred = encoder.decode(torch.flip(mod_labels, [0]))
                print(f"current input : {input_output_text}\n")
                print(f"predicted text : {pred_in_text}\n")
                print(f"expected prediction : {expected_pred}\n")
            
            loss = predictions['loss']
            all_val_loss_list.append(loss.item())
            all_val_ppl_list.append(torch.exp(loss).item())
            cur_epoch_val_ppl.append(torch.exp(loss).item())
            cur_epoch_val_loss.append(loss.item())
        
        # print loss metrics
        avg_val_loss =  np.array(cur_epoch_val_loss).sum() / len(cur_epoch_val_loss)
        avg_val_ppl =  np.array(cur_epoch_val_ppl).sum() / len(cur_epoch_val_loss)
        print(f" ===> Val Epoch {epoch + 1}")
        print(f" - Average Validation loss metrics: {avg_val_loss}")
        print(f" - Average Validation perplexity metrics: {avg_val_ppl}")
        val_loss_list.append(avg_val_loss)
        val_ppl_list.append(avg_val_ppl)
        
        epoch_list.append(epoch)

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

        if (epoch == 0) or (epoch != 0 and prev_loss > avg_val_loss):
            print(f"save model for epoch {epoch + 1}!")
            model = mymodel.get_model()
            model.save_pretrained("/home/mjia8/NLP_final_project/params/fine_tuned_opengpt2_model_backward_trial_alpaca_alldrop0.15")
            prev_loss = avg_val_loss
    
    mymodel.eval()
    cur_test_ppl = []
    cur_test_loss = []

    for index, batch in tqdm(enumerate(test_dataloader)):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        predictions = mymodel(input_ids, attention_mask, labels)

        if index == 0:
            non_input_length = attention_mask.size()[0] - torch.sum(attention_mask)
            input_output_text = encoder.decode(torch.flip(input_ids[0], [0]))
            pred_in_text = encoder.decode(torch.flip(torch.argmax(predictions['logits'][0], dim=1), [0]))
            mod_labels = torch.where(labels[0] == -100, torch.tensor(0), labels[0])
            expected_pred = encoder.decode(torch.flip(mod_labels, [0]))
            print(f"current input : {input_output_text}\n")
            print(f"predicted text : {pred_in_text}\n")
            print(f"expected prediction : {expected_pred}\n")
            
        loss = predictions['loss']
        cur_test_ppl.append(torch.exp(loss).item())
        cur_test_loss.append(loss.item())
        
        # print loss metrics
    test_ppl =  np.array(cur_test_ppl).sum() / len(cur_test_ppl)
    test_loss =  np.array(cur_test_loss).sum() / len(cur_test_loss)
    print(f" ===> TESTING!!")
    print(f" - Dead loss metrics: {test_loss}")
    print(f" - Dead perplexity metrics: {test_ppl}")

    plot(train_loss_list, val_loss_list, 'Train vs Validation Loss Graph Alpaca per Epoch', False)
    plot(train_ppl_list, val_ppl_list, 'Train vs Validation Perpexity Graph Alpaca per Epoch', True)
    plot(all_train_loss_list, all_val_loss_list, 'Train vs Validation Loss Graph Alpaca per Batch', False, False)
    plot(all_train_ppl_list, all_val_ppl_list, 'Train vs Validation Perpexity Graph Alpaca per Batch', True, False)

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
    model, encoder = load_models(device=device, is_backward=is_backward)
    pretrained_model = CustomModelforSequenceGeneration(model=model, device=device)

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader, val_dataloader, test_dataloader = load_datasets(is_core=small_subset, encoder=encoder, 
                                                                      is_backward=is_backward, batch_size=batch_size,
                                                                        max_length=max_length, device=device)


    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, encoder, train_dataloader, val_dataloader, test_dataloader

# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset",action='store_true', default=False)
    parser.add_argument("--is_backward",action='store_true', default=False)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=100)

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    # global prefix_length
    # prefix_length = args.prefix_length
    #load the data and models
    pretrained_model, encoder, train_dataloader, val_dataloader, test_dataloader = pre_process(
        device, args.small_subset, args.is_backward,
        args.max_length, args.batch_size)
    print(" >>>>>>>>  Starting training ... ")
    backward_train(pretrained_model, args.num_epochs,
          device, args.lr, encoder, 
          train_dataloader, val_dataloader, test_dataloader)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()