import os
import re
import time

import psutil
import torch


def train_model(model,
                loader,
                criterion,
                optimizer,
                device,
                curr_epoch,
                loss_history,
                train_for=0,
                verbose=True,
                checkpoint_every=0,
                print_cuda_mem=False):
    """Trains model, prints cuda mem, saves checkpoints, resumes training.

    Args:
        device : torch.device()
        curr_epoch : Current epoch the model is. Zero if it was never trained
        loss_history ([list]): Empty list if the model was never trained
        train_for (int, optional): Number of epochs to train. Defaults to 0.
        verbose (bool, optional): Print epoch counter and time of each epoch. Defaults to True.
        checkpoint_every (int, optional): Save checkpoint every "checkpoint_every" epochs. Defaults to 0.
        print_cuda_mem (bool, optional): Defaults to False.
    """

    last_epoch = train_for + curr_epoch

    if curr_epoch > 1:
        curr_epoch += 1  # If curr_epoch not zero, the argument passed to
        # curr_epoch is the last epoch the
        # model was trained in the loop before

    if print_cuda_mem:
        print_cuda_memory()

    while True:
        if verbose:
            print("EPOCH:", curr_epoch,  end=' ')
        start = time.time()
        for (data, targets) in loader:

            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        if print_cuda_mem:
            print()
            print_cuda_memory()
            print_cuda_mem = False

        # Time
        end = time.time()
        if verbose:
            print(f'Time elapsed: {(end - start):.2f} secs.')

        loss_history.append(loss.clone().detach().cpu().numpy())

        PATH = "checkpoints/model_epoch" + str(curr_epoch) + ".pt"

        if checkpoint_every:
            if curr_epoch % checkpoint_every == 0:
                torch.save({
                    'epoch': curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history,
                }, PATH)

        if train_for:
            if curr_epoch == last_epoch:
                return loss_history

        curr_epoch += 1


def print_cuda_memory():
    print('Memory Usage:')
    print(f'\t Allocated: {(torch.cuda.memory_allocated(0)/1024**2):.5f} MB.')
    print(f'\t Cached: {(torch.cuda.memory_reserved(0)/1024**2):.5f} MB.')
    return


def print_proc_mem():
    process = psutil.Process(os.getpid())
    print(f'{process.memory_info().rss} Byets.')  # in bytes
    return


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

def get_last_checkpoint(path):
    checkpoints = os.listdir(path)
    epochs = [int(epoch) for cp in checkpoints for epoch in re.findall(r'\d+', cp) ]
    last_epoch = max(epochs)
    return last_epoch
    