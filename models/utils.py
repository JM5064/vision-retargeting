import torch
import os


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)


def log_results(file_path, metrics):
    # Open file
    file = open(file_path, "a")

    # Create header if file is blank
    if os.path.getsize(file_path) == 0:
        for metric in metrics:
            file.write(f'{metric},')
        
        file.write('\n')

    # Log metrics
    for metric in metrics:
            file.write(f'{metrics[metric]},')

    file.write('\n')
    # Makes file update immediately
    file.flush()
    os.fsync(file.fileno())


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch']
    

def freihand_to_allegro(i):
    mapping = {
        0 : 0,      # Wrist -> base link
        4 : 20,     # Thumb tip -> link_15.0_tip
        8 : 5,      # Index tip -> link_3.0_tip
        12 : 10,    # Middle tip -> link_7.0_tip
        16 : 15,    # Ring tip -> link_11.0_tip
    }

    return mapping[i]
