import torch
import torch.optim as optim

from data_prep.data_initializer import DDInitializer
from ddpm.vector_combination.combiner_unet import VectorCombinationUNet  # Importing your specific model
from ddpm.vector_combination.combination_loss import PhysicsInformedLoss


def combine_fields(known, inpainted, mask):

    #Get Config Data
    dd = DDInitializer()

    #Calculate initial data
    naive = known * (1 - mask) + (inpainted * mask)

    #Get Combination Type
    if dd.get_use_comb_net():
        with torch.enable_grad():
            return train_comb_net(dd, naive, known, inpainted, mask)
    else:
        return naive


def train_comb_net(dd, naive, known, inpainted, mask):

    #Configure environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    naive.to(device)

    #Combine inputs
    combined_input = torch.cat([naive, mask], dim=1)

    # Set up model and optimizer
    unet = VectorCombinationUNet(n_channels=4, n_classes=2).to(device)
    unet.train()
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)

    # Initialize your Loss Function
    physics_informed_loss = PhysicsInformedLoss(dd.get_attribute("fidelity_weight"),
                                                dd.get_attribute("physics_weight"),
                                                dd.get_attribute("smooth_weight"))

    # training loop

    num_steps = dd.get_attribute("comb_training_steps")

    for i in range(num_steps):
        # 1. Zero Gradients: Clear the "ledger" from the previous step
        optimizer.zero_grad()

        # 2. Forward Pass: Run data through the model
        prediction = unet(combined_input)

        # 3. Calculate Loss: Check physics constraints
        loss, stats = physics_informed_loss(prediction, known, inpainted, mask)

        # 4. Backward Pass: Calculate how to adjust weights
        loss.backward()

        # 5. Optimizer Step: Apply the adjustments
        optimizer.step()

    # Extract and return data
    final_prediction = prediction.detach()
    return final_prediction
