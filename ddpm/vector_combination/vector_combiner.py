import torch
import torch.optim as optim
from pathlib import Path

from plots.visualization_tools.plot_vector_field_tool import plot_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence
import logging
from pandas import DataFrame

from data_prep.data_initializer import DDInitializer
from ddpm.vector_combination.combiner_unet import VectorCombinationUNet  # Importing your specific model
from ddpm.vector_combination.combination_loss import PhysicsInformedLoss


def combine_fields(known, inpainted, mask, save_dir=""):

    #Get Config Data
    dd = DDInitializer()

    #Calculate initial data
    naive = known * (1 - mask) + (inpainted * mask)

    #Get Combination Type
    if dd.get_use_comb_net():
        with torch.enable_grad():
            return train_comb_net(dd, naive, known, inpainted, mask, save_dir=save_dir)
    else:
        return naive


def train_comb_net(config_data: DDInitializer,
                   naive: torch.Tensor, known: torch.Tensor, inpainted: torch.Tensor, mask : torch.Tensor,
                   save_dir="") -> torch.Tensor:

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
    physics_informed_loss = PhysicsInformedLoss(config_data.get_attribute("fidelity_weight"),
                                                config_data.get_attribute("physics_weight"),
                                                config_data.get_attribute("smooth_weight"))

    # training loop

    num_steps = config_data.get_attribute("comb_training_steps")

    save_data = config_data.get_attribute("save_combination_data")

    if save_data:
        logger = _ResultLogger(save_dir)
        prev = naive

    for i in range(num_steps):
        # 1. Zero Gradients: Clear the "ledger" from the previous step
        optimizer.zero_grad()

        # 2. Forward Pass: Run data through the model
        prediction = unet(combined_input)

        # 3. Calculate Loss: Check physics constraints
        loss, stats = physics_informed_loss(prediction, known, inpainted, mask)

        # 3.5 Save prediction and results
        if save_data:
            logger.save_result(prediction, prev, stats, i)
            prev = prediction

        # 4. Backward Pass: Calculate how to adjust weights
        loss.backward()

        # 5. Optimizer Step: Apply the adjustments
        optimizer.step()

    # Extract and return data
    final_prediction = prediction.detach()
    if save_data:
        logger.write_stats()
    return final_prediction


class _ResultLogger:

    def __init__(self, save_loc: str):

        try:
            self.stats_list = []
            #Set up directories
            self.save_dir = Path(save_loc)
            self.save_dir.mkdir(parents=True, exist_ok=True)

            self.raw_dir = self.save_dir / "predicted_raw"
            self.visual_dir = self.save_dir / "predicted_visualized"
            self.change_dir = self.save_dir / "changed_visualized"
            self.div_dir = self.save_dir / "divergence_visualized"

            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.visual_dir.mkdir(parents=True, exist_ok=True)
            self.change_dir.mkdir(parents=True, exist_ok=True)
            self.div_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.exception(f"Exception when initializing vector combination logging: {e}")
            #TODO implement better error handling

    def save_result(self, pred: torch.Tensor, prev: torch.Tensor, stats: dict, step_num: int):
        try:
            self.stats_list.append(stats)
            #Save raw combined field
            torch.save(pred.detach(), self.raw_dir / f"raw_{step_num}.pt")
            #Save visualized predicted combined field
            plot_vector_field(pred[0][0].detach(), pred[0][1].detach(),
                              title=f"UNet Combined Field at Step{step_num}",
                              file=str(self.visual_dir / f"visualized_{step_num}.png"))
            #Save visualized divergence
            divergence = compute_divergence(pred[0][0].detach(), pred[0][1].detach())
            plot_vector_field(divergence[0][0], divergence[0][1],
                              title=f"Combined Field Divergence at Step{step_num}",
                              file=str(self.div_dir / f"divergence_{step_num}.png"))
            #Save visualized change predicted
            change = (prev - pred).detach()
            plot_vector_field(change[0][0], change[0][1],
                              title=f"Change in field at step {step_num}",
                              file=str(self.change_dir / f"change_{step_num}.png"))

        except Exception as e:
            logging.exception(f"Exception when saving during vector combination logging: {e}")

    def write_stats(self):
        stats_df = DataFrame(self.stats_list)
        stats_df.to_csv(self.raw_dir / "loss_data.csv", index=False)
