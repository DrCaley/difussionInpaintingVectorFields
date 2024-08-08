# difussionInpaintingVectorFields
Notes for future researchers in case I get hit by a bus or a better offer next summer. As of early August 2024:
- Most of the exciting and functional code is in the medium_ddpm folder
- We've been mostly using models trained by train-ocean-xl.py, which is the extra large model that is between okay and good at predicting currents
- The script train-ocean-xl-wl.py (extra large with stream equation loss function) is functional but we haven't determined how good it is yet.
- All the other code for training is outdated and may or may not work.
- We use inpainting-model-test.py for data collection and inpainting.
- Everything in the "scripts" folder is failed early attempts
- A promising next step that we haven't done yet would be making each "image" follow the incompressible flow equation at every diffusion step. 

Happy Coding! - Ember McEwen