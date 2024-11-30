from clip import ClIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import model_converter


def preload_models(ckpt_path, device):
    # Load state dictionary from checkpoint
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # Initialize and load models
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = ClIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    # Return models as a dictionary
    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion
    }
