import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

def initialize_model():
    """Loads the models in the memory (runs once in the server)"""
    print("Loading Shap-e models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    
    print(f"Models loaded sucessfully on the device: {device}")
    return {
        "xm": xm,
        "model": model,
        "diffusion": diffusion,
        "device": device
    }

def generate_3d_model(components, prompt: str, output_path: str = "output.obj"):
    """Executes the inference and generates the .obj file directly"""
    print(f"Initializing native generation of: '{prompt}'...")
    
    device = components["device"]
    is_cuda = (device.type == "cuda")

    latents = sample_latents(
        batch_size=1,
        model=components["model"],
        diffusion=components["diffusion"],
        guidance_scale=15.0,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=is_cuda,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    print("Decoding the latents of the mesh...")
    mesh = decode_latent_mesh(components["xm"], latents[0]).tri_mesh()

    # Saves as .obj
    with open(output_path, "w") as f:
        mesh.write_obj(f)

    print(f"Sucess! File saved at: {output_path}")
    return output_path

def generate(prompt):
    models = initialize_model()
    generate_3d_model(models, prompt, "generated_model.obj")