import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

def inicializar_modelo():
    """Carrega os modelos na memória (Roda apenas uma vez no startup do servidor)"""
    print("Carregando modelos nativos do Shap-E...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    
    print(f"Modelos carregados com sucesso no dispositivo: {device}")
    return {
        "xm": xm,
        "model": model,
        "diffusion": diffusion,
        "device": device
    }

def gerar_modelo_3d(componentes, prompt: str, output_path: str = "saida.obj"):
    """Executa a inferência e gera o arquivo OBJ diretamente"""
    print(f"Iniciando geração nativa para: '{prompt}'...")
    
    device = componentes["device"]
    is_cuda = (device.type == "cuda")

    # Gera os latentes (a matemática do objeto)
    latents = sample_latents(
        batch_size=1,
        model=componentes["model"],
        diffusion=componentes["diffusion"],
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

    print("Decodificando os latentes para malha 3D...")
    mesh = decode_latent_mesh(componentes["xm"], latents[0]).tri_mesh()

    # Salva diretamente em OBJ
    with open(output_path, "w") as f:
        mesh.write_obj(f)

    print(f"Sucesso! Arquivo salvo em: {output_path}")
    return output_path

if __name__ == "__main__":
    # Teste Local
    modelos = inicializar_modelo()
    gerar_modelo_3d(modelos, "a red birthday cake", "birthday_cake.obj")