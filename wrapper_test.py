import torch

from WSI_LLaVA.llava.model.multimodal_encoder.resampler_wrapper import ResamplerWrapper


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    model = ResamplerWrapper().to(device)
    model.eval()

    x_list = [
        torch.randn(120, 1536, device=device),
        torch.randn(85, 1536, device=device),
        torch.randn(300, 1536, device=device),
    ]

    with torch.no_grad():
        out = model(x_list)

    print("[INFO] output shape:", out.shape)
    print("[INFO] expected shape: (3, 576, 1024)")
    assert out.shape == (3, 576, 1024), f"Unexpected output shape: {out.shape}"
    print("[PASS] ResamplerWrapper forward test passed.")


if __name__ == "__main__":
    main()