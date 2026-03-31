import os
import unittest
import torch
from torchvision.utils import save_image

OUTPUT_DIR = "outputs/tests"
MODEL_NAME = "Daniel0724/SimpleAR-0.5B-RL"
VQ_CKPT = "./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16"
PROMPT = "a photo of a cat"
CODEBOOK_SIZE = 64000
LATENT_SIZE = 64


class TestGenerate(unittest.TestCase):

    def test_generate_image(self):
        from transformers import AutoTokenizer
        from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
        from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer
        from simpar.model.language_model.simpar_qwen2 import SimpARForCausalLM, CFGLogits
        from transformers.generation import LogitsProcessorList

        device = "cuda:0"

        model = SimpARForCausalLM.from_pretrained(MODEL_NAME, device_map=device, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        tokenizer_config = TokenizerConfigs["DV"].value
        tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
        vq_model = CausalVideoTokenizer(
            checkpoint_enc=f"{VQ_CKPT}/encoder.jit",
            checkpoint_dec=f"{VQ_CKPT}/decoder.jit",
            tokenizer_config=tokenizer_config,
        )
        vq_model.eval().requires_grad_(False)

        input_ids = tokenizer(
            "<|t2i|>A highly realistic image of " + PROMPT + "<|soi|>",
            return_tensors="pt",
        ).input_ids.to(device)
        uncond_ids = tokenizer(
            "<|t2i|>An image of aerial view, overexposed, low quality, deformation, "
            "a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion<|soi|>",
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                logits_processor=LogitsProcessorList([CFGLogits(6.0, uncond_ids, model)]),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=CODEBOOK_SIZE,
                max_new_tokens=LATENT_SIZE ** 2,
                use_cache=True,
            )

        tokens = output_ids[:, input_ids.shape[1]: input_ids.shape[1] + LATENT_SIZE ** 2].clone()
        tokens = (tokens - len(tokenizer)).clamp(0, CODEBOOK_SIZE - 1)
        tokens = tokens.reshape(-1, LATENT_SIZE, LATENT_SIZE).unsqueeze(1)

        with torch.inference_mode():
            image = vq_model.decode(tokens).squeeze(2)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "test_generate.png")
        save_image(image, output_path, normalize=True, value_range=(-1, 1))

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
