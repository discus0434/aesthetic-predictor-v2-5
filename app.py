import inspect

import gradio as gr
import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image


class AestheticPredictor:
    def __init__(self):
        # load model and preprocessor
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self.model = self.model.to(torch.bfloat16).cuda()

    def inference(self, image: Image.Image) -> float:
        # preprocess image
        pixel_values = self.preprocessor(
            images=image.convert("RGB"), return_tensors="pt"
        ).pixel_values

        if torch.cuda.is_available():
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # predict aesthetic score
        with torch.inference_mode():
            score = self.model(pixel_values).logits.squeeze().float().cpu().numpy()

        return score


if __name__ == "__main__":
    aesthetic_predictor = AestheticPredictor()
    with gr.Blocks(theme="soft") as blocks:
        markdown = gr.Markdown(
            value=inspect.cleandoc(
                """
                # Aesthetic Predictor V2.5

                This app predicts the aesthetic score of input images such as paintings,
                photographs, and illustrations.

                The aesthetic score is a floating-point number between 1 and 10.

                5.5+ is considered to be a good aesthetic score.
                """
            )
        )

        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input Image", type="pil")
                button = gr.Button("Predict")

            with gr.Column():
                score = gr.Textbox(label="Aesthetic Score")

        button.click(aesthetic_predictor.inference, inputs=image, outputs=score)

    blocks.queue().launch()
