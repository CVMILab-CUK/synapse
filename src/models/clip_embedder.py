import torch 
import torch.nn as nn
import open_clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch.nn.functional as F

class Frozen_CLIPImageEmbedder():
    def __init__(self, model_id='ViT-H-14', pretrained='laion2b_s32b_b79k', force_custom_text=True):
        self.model, self.train_transform, self.eval_transform = open_clip.create_model_and_transforms(model_id, pretrained=pretrained, force_custom_text=force_custom_text)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def __call__(self, img):
        img = F.interpolate(img, size=224)
        return self.model.encode_image(img)


class Frozen_CLIPImage2TextEmbedder(nn.Module):
    def __init__(self, 
                 model_id='ViT-H-14', 
                 caption_model_id = "Salesforce/blip2-flan-t5-xl", 
                 pretrained='laion2b_s32b_b79k', 
                 force_custom_text=True,
                 max_new_tokens = 75,
                 num_beams = 3,
                ):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.num_beams      = num_beams

        # Caption Maker
        self.blip_processor = Blip2Processor.from_pretrained(caption_model_id)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            caption_model_id,
            # device_map="auto",
            torch_dtype=torch.float16
        ).eval()

        self.model, self.train_transform, self.eval_transform = open_clip.create_model_and_transforms(model_id, pretrained=pretrained, force_custom_text=force_custom_text)
        self.model.eval()
        self.text_encoder = self.model.text
        self.tokenizer = open_clip.get_tokenizer(model_id)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, img:torch.tensor, ori_img):
        device = img.device
        blip_inputs = self.blip_processor(images=ori_img, return_tensors="pt").to(device, torch.float16)
        clip_image_inputs = F.interpolate(img, size=224)
        caption_ids = self.blip_model.generate(
            **blip_inputs,
            max_new_tokens=self.max_new_tokens,   # Ensure long enough caption
            num_beams=self.num_beams,
            do_sample=False
        )
        caption = self.blip_processor.batch_decode(caption_ids, do_rescale=False, skip_special_tokens=True)[0] # Maded caption
        # print(caption)
        tokens = self.tokenizer(caption).to(device)
        x = self.model.text.token_embedding(tokens)  # [B, 77, D]
        x = x + self.model.text.positional_embedding
        x = self.model.text.transformer(x)
        x = self.model.text.ln_final(x)  # [B, 77, D]
        return x, self.model.encode_image(clip_image_inputs)
