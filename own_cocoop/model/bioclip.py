import torch
import torch.nn as nn
from torch.nn import functional as F
import open_clip

_tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask
        self.cast_dtype = torch.float16
        self.token_embedding = clip_model.token_embedding


    def forward(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x
    
    
def load_bioclip_to_cpu():
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    model.eval()
    return model, preprocess_train, preprocess_val

class CustomCLIP(nn.Module):
    def __init__(self, classnames, bioclip_model):
        super().__init__()
        self.classnames = classnames
        self.tokenized_prompts = self._tokenize_prompts(classnames)
        self.image_encoder = bioclip_model.encode_image
        self.text_encoder = TextEncoder(bioclip_model)
        self.logit_scale = bioclip_model.logit_scale  
        self.dtype = torch.float32

    def _tokenize_prompts(self, classnames):
        prompts = [f"a photo of a {name}." for name in classnames]
        tokenized_prompts = torch.cat([_tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        return tokenized_prompts

    def forward(self, images):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(images.type(self.dtype)).to(images.device)
        image_features = F.normalize(image_features, p=2, dim=-1)

        tokenized_prompts = self.tokenized_prompts.to(images.device) 
        text_features = self.text_encoder(tokenized_prompts).to(images.device) 
        text_features = F.normalize(text_features, p=2, dim=-1)

        logits = logit_scale * image_features @ text_features.t()

        return logits