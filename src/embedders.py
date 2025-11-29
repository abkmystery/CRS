import torch
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image
import numpy as np


class MultimodalEmbedder:
    def __init__(self):
        # Force CPU if no NVIDIA GPU found (SentenceTransformers handles Intel optimizations automatically)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Loading Text Model on {self.device}...")
        self.text_model = SentenceTransformer('all-mpnet-base-v2', device=self.device)

        print(f"Loading Image Model on {self.device}...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                                         pretrained='laion2b_s34b_b79k')
        if self.device == 'cuda':
            self.clip_model.cuda()
        self.clip_model.eval()

    def embed_text_batch(self, texts):
        # The Secret Sauce for Speed:
        # SentenceTransformers automatically uses parallel threads on CPU for batches
        if not texts: return []
        embeddings = self.text_model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_text(self, text):
        if not text: return None
        return self.embed_text_batch([text])[0]

    def embed_image(self, image_path):
        if not image_path: return None
        try:
            image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0)
            if self.device == 'cuda':
                image = image.cuda()
            with torch.no_grad():
                emb = self.clip_model.encode_image(image)
            return emb.cpu().numpy()[0].tolist()
        except:
            return None