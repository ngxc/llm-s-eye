# tiny_minimind_vlm_clip_train_gpt_medium_clip_large.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import math
import random


IMAGE_DIR = r"Images"
TOKEN_FILE = r"captions.txt"

BATCH_SIZE = 1
MAX_LEN = 600
LR = 5e-5
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PLACEHOLDER_TOKEN = "<image>"
NUM_PATCHES = 196
NUM_PLACEHOLDER = 32
CONTRASTIVE_WEIGHT = 0.5
DATA_RATIO = 0.1
GRAD_ACCUM_STEPS = 5


class FlickrDataset(Dataset):
    def __init__(self, data_list, tokenizer, processor, max_len=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item['image_path']).convert("RGB")
        vis = self.processor(images=image, return_tensors="pt")

        caption_with_placeholder = f"{(' ' + PLACEHOLDER_TOKEN) * NUM_PLACEHOLDER}\n{item['caption']}"
        txt = self.tokenizer(
            caption_with_placeholder,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )

        vis = {k: v.squeeze(0) for k, v in vis.items()}
        txt = {k: v.squeeze(0) for k, v in txt.items()}
        return vis, txt


class VisionToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        return x



class TinyMiniMindCLIPContrastiveVLM(nn.Module):
    def __init__(self, text_model, vision_model, proj, placeholder_token_id, mem_size=8):
        super().__init__()
        self.text_model = text_model
        self.vision_model = vision_model
        self.proj = proj
        self.vis_norm = nn.LayerNorm(vision_model.config.hidden_size).to(DEVICE)
        self.placeholder_token_id = placeholder_token_id

        # memory bank
        self.mem_size = mem_size
        self.memory_text = []
        self.memory_img = []

    def forward(self, vision_inputs, input_ids, attention_mask, labels=None):
        # 图像编码
        for k in vision_inputs:
            vision_inputs[k] = vision_inputs[k].to(DEVICE)
        vis_out = self.vision_model(**vision_inputs)
        vis_feats = vis_out.last_hidden_state
        vis_feats = self.vis_norm(vis_feats)
        vis_proj = self.proj(vis_feats)


        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        text_emb = self.text_model.transformer.wte(input_ids)


        placeholder_mask = (input_ids == self.placeholder_token_id)
        placeholder_embs = []
        patches_per_token = math.ceil(vis_proj.size(1) / NUM_PLACEHOLDER)

        for b in range(text_emb.size(0)):
            idxs = torch.nonzero(placeholder_mask[b], as_tuple=True)[0]
            for i, idx in enumerate(idxs):
                start = i * patches_per_token
                end = min((i + 1) * patches_per_token, vis_proj.size(1))
                if start >= vis_proj.size(1):
                    break
                text_emb[b, idx] = vis_proj[b, start:end].mean(dim=0)
            placeholder_embs.append(vis_proj[b].mean(dim=0))

        placeholder_embs = torch.stack(placeholder_embs)


        outputs = self.text_model(inputs_embeds=text_emb, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss


        text_emb_pool = nn.functional.normalize(placeholder_embs, dim=-1)
        img_emb_pool = nn.functional.normalize(vis_proj.mean(dim=1), dim=-1)


        self.memory_text.append(text_emb_pool.detach())
        self.memory_img.append(img_emb_pool.detach())
        if len(self.memory_text) > self.mem_size:
            self.memory_text.pop(0)
            self.memory_img.pop(0)

        contrastive_loss = torch.tensor(0.0, device=DEVICE)


        if len(self.memory_text) >= 2:
            text_batch = torch.cat(self.memory_text, dim=0)
            img_batch = torch.cat(self.memory_img, dim=0)

            sim_matrix = text_batch @ img_batch.t() / 0.07
            labels_contrastive = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            contrastive_loss = (nn.CrossEntropyLoss()(sim_matrix, labels_contrastive) +
                                nn.CrossEntropyLoss()(sim_matrix.t(), labels_contrastive)) / 2

        total_loss = lm_loss + CONTRASTIVE_WEIGHT * contrastive_loss
        return total_loss, lm_loss, contrastive_loss



if __name__ == "__main__":

    data_list = []
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                img_tag, caption = line.split(",", 1)
            except ValueError:
                continue
            img_name = img_tag.split("#")[0]
            img_path = os.path.join(IMAGE_DIR, img_name)
            if os.path.exists(img_path):
                data_list.append({"image_path": img_path, "caption": caption})


    if DATA_RATIO < 1.0:
        random.shuffle(data_list)
        num_samples = int(len(data_list) * DATA_RATIO)
        data_list = data_list[:num_samples]

    print(f"总样本数: {len(data_list)} (采样比例: {DATA_RATIO})")


    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    text_model.config.pad_token_id = tokenizer.pad_token_id
    text_model.to(DEVICE)

    if PLACEHOLDER_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([PLACEHOLDER_TOKEN])
    placeholder_token_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    text_model.resize_token_embeddings(len(tokenizer))


    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vision_model.to(DEVICE)

    vision_dim = vision_model.config.hidden_size
    text_dim = text_model.config.n_embd
    proj = VisionToTextProj(vision_dim, text_dim).to(DEVICE)

    vlm = TinyMiniMindCLIPContrastiveVLM(text_model, vision_model, proj, placeholder_token_id).to(DEVICE)


    dataset = FlickrDataset(data_list, tokenizer, vision_processor, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW(list(proj.parameters()) + list(text_model.parameters()), lr=LR)


    vlm.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, (vis_inputs, txt_inputs) in enumerate(loop):
            input_ids = txt_inputs['input_ids'].to(DEVICE)
            attention_mask = txt_inputs['attention_mask'].to(DEVICE)
            labels = input_ids.clone()

            total_loss, lm_loss, contrastive_loss = vlm(vis_inputs, input_ids, attention_mask, labels)
            total_loss = total_loss / GRAD_ACCUM_STEPS
            total_loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            loop.set_postfix(total_loss=total_loss.item()*GRAD_ACCUM_STEPS,
                             lm_loss=lm_loss.item(),
                             contrastive_loss=contrastive_loss.item())


        save_dict = {
            "proj_state_dict": proj.state_dict(),
            "text_model_state_dict": text_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(save_dict, f"tiny_minimind_clip_medium_large_epoch{epoch}.pt")
        print(f"Saved: tiny_minimind_clip_medium_large_epoch{epoch}.pt")

    print("训练完成。")
