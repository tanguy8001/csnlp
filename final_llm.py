from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

class FrozenLLMTranslator(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", device='cpu', soft_prompt="The sign language translation is: ", normalize_visual_embeds=True):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        # Ensure PAD token is set
        if self.tokenizer.pad_token is None:
            # Don't use EOS as padding - add a dedicated pad token
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            # Resize embeddings to account for the new token
            self.model.resize_token_embeddings(len(self.tokenizer))
        print(f"DEBUG: EOS token ID: {self.tokenizer.eos_token_id}, PAD token ID: {self.tokenizer.pad_token_id}") # DEBUG

        self.model.eval()
        self.soft_prompt = soft_prompt
        print(f"DEBUG: Textual soft prompt: '{self.soft_prompt}'") # DEBUG
        self.prompt_embeds = self._get_prompt_embeddings(self.soft_prompt).to(self.model.dtype)
        print(f"DEBUG: Textual soft prompt token IDs: {self.tokenizer(self.soft_prompt, return_tensors='pt').input_ids.to(self.device)}") # DEBUG

        self.normalize_visual_embeds = normalize_visual_embeds
        if self.normalize_visual_embeds:
            self.visual_embed_layer_norm = None

    def _get_prompt_embeddings(self, prompt_text):
        with torch.no_grad():
            token_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
            embeddings = self.get_text_embeddings(token_ids)  # (1, P, D)
        return embeddings

    def get_text_embeddings(self, token_ids):
        return self.model.get_input_embeddings()(token_ids)

    def compute_loss(self, visual_embeddings, target_token_ids):
        """
        visual_embeddings: (B, V, D)
        target_token_ids: (B, T)
        """
        B, V, D = visual_embeddings.shape
        T_target = target_token_ids.size(1)

        # Expand prompt_embeds to batch size
        prompt_embeds_batch = self.prompt_embeds.expand(B, -1, -1) # (B, P, D)
        P_len = prompt_embeds_batch.size(1)

        # Text embeddings for all but last target token
        target_input_ids = target_token_ids[:, :-1] # (B, T_target-1)
        text_embeds = self.get_text_embeddings(target_input_ids)  # (B, T_target-1, D)

        # Full input embeddings: [prompt; visual; text]
        processed_visual_embeddings = visual_embeddings
        if self.normalize_visual_embeds:
            if self.visual_embed_layer_norm is None or self.visual_embed_layer_norm.normalized_shape[0] != D:
                self.visual_embed_layer_norm = nn.LayerNorm(D, device=self.device)
            processed_visual_embeddings = self.visual_embed_layer_norm(visual_embeddings)

        inputs_embeds = torch.cat([prompt_embeds_batch, processed_visual_embeddings, text_embeds], dim=1)  # (B, P + V + T_target-1, D)

        # Build attention mask: 1 for all input tokens (prompt, visual, text)
        attention_mask = torch.ones(inputs_embeds.size()[:2], device=self.device, dtype=torch.long)

        # Labels:
        # - Ignore prompt positions by setting label to -100
        # - Ignore visual embeddings positions by setting label to -100
        # - For text part, labels should be w2, ..., w_n, which is target_token_ids[:, 1:]
        labels_for_text = target_token_ids[:, 1:] # (B, T_target-1)
        
        # Create padding mask: True for padding tokens
        padding_mask = labels_for_text == self.tokenizer.pad_token_id
        # Replace padding tokens with -100 to ignore them in loss computation
        labels_for_text = labels_for_text.masked_fill(padding_mask, -100)

        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device),      # Ignore prompt
            torch.full((B, V), -100, dtype=torch.long, device=self.device),        # Ignore visual
            labels_for_text                                                         # Predict these
        ], dim=1)  # (B, P + V + T_target-1)

        # Forward pass with labels (will compute cross‐entropy internally)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    def generate(self, embeddings, max_new_tokens=150, num_beams=1, early_stopping=True, do_sample=True, temperature=0.7, top_k=50):
        """
        embeddings: (B, N_chunks, D_llm)
        """
        self.model.eval()
        B, N_chunks, D_llm = embeddings.shape
        P_embeds = self.prompt_embeds.expand(B, -1, -1)
        P_len = self.prompt_embeds.shape[1]

        processed_embeddings = embeddings
        if self.normalize_visual_embeds:
            if self.visual_embed_layer_norm is None or self.visual_embed_layer_norm.normalized_shape[0] != D_llm:
                self.visual_embed_layer_norm = nn.LayerNorm(D_llm, device=self.device).to(self.model.dtype)
            processed_embeddings = self.visual_embed_layer_norm(embeddings)
        else:
            print("  LLM Generate - Skipping visual embed normalization.")

        input_embeds = torch.cat([P_embeds, processed_embeddings], dim=1)

        attention_mask = torch.ones(input_embeds.size()[:2], device=self.device, dtype=torch.long)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)

        return decoded_texts
