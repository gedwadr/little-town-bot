import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class BotV1:
    def __init__(self, model_path: str, adapter_path: str):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

    def get_actions(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=128,      # actions are short
                do_sample=False,         # greedy — deterministic output
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (skip the input)
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
