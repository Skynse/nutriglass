from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
import torch

base_id = "unsloth/Qwen3-VL-4B-Instruct-bnb-4bit"
# The checkpoint's skip list uses bare names like "visual", which transformers 5.x
# only matches as a prefix, so the unquantized vision tower gets wrapped in
# Linear4bit. Spell out the full module paths instead.
config = AutoConfig.from_pretrained(base_id)
config.quantization_config["llm_int8_skip_modules"] = [
    "model.visual",
    "lm_head",
    "model.language_model.embed_tokens",
]
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_id, config=config, dtype=torch.bfloat16, device_map="cuda"
)
model = PeftModel.from_pretrained(model, "hungvtm/qwen3vl-food-lora")
processor = AutoProcessor.from_pretrained("hungvtm/qwen3vl-food-lora")

image = Image.open("food.jpg").convert("RGB")
prompt = (
    "Analyze this food image and provide a complete description including: "
    "dish name, food type, main ingredients, cooking method, portion sizes, "
    "and nutritional information (calories, protein, fat, carbohydrates)."
)
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.3)
print(processor.decode(out[0], skip_special_tokens=True))
