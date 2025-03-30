from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
from transformers.image_utils import load_image
import re
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "C:\\Users\\Ivan\\PycharmProjects\\SmolVLM\\.venv\\SmolVLM-256M-Instruct_count_objects\\checkpoint-1297",
    torch_dtype=torch.bfloat16,
    _attn_implementation=None,
).to(DEVICE)


image_url = "https://uchebnik.mos.ru/system/atomic_objects/files/010/675/568/original/Apples_Closeup_White_background_Red_Drops_543779_5184x3456.jpg"

data_files = {"train": "train-00000-of-00027.parquet"}
data = load_dataset("parquet", data_dir=".", data_files=data_files, split="train[:10%]")

image = load_image(image_url)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "How many objects are in the image?, Answer in one digit"}
        ]
    },
]


prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

acc = 0
for sample in tqdm(data):
    inputs = processor(text=prompt, images=[sample["image"]], return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output = int(re.findall(r'\d+',  generated_texts)[-1])
    label = int(sample["solution"].replace("<answer> ", "").replace(" </answer>", ""))
    if output == label:
        acc +=1

print(acc/len(data))

# print(int(re.findall(r'\d+',  generated_texts)[-1]))
# print(int(data["solution"][i].replace("<answer> ", "").replace(" </answer>", "")))