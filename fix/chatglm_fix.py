from transformers import AutoModel,AutoTokenizer
import torch
from peft import PeftModel
import json
from config import *


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto', proxies=PROXIES)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, proxies=PROXIES)

if FINETUNE_SHARED.IS_FINETUNED:
    model = PeftModel.from_pretrained(model, FINETUNE_SHARED.FINETUNE_MODEL_PATH)
model.half()

instructions = json.load(open(os.path.join(FINETUNE_DATA_DIR, FIX_DATASET + '.json')))

with torch.no_grad():
    for idx, item in enumerate(instructions):
        feature = item
        input_text = feature['context']
        wrong_file_path = feature['wrong_file_path']
        '''
        wrong_file_path: $DATA_DIR/problemID/buggy/userID_buggy.cpp
        fixed_file_path: $DATA_DIR/problemID/{FINETUNE_SHARED.FIXED_DIR}/userID_fixed.cpp
        '''
        fixed_file_path = wrong_file_path.replace('buggy', FINETUNE_SHARED.FIXED_DIR, 1).replace('buggy', 'fixed')
        ids = tokenizer.encode(input_text)
        
        input_ids = torch.LongTensor([ids])
        input_ids = input_ids.to(device)
        out = model.generate(
            input_ids=input_ids,
            max_length=1500,
            do_sample=False,
            temperature=0,
        )
        out_text = tokenizer.decode(out[0])
        real_input = tokenizer.decode(input_ids[0])
        answer = out_text.replace(real_input, "").replace("\nEND", "").strip()
        # write to file
        if not os.path.exists(os.path.dirname(fixed_file_path)):
            os.makedirs(os.path.dirname(fixed_file_path))
        with open(fixed_file_path, 'w', encoding='utf-8') as f:
            f.write(answer)
            print(f"write to {fixed_file_path}")
        f.close()