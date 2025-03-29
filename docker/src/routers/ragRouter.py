from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import time
import torch
from threading import Thread
import numpy as np


class LLM:
    def __init__(self, gen_model='speakleash/Bielik-11B-v2.3-Instruct'):
        self.llm_loaded = False
        self.gen_model = 'speakleash/Bielik-11B-v2.3-Instruct'

    def load_modules(self):
        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.gen_model)
        self.chat_llm = AutoModelForCausalLM.from_pretrained(self.gen_model, torch_dtype=torch.bfloat16, device_map="auto")
        self.chat_streamer = TextIteratorStreamer(self.chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.llm_loaded = True

    def apply_template(self, data):
        messages = []
        messages.append({"role": "system", "content": data.system_message})
        messages.append({"role": "user", "content": data.user_message})
        inputs_tp = self.chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return inputs_tp
    
    def generate_response(self, data):
        if not self.llm_loaded:
            self.load_modules()

        inputs_tp = self.apply_template(data)

        inputs = self.chat_tokenizer(inputs_tp, return_tensors="pt", padding=True)
        attention_mask = inputs["attention_mask"]
        generation_kwargs = dict(inputs=inputs['input_ids'], attention_mask=attention_mask, pad_token_id=self.chat_tokenizer.eos_token_id, streamer=self.chat_streamer, max_new_tokens=1000, do_sample=False)

        thread = Thread(target=self.chat_llm.generate, kwargs=generation_kwargs)
        thread.start()

        time.sleep(2)
        generated_text = ""
        for new_text in self.chat_streamer:
            generated_text += new_text
        
        return generated_text

router = APIRouter(
    prefix='/api',
    tags=['api'],
    responses={404: {'description': 'Not found'}},
)

# Define input data
class InputData(BaseModel):
    user_message: str
    system_message: str

llm_model = LLM()

# Create endpoint for the Bielik LLM
@router.post("/generate")
async def get_prediction(data: InputData):

    response_text = llm_model.generate_response(data)
    
    result = {
        "model_response": response_text
    }
    return result
