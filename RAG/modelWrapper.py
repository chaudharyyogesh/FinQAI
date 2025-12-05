import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Path to your locally stored fine-tuned model
model_path = "./your-local-llama-model-folder" 

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto", # Uses GPU if available
    torch_dtype=torch.float16,
    load_in_4bit=True # Optional: Use if you are low on VRAM
)

# Create the pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1, # Keep low for financial facts
    repetition_penalty=1.1
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)