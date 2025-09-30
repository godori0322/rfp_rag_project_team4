from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


def load_model_and_tokenizer(model_name, use_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=get_bnb_config(), trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    return model, tokenizer


def load_llm(model_name, use_4bit=True):
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit)
    pipe = pipeline(
        model = model,
        tokenizer = tokenizer,
        task = "text-generation",
        do_sample = True,
        temperature = 0.2,
        repetition_penalty = 1.2,
        return_full_text = False,
        max_new_tokens = 1000,
    )
    return HuggingFacePipeline(pipeline=pipe)