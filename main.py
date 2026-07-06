from transformers import AutoModelForCausalLM, AutoTokenizer
def main():
    AutoTokenizer.from_pretrained("microsoft/phi-4")
    AutoModelForCausalLM.from_pretrained("microsoft/phi-4", device_map="cpu", trust_remote_code=True)

    AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="cpu", trust_remote_code=True)

if __name__ == "__main__":
    main()
