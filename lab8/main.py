from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


def load_file(filename, encoding='utf-8'):
    with open(filename, 'r', encoding=encoding) as f:
        return f.read()


def save_file(filename, content, encoding='utf-8'):
    with open(filename, 'w', encoding=encoding) as f:
        f.write(content)


def process_prompt(model, tokenizer, prompt_text, data):
    text_for_analysis = data[0:5000]
    final_prompt = prompt_text.replace("{текст_для_анализа}", text_for_analysis)

    messages = [
        {"role": "user", "content": final_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    generated_ids_ = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids_, skip_special_tokens=True)[0]

    return response


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    # model_dir = snapshot_download(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = load_file('/kaggle/input/datasets/ilyach12/123456/ENG_article.txt', encoding='utf-8')

    prompts_to_process = [
        {"prompt_file": "/kaggle/input/datasets/ilyach12/123456/prompt1.txt", "result_file": "result1.txt"},
        {"prompt_file": "/kaggle/input/datasets/ilyach12/123456/prompt2.txt", "result_file": "result2.txt"}
    ]

    all_results = {}

    for item in prompts_to_process:
        prompt_file = item["prompt_file"]
        result_file = item["result_file"]

        prompt_text = load_file(prompt_file)

        result = process_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            data=data,
        )

        all_results[prompt_file] = result

    summary_content = "Результат\n\n"

    for i, (prompt_file, result) in enumerate(all_results.items(), 1):
        summary_content += f"РЕЗУЛЬТАТ {i} (из {prompt_file}):\n"
        summary_content += "-" * 40 + "\n"

        lines = result.split('\n')
        answer_lines = [line for line in lines if line.strip().startswith(('1.', '2.', '3.'))]

        for line in answer_lines[:3]:
            summary_content += line + "\n"

    save_file("summary_report.txt", summary_content)
