from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

model_path = "/data0/pretrained-models/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def calculate_token_num_single(example, text=False):
    history = example["history"]
    input = example["instruction"] + "\n" + example["input"]
    output = example["output"]
    messages = []
    if len(history):
        for his in history:
            messages.append({"role": "user", "content": his[0]})
            messages.append({"role": "assistant", "content": his[1]})
    messages.append({"role": "user", "content": input})
    messages.append({"role": "assistant", "content": output})

    if text:
        token = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return token
    else:
        token = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        return len(token)


def calculate_token_num(examples):
    outputs = {
        "instruction": [],
        "input": [],
        "output": [],
        "system": [],
        "history": [],
        "source": [],
        "nums": [],
    }
    for i in range(len(examples["instruction"])):
        history = examples["history"][i]
        input = examples["instruction"][i] + "\n" + examples["input"][i]
        output = examples["output"][i]
        messages = []
        if len(history):
            for his in history:
                messages.append({"role": "user", "content": his[0]})
                messages.append({"role": "assistant", "content": his[1]})
        messages.append({"role": "user", "content": input})
        messages.append({"role": "assistant", "content": output})

        token = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        outputs["nums"].append(len(token))
        outputs["instruction"].append(examples["instruction"][i])
        outputs["input"].append(examples["input"][i])
        outputs["output"].append(examples["output"][i])
        outputs["system"].append(examples["system"][i])
        outputs["history"].append(examples["history"][i])
        outputs["source"].append(examples["source"][i])
    return outputs


def merge_train_data(examples, num_threshold=8000):
    res = {
        "instruction": [],
        "input": [],
        "output": [],
        "system": [],
        "history": [],
        "source": [],
        "nums": [],
    }
    res_temp = {
        "instruction": [],
        "input": [],
        "output": [],
        "system": [],
        "history": [],
        "source": [],
        "nums": [],
    }
    for i in range(len(examples["instruction"])):
        nums = examples["nums"][i]
        history = examples["history"][i]
        if nums > num_threshold:
            continue
        if nums > 3000 or len(history):
            res["instruction"].append(examples["instruction"][i])
            res["input"].append(examples["input"][i])
            res["output"].append(examples["output"][i])
            res["system"].append(examples["system"][i])
            res["history"].append(examples["history"][i])
            res["source"].append(examples["source"][i])
            res["nums"].append(examples["nums"][i])
        else:
            res_temp["instruction"].append(examples["instruction"][i])
            res_temp["input"].append(examples["input"][i])
            res_temp["output"].append(examples["output"][i])
            res_temp["system"].append(examples["system"][i])
            res_temp["history"].append(examples["history"][i])
            res_temp["source"].append(examples["source"][i])
            res_temp["nums"].append(examples["nums"][i])

    # 随机拼凑样本
    index = 0
    rounds = [1, 2, 3, 4]
    random.seed(42)
    while index < len(res_temp["instruction"]):
        r = random.choice(rounds)
        if r + index >= len(res_temp["instruction"]):
            r = len(res_temp["instruction"]) - index - 1
            if r < 1:
                break

        example_temp = {}
        instructions = res_temp["instruction"][index : index + r]
        inputs = res_temp["input"][index : index + r]
        outputs = res_temp["output"][index : index + r]
        example_temp["instruction"] = instructions[0]
        example_temp["input"] = inputs[0]
        example_temp["output"] = outputs[0]
        example_temp["system"] = ""
        example_temp["source"] = "merge"
        historys = []
        for j in range(1, len(instructions)):
            historys.append([instructions[j] + "\n" + inputs[j], outputs[j]])
        example_temp["history"] = historys
        # 计算拼凑后的样本是否超过阈值
        token_num = calculate_token_num_single(example_temp)
        example_temp["nums"] = token_num
        if token_num > num_threshold:
            r = 2
            example_temp = {}
            instructions = res_temp["instruction"][index : index + r]
            inputs = res_temp["input"][index : index + r]
            outputs = res_temp["output"][index : index + r]
            example_temp["instruction"] = instructions[0]
            example_temp["input"] = inputs[0]
            example_temp["output"] = outputs[0]
            example_temp["system"] = ""
            example_temp["source"] = "merge"
            historys = []
            for j in range(1, len(instructions)):
                historys.append([instructions[j] + "\n" + inputs[j], outputs[j]])
            example_temp["history"] = historys
            example_temp["nums"] = token_num

        res["instruction"].append(example_temp["instruction"])
        res["input"].append(example_temp["input"])
        res["output"].append(example_temp["output"])
        res["system"].append(example_temp["system"])
        res["history"].append(example_temp["history"])
        res["source"].append(example_temp["source"])
        res["nums"].append(example_temp["nums"])
        index += r
    return res


if __name__ == "__main__":
    dataset = load_dataset(
        "json",
        data_files="/data3/yss/sft_datas/0628/sft_data_merge_v4_filter.jsonl",
        split="train",
    )
    dataset = dataset.map(calculate_token_num, batched=True, num_proc=48)
    print("ori num:", len(dataset))
    dataset1 = dataset.map(
        merge_train_data, batched=True, remove_columns=["nums"], num_proc=48
    )
    print("merge num:", len(dataset1))
    dataset1.to_json(
        "/data3/yss/sft_datas/0628/sft_data_merge_v4_filter_merge.jsonl",
        batch_size=1000,
        num_proc=48,
    )
    # for data in dataset1:
    #     if len(data["history"])>2:
    #         example = calculate_token_num_single(data,text=True)
    #         print(example)
    #         print("---------->",data["source"])
    #         print("*"*100)
    #         exit()