from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from datasets import Dataset, concatenate_datasets, Features, Sequence, Value
import json

model_path = "/data0/pretrained-models/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def split_by_source(dataset: Dataset, test_size=0.02, num_threshold=8000):
    datas = {}
    for data in tqdm(dataset):
        sr = data["source"]
        if not sr in datas:
            datas[sr] = {
                "system": [],
                "tools": [],
                "conversations": [],
                "source": [],
                "nums": [],
            }
        # if data["nums"] < num_threshold and data["source"] in [
        #     "multi_turn_datas_0808.json",
        #     "table_python_code_datas.json",
        # ]:
        if data["nums"] < num_threshold:
            datas[sr]["system"].append(data["system"])
            datas[sr]["tools"].append(data["tools"])
            datas[sr]["conversations"].append(data["conversations"])
            datas[sr]["source"].append(data["source"])
            datas[sr]["nums"].append(data["nums"])
    ds = {}
    features = Features(
        {
            "system": Value("string"),
            "tools": Value("string"),
            "conversations": Sequence(Value("dict")),
            "source": Value("string"),
            "nums": Value("int32"),
        }
    )
    for key, value in datas.items():
        ds[key] = Dataset.from_dict(value, features=features)

    trains = []
    tests = []
    for source, d in ds.items():
        d = d.train_test_split(test_size=test_size, seed=42)
        # ds_train[source] = d["train"]
        # ds_test[source] = d["test"]
        trains.append(d["train"])
        tests.append(d["test"])
    ds_train = concatenate_datasets(trains)
    ds_test = concatenate_datasets(tests)
    ds_train = ds_train.shuffle(seed=42)
    ds_test = ds_test.shuffle(seed=42)
    return ds_train, ds_test


def calculate_token_num_single(example, text=False):
    history = example["history"]
    input = example["instruction"] + "\n" + example["input"]
    output = example["output"]
    system = example["system"]
    messages = []
    if len(system):
        messages.append({"role": "system", "content": system})
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


def filter_token_num(examples, num_threshold=7900):
    outputs = {
        "system": [],
        "tools": [],
        "conversations": [],
        "source": [],
    }
    for i in range(len(examples["conversations"])):
        system = examples["system"][i]
        tools = examples["tools"][i]
        conversations = examples["conversations"][i]
        messages = []
        # if not len(tools):
        #     continue
        # if len(conversations)<=2:
        #     continue
        if len(system):
            messages.append({"role": "system", "content": system})
        if len(tools):
            messages.append({"role": "tools", "content": tools})

        for conv in conversations:
            messages.append({"role": conv["from"], "content": conv["value"]})

        token = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if len(token)>num_threshold:
            continue
        outputs["system"].append(examples["system"][i])
        outputs["tools"].append(examples["tools"][i])
        outputs["conversations"].append(examples["conversations"][i])
        outputs["source"].append("source")
    return outputs


def merge_train_data(examples, num_threshold=8000, seed=42):
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
    random.seed(seed)
    while index < len(res_temp["instruction"]):
        # index不能超过数据长度
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
            # 拼接后样本长度程度超出阈值，直接设定r=2
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


def save_jsonl(dataset: Dataset, output_path):
    with open(output_path, "w") as f:
        for data in tqdm(dataset):
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    dataset = []
    with open(
        "/data3/yss/sft_datas/0912/sft_data_merge_v18_quality_filtered_sharegpt.jsonl",
        mode="r",
    ) as f:
        for line in f:
            dataset.append(json.loads(line))
    datas = []
    
    for data in tqdm(dataset):
        data_new = {}
        if "tools" in data:
            if isinstance(data["tools"], list):
                if len(data["tools"]):
                    data["tools"] = json.dumps(data["tools"])
                else:
                    data["tools"] = ""
            elif (
                data["tools"] == "[]" or data["tools"] == "[ ]" or data["tools"] == " "
            ):
                data["tools"] = ""
            else:
                data["tools"] = data["tools"]
        else:
            data["tools"] = ""

        data_new["tools"] = data["tools"]
        data_new["system"] = data["system"]
        data_new["conversations"] = data["conversations"]
        datas.append(data_new)
    data_path_trans = "/data3/yss/sft_datas/0912/sft_data_merge_v18_quality_filtered_sharegpt_trans.jsonl"
    # save_jsonl(datas, data_path_trans)

    dataset = load_dataset(
        "json",
        data_files=data_path_trans,
        split="train",
    )
    print(dataset)

    print("ori num:", len(dataset))
    # 1.计算样本token长度
    dataset = dataset.map(filter_token_num, batched=True, num_proc=96)
    print(len(dataset))
    # dataset.to_json(
    #     "/data3/yss/sft_datas/0912/sft_data_merge_v18_quality_filtered_sharegpt_trans_tools_trains.jsonl",
    #     batch_size=1000,
    #     num_proc=48,
    # )
    exit()
    # 2.过滤token长度过长的样本
    # dataset = dataset.map(filter_by_tokens, batched=True, num_proc=48)
    # print("filter num:", len(dataset))
    # 3.根据source划分train和test
    ds_train, ds_test = split_by_source(dataset, num_threshold=8000)
    print("train:", len(ds_train), "test:", len(ds_test))

    # 仅仅table数据
    # ds_table = concatenate_datasets([ds_train, ds_test])
    # print("dataset table",len(ds_table))
    # save_jsonl(ds_table, "/data3/yss/sft_datas/0808/sft_data_merge_v16_quality_table.jsonl")

    # TODO split_by_source处理后的dataset后续处理非常慢，先保存再load出来
    train_tmp_path = "/data3/yss/sft_datas/0817/sft_data_merge_v17_quality_train.jsonl"
    test_tmp_path = "/data3/yss/sft_datas/0817/sft_data_merge_v17_quality_test.jsonl"
    save_jsonl(ds_train, train_tmp_path)
    save_jsonl(ds_test, test_tmp_path)
    ds_train = load_dataset(
        "json",
        data_files=train_tmp_path,
        split="train",
    )
    ds_test = load_dataset(
        "json",
        data_files=test_tmp_path,
        split="train",
    )

    dataset1 = ds_train.map(
        lambda batch: merge_train_data(batch, num_threshold=7500, seed=42),
        batched=True,
        num_proc=48,
        remove_columns=["nums"],
    )
    print("merge1 num:", len(dataset1))
    dataset2 = ds_train.map(
        lambda batch: merge_train_data(batch, num_threshold=7500, seed=0),
        batched=True,
        num_proc=48,
        remove_columns=["nums"],
    )
    print("merge2 num:", len(dataset2))

    ds_train = concatenate_datasets([dataset1, dataset2])
    print("final train dataset:", len(ds_train))
    print("final test dataset:", len(ds_test))

    dataset = concatenate_datasets([ds_train, ds_test])
    print("final dataset:", len(dataset))
    dataset.to_json(
        "/data3/yss/sft_datas/0817/sft_data_merge_v17_quality_all.jsonl",
        batch_size=1000,
        num_proc=48,
    )
