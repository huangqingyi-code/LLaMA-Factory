from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from datasets import Dataset, concatenate_datasets, Features, Sequence, Value
import json

model_path = "/data4/models/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def split_by_source(dataset: Dataset, test_size=0.02, num_threshold=8000):
    datas = {}
    for data in tqdm(dataset):
        sr = data["source"]
        if not sr in datas:
            datas[sr] = {
                "instruction": [],
                "input": [],
                "output": [],
                "system": [],
                "history": [],
                "source": [],
                "nums": [],
            }
        # if data["nums"] < num_threshold and data["source"] in [
        #     "multi_turn_datas_0808.json",
        #     "table_python_code_datas.json",
        # ]:
        if data["nums"] < num_threshold:
            datas[sr]["instruction"].append(data["instruction"])
            datas[sr]["input"].append(data["input"])
            datas[sr]["output"].append(data["output"])
            datas[sr]["system"].append(data["system"])
            datas[sr]["history"].append(data["history"])
            datas[sr]["source"].append(data["source"])
            datas[sr]["nums"].append(data["nums"])
    ds = {}
    features = Features(
        {
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
            "system": Value("string"),
            "history": Sequence(Sequence(Value("string"))),
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
        system = examples["system"][i]
        messages = []
        if len(system):
            messages.append({"role": "system", "content": system})
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
        system = examples["system"][i]
        if nums > num_threshold:
            continue
        if nums > num_threshold // 2 or len(history) or len(system):
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
    data_file = (
        "/data4/yss/sft_datas/1016/sft_data_merge_v20_quality_filtered_trans.jsonl"
    )
    train_save_path = (
        "/data4/yss/sft_datas/1016/sft_data_merge_v20_quality_filtered_train.jsonl"
    )
    test_save_path = (
        "/data4/yss/sft_datas/1016/sft_data_merge_v20_quality_filtered_test.jsonl"
    )
    all_save_path = (
        "/data4/yss/sft_datas/1016/sft_data_merge_v20_quality_filtered_all.jsonl"
    )

    dataset = load_dataset(
        "json",
        data_files=data_file,
        split="train",
    )

    print("ori num:", len(dataset))
    # 1.计算样本token长度
    dataset = dataset.map(calculate_token_num, batched=True, num_proc=48)

    # 2.根据source划分train和test以及根据token长度过滤数据
    ds_train, ds_test = split_by_source(dataset, num_threshold=7800)
    print("train:", len(ds_train), "test:", len(ds_test))

    # TODO split_by_source处理后的dataset后续处理非常慢，先保存再load出来
    train_tmp_path = train_save_path
    test_tmp_path = test_save_path

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

    # 将2个epoch短样本数据分别合并在一起
    dataset1 = ds_train.map(
        lambda batch: merge_train_data(batch, num_threshold=7800, seed=42),
        batched=True,
        num_proc=48,
        remove_columns=["nums"],
    )
    print("merge1 num:", len(dataset1))
    dataset2 = ds_train.map(
        lambda batch: merge_train_data(batch, num_threshold=7800, seed=0),
        batched=True,
        num_proc=48,
        remove_columns=["nums"],
    )
    print("merge2 num:", len(dataset2))

    # 两个epoch的数据合并在一起
    ds_train = concatenate_datasets([dataset1, dataset2])
    print("final train dataset:", len(ds_train))
    print("final test dataset:", len(ds_test))

    dataset = concatenate_datasets([ds_train, ds_test])
    print("final dataset:", len(dataset))
    dataset.to_json(
        all_save_path,
        batch_size=1000,
        num_proc=48,
    )
    ds_train.to_json(
        train_save_path,
        batch_size=1000,
        num_proc=48,
    )
    ds_test.to_json(
        test_save_path,
        batch_size=1000,
        num_proc=48,
    )
