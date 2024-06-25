import dspy
import pandas as pd


def load_csv(file_path: str, train_size: float):
    data = pd.read_csv(file_path)
    data = data.to_dict(orient="records")

    trainset = data[: int(len(data) * train_size)]
    trainset = [
        dspy.Example(question=d["question"], answer=d["answer"]).with_inputs("question")
        for d in trainset
    ]
    devset = data[int(len(data) * train_size) :]
    devset = [
        dspy.Example(question=d["question"], answer=d["answer"]).with_inputs("question")
        for d in devset
    ]

    return trainset, devset
