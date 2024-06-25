import dspy
from dspy.teleprompt import BootstrapFewShot
from rag.utils import GenerateAnswer, validate_context_and_answer
from utils import settings
from utils.csv_loder import load_csv
from utils.models import cohere_llm
from utils.retriver_model import retriever_model

dspy.settings.configure(lm=cohere_llm, rm=retriever_model)


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


print("------ Load Train / Dev Sets ------ ")

train, dev = load_csv(file_path=settings.CSV_SET_PATH, train_size=settings.TRAIN_SIZE)

print("------ Data Loaded ------ ")


print("------ Start RAG Compiler ------ ")

teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=train)

print("------ We are Good to Go !!------ ")
