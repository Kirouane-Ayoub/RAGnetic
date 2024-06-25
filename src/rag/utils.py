import dspy


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="detailed explanation")


def validate_context_and_answer(example, pred, trace=None):
    answer_em = dspy.evaluate.answer_exact_match(example, pred)
    answer_pm = dspy.evaluate.answer_passage_match(example, pred)
    return answer_em and answer_pm
