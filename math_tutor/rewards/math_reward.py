"""Verifiable outcome reward; no reward for formatting alone."""
from math_tutor.evaluation.scoring import score_response

def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return ''.join(message['content'] for message in completion)

def correctness_reward(completions, answer, question, **kwargs):
    if not (len(completions) == len(answer) == len(question)):
        raise ValueError('Reward batch lengths differ')
    return [float(score_response(q, completion_text(c), a)['correct'] is True)
            for c, a, q in zip(completions, answer, question)]
