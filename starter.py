"""Compatibility CLI. Prefer: python -m math_tutor <command>."""
from math_tutor.common import *
from math_tutor.data.prepare import expression, family, convert, prepare, audit
from math_tutor.data.download import download_data, download_model
from math_tutor.utils.environment import check
from math_tutor.inference.runtime import load_inference, generate, chat
from math_tutor.evaluation.runner import extract_answer, evaluate, rescore
from math_tutor.training.launcher import launch
from math_tutor.cli import main
if __name__ == "__main__":
    main()
