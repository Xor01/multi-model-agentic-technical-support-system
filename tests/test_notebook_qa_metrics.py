"""Exercise the Model B notebook's span scoring without a Colab runtime."""

import ast
from collections import Counter
import json
from pathlib import Path
import re
import string
import unittest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "src/support_agent/models/qa_model/Model_B_—_Technical_Extractive_QA.ipynb"
)


class FakeEncoding(dict):
    def sequence_ids(self, index):
        return [None, 0, None, 1, 1, None]


class FakeTokenizer:
    def __call__(self, *args, **kwargs):
        if "add_special_tokens" in kwargs:
            return {"input_ids": [1, 2, 3]}
        return FakeEncoding(
            overflow_to_sample_mapping=[0],
            offset_mapping=[[(0, 0), (0, 0), (0, 0), (4, 17), (18, 24), (0, 0)]],
        )


class FakeTrainer:
    def predict(self, features):
        self.seen_features = features
        return type("Output", (), {"predictions": (
            [[0, 100, 0, 10, 0, 0]],
            [[0, 100, 0, 10, 0, 0]],
        )})()


class NotebookQAMetricsTests(unittest.TestCase):
    def test_scores_context_span_without_question_answering_pipeline(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        tree = ast.parse(source)
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"qa_tokens", "score_qa_model"}
        ]
        namespace = {
            "Counter": Counter, "re": re, "string": string,
            "tokenizer_b": FakeTokenizer(), "MAX_LENGTH": 384, "DOC_STRIDE": 96,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        trainer = FakeTrainer()
        result = namespace["score_qa_model"](
            trainer,
            [{"question": "Which header?", "context": "Use Authorization header", "answer_text": "Authorization"}],
            ["feature"],
        )
        self.assertIs(trainer.seen_features[0], "feature")
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["token_f1"], 1.0)
        self.assertEqual(result["long_context_errors"], [])


if __name__ == "__main__":
    unittest.main()
