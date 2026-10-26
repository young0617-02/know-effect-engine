from types import SimpleNamespace

import numpy as np

from kee.ingest.mineru_onnx import Box, FormulaModel, MinerUPdfParser
from kee.ingest.parsers import ParsedFigure


class FakeSession:
    def __init__(self):
        self.shapes = []

    def get_inputs(self):
        return [SimpleNamespace(name="encoder/image")]

    def run(self, output_names, feeds):
        arr = feeds["encoder/image"]
        self.shapes.append(tuple(arr.shape))
        return [np.array([[0, 11, 12, 2]], dtype=np.int64)]


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "x^{2}+y^{2}=z^{2}"


def test_formula_preprocess_shape_and_range():
    bgr = np.full((80, 240, 3), 255, dtype=np.uint8)
    bgr[30:50, 40:120] = 0
    x = FormulaModel._preprocess(bgr)
    assert x.shape == (1, 1, 384, 384)
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_formula_model_decode_with_injected_session():
    sess = FakeSession()
    model = FormulaModel(session=sess, tokenizer=FakeTokenizer())
    bgr = np.full((80, 240, 3), 255, dtype=np.uint8)
    bgr[30:50, 40:120] = 0
    assert model.recognize(bgr) == "x^{2}+y^{2}=z^{2}"
    assert sess.shapes == [(1, 1, 384, 384)]


def test_formula_and_figure_items_become_paragraphs():
    pages = [[
        (Box(10, 10, 200, 40, 0.9, "formula", "E=mc^2"), "formula"),
        (Box(10, 60, 300, 120, 0.9, "figure", "【图表 第1页】图内文字：Q1 10"), "figure"),
    ]]
    notes = []
    doc = MinerUPdfParser._to_document(
        object(), pages, "论文.pdf", notes,
        figures=[ParsedFigure(kind="formula", page=1, text="E=mc^2")],
    )
    text = "\n".join(p for s in doc.sections for p in s.paragraphs)
    assert "【公式】$E=mc^2$" in text
    assert "【图表 第1页】" in text
    assert len(doc.figures) == 1
