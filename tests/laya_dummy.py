"""Build a dummy Laya ONNX bundle with the real interface, for testing the bridge plumbing.

Same inputs/outputs as Laya's exported graph (input_ids, attention_mask, marker_pos, marker_mask,
qtype -> logits, act_probs), a WordLevel tokenizer with Laya's special tokens, and laya_config.json.
Its logits are -0.01 * marker position, so it always prefers the first option: useful only to
check that requests, tokenization and answer parsing work end to end.
"""

import json
from pathlib import Path


def build(dest: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer, models, pre_tokenizers

    dest.mkdir(parents=True, exist_ok=True)
    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["n", "L"]),
              helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["n", "L"]),
              helper.make_tensor_value_info("marker_pos", TensorProto.INT64, ["n", "K"]),
              helper.make_tensor_value_info("marker_mask", TensorProto.BOOL, ["n", "K"]),
              helper.make_tensor_value_info("qtype", TensorProto.INT64, ["n"])]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", "K"]),
               helper.make_tensor_value_info("act_probs", TensorProto.FLOAT, ["n", 1])]
    nodes = [helper.make_node("Cast", ["marker_pos"], ["pos_f"], to=TensorProto.FLOAT),
             helper.make_node("Constant", [], ["scale"], value=helper.make_tensor("s", TensorProto.FLOAT, [], [-0.01])),
             helper.make_node("Mul", ["pos_f", "scale"], ["logits"]),
             helper.make_node("ReduceMean", ["logits"], ["mean"], axes=[1], keepdims=1),
             helper.make_node("Sigmoid", ["mean"], ["act_probs"])]
    graph = helper.make_graph(nodes, "laya_dummy", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, dest / "laya.onnx")

    import string
    chars = [c for c in string.printable if not c.isspace()]
    pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    vocab = {t: i for i, t in enumerate(pieces)}
    tok = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    (dest / "tokenizer").mkdir(exist_ok=True)
    tok.save(str(dest / "tokenizer" / "tokenizer.json"))
    (dest / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 512}))
    (dest / "laya_config.json").write_text(json.dumps(
        {"max_len": 512, "head_max_len": 192, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}))
    return dest
