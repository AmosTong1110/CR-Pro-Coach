"""Write the tiny test fixture tests/fixtures/crbot_il_dummy/crbot_il_v2.{onnx,json}.

Same inputs / outputs as a real crbot_il_v2 model (obs [B,1752] il_obs_v2 -> gate_logit [B], slot_logits [B,4],
tile_logits [B,4,576], value [B]) but hand-set weights, so NeuralBot can be tested with onnxruntime without a
trained model:
  * gate   = 10 * own_elixir_fraction - 3   (plays once elixir > 3)
  * slots  = constant preference for slot 3 > 2 > 1 > 0
  * tiles  = -distance to tile (gx 3, gy 20) in the acting side's frame: the favourite tile is on the enemy side,
             so for troops the legal-tile mask must move the play to the nearest own-half tile
  * value  = 0.5
Needs the `onnx` package (not an app dependency):  pip install onnx && python tools/make_dummy_il_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend_engine.brain import obs_encoder as F

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "crbot_il_dummy"


def build(out: Path = OUT) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    dim = F.OBS_DIM
    gy, gx = np.divmod(np.arange(F.NUM_TILES), F.GRID_W)
    pref = -np.sqrt((gx - 3) ** 2 + (gy - 20) ** 2).astype(np.float32)
    inits = [
        numpy_helper.from_array(np.array([0, 7], np.int64), "elx_start"),
        numpy_helper.from_array(np.array([1, 8], np.int64), "elx_end"),
        numpy_helper.from_array(np.array([10.0], np.float32), "ten"),
        numpy_helper.from_array(np.array([3.0], np.float32), "three"),
        numpy_helper.from_array(np.array([1], np.int64), "axis1"),
        numpy_helper.from_array(np.zeros((dim, 1), np.float32), "zero_w"),
        numpy_helper.from_array(np.array([[0.0, 0.1, 0.2, 0.3]], np.float32), "slot_pref"),
        numpy_helper.from_array(np.tile(pref, 4)[None, :], "tile_pref"),
        numpy_helper.from_array(np.array([-1, 4, F.NUM_TILES], np.int64), "tile_shape"),
    ]
    nodes = [
        helper.make_node("Slice", ["obs", "elx_start", "elx_end"], ["elx"]),          # obs[:, 7:8]
        helper.make_node("Mul", ["elx", "ten"], ["elx10"]),
        helper.make_node("Sub", ["elx10", "three"], ["gate2d"]),
        helper.make_node("Squeeze", ["gate2d", "axis1"], ["gate_logit"]),
        helper.make_node("MatMul", ["obs", "zero_w"], ["zero"]),                          # [B, 1] zeros
        helper.make_node("Add", ["zero", "slot_pref"], ["slot_logits"]),
        helper.make_node("Add", ["zero", "tile_pref"], ["tile_flat"]),
        helper.make_node("Reshape", ["tile_flat", "tile_shape"], ["tile_logits"]),
        helper.make_node("Sigmoid", ["zero"], ["value2d"]),
        helper.make_node("Squeeze", ["value2d", "axis1"], ["value"]),
    ]
    graph = helper.make_graph(
        nodes, "crbot_il_dummy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", dim])],
        [helper.make_tensor_value_info("gate_logit", TensorProto.FLOAT, ["batch"]),
         helper.make_tensor_value_info("slot_logits", TensorProto.FLOAT, ["batch", 4]),
         helper.make_tensor_value_info("tile_logits", TensorProto.FLOAT, ["batch", 4, F.NUM_TILES]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="crbot-test")
    model.ir_version = 8
    onnx.checker.check_model(model)
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out / "crbot_il_v2.onnx")
    meta = {"name": "crbot_il_v2", "dummy": True, **F.schema(), "gate_threshold": 0.5}
    (out / "crbot_il_v2.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    build()
