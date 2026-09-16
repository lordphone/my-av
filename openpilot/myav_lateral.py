# myav_lateral.py
# Runs the my-av lateral model inside openpilot's modeld.
# Copy to openpilot/selfdrive/modeld/myav_lateral.py in your openpilot fork.

import math
import numpy as np
from tinygrad.device import Buffer
from tinygrad.dtype import DType
from tinygrad.engine.realize import lower_and_compile
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp

from openpilot.selfdrive.modeld.helpers import MODELS_DIR, load_oob

STATE_NAMES = ('prev_img', 'hidden')


def input_view(buffer: Buffer, shape: tuple[int, ...], dtype: DType, offset: int) -> Tensor:
  view = buffer.view(math.prod(shape), dtype, offset).ensure_allocated()
  return Tensor(UOp.from_buffer(view)).reshape(shape)


class MyAvLateral:
  def __init__(self, path=MODELS_DIR / 'myav_tinygrad.pkl'):
    jits = load_oob(path)
    self.device = jits['input_specs']['new_img'][2]
    self.run_model = jits['run']
    self.run_model.captured._linear = lower_and_compile(self.run_model.captured._linear)

    self.states = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
                   for name, (shape, dtype, device) in jits['input_specs'].items() if name in STATE_NAMES}
    self.outputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
                    for name, (shape, dtype, device) in jits['output_specs'].items()}
    # next_X outputs write straight back into the X state buffers, same as modeld does
    for name, state in self.states.items():
      self.outputs[f'next_{name}'] = input_view(state._buffer(), state.shape, state.dtype, 0)

  def run(self, new_img: Tensor, v_ego: float, action_t: float) -> float:
    self.run_model(output_buffers=self.outputs,
                   new_img=new_img,
                   v_ego=Tensor(np.array([[v_ego]], dtype=np.float32), device=self.device).realize(),
                   action_t=Tensor(np.array([[action_t]], dtype=np.float32), device=self.device).realize(),
                   **self.states)
    return float(self.outputs['lat_accel'].numpy().reshape(-1)[0])

  def reset(self) -> None:
    for state in self.states.values():
      state.assign(0).realize()
