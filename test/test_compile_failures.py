import unittest
from tinygrad import Tensor
from tinygrad.engine.realize import lower_schedule

class TestCompileFailures(unittest.TestCase):
  def compile(self, out:Tensor):
    for _ in lower_schedule(out.schedule()): pass

  def test_interpolate_atari(self):
    self.compile(Tensor.empty(210, 160, dtype='uint8').interpolate((64, 64)))

if __name__ == '__main__':
  unittest.main()
