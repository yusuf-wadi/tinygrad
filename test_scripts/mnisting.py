from tinygrad import Device
print(Device.DEFAULT)

from tinygrad import Tensor, nn

class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))

from tinygrad.nn.datasets import mnist
X_train, Y_train, X_test, Y_test = mnist()
print('mnist shape:\n', 'x_train: ', X_train.shape, X_train.dtype, '\n y_train: ', Y_train.shape, Y_train.dtype, '\n')
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar

model = Model()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# NOTE: tinygrad is lazy, and hasn't actually run anything by this point
print('random model acc: ', acc.item())  # ~10% accuracy, as expected from a random model
print(nn.state.get_parameters(model))
optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
  Tensor.training = True  # makes dropout work
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss

import timeit
print('no jit: ', timeit.repeat(step, repeat=5, number=1))
#[0.08268719699981375,
# 0.07478952900009972,
# 0.07714716600003158,
# 0.07785399599970333,
# 0.07605237000007037]

from tinygrad import TinyJit
jit_step = TinyJit(step)

import timeit
print('with jit: ', timeit.repeat(jit_step, repeat=5, number=1))
# [0.2596786549997887,
#  0.08989566299987928,
#  0.0012115650001760514,
#  0.001010227999813651,
#  0.0012164899999334011]
print('jit speedup: ', 0.077/0.001)

print('training...')
for step in range(7000):
  loss = jit_step()
  if step%100 == 0:
    Tensor.training = False
    acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
    print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
