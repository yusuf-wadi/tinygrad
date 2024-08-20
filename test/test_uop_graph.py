import unittest
from test.helpers import TestUOps
from tinygrad import dtypes, Variable
from tinygrad.dtype import PtrDType
from tinygrad.helpers import DEBUG
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps, ReduceOps, UOps, UOp, NOp, PatternMatcher
from tinygrad.codegen.uopgraph import linearize_uop, graph_rewrite, expander, reducer, constant_folder, float4_folding

simple_pm = PatternMatcher([
  (NOp.cvar('x', dtypes.int), lambda x: UOp.const(dtypes.float, 1.0) + UOp.const(dtypes.float, 2.0)),
  (NOp.cvar('x') + NOp.cvar('y'), lambda x,y: UOp.const(dtypes.float, x.arg+y.arg)),
  (NOp.cvar('x') * NOp.cvar('y') * NOp.cvar('z'), lambda x,y,z: UOp.const(dtypes.float, x.arg*y.arg*z.arg)),
  ((NOp.var('x') + NOp.cvar('c1')) + NOp.cvar('c2'), lambda x,c1,c2: x + x.const(c1.arg+c2.arg)),
])

class TestGraphRewrite(unittest.TestCase):
  def test_dedup(self):
    v1 = UOp(UOps.DEFINE_VAR, dtypes.float)
    v2 = UOp(UOps.DEFINE_VAR, dtypes.float)
    nout = graph_rewrite(v1+v2, PatternMatcher([]))
    self.assertIs(nout.src[0], nout.src[1])

  def test_simple(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    nout = graph_rewrite(c1+c2, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 3.0)

  def test_depth_2_late(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite(c1*c2*(c3+c3), simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 12.0)

  def test_double(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite(c1+c2+c3, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 6.0)

  def test_triple(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    c4 = UOp.const(dtypes.float, 4.0)
    nout = graph_rewrite(c1+c2+c3+c4, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 10.0)

  def test_diamond(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite((c1+c2)+(c1+c3), simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 7.0)

  def test_magic_4(self):
    c1 = UOp.const(dtypes.int, 4.0)
    nout = graph_rewrite(c1, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 3.0)

  def test_depth_2_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.float)
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    nout = graph_rewrite(v+c1+c2, simple_pm)
    self.assertEqual(nout.op, UOps.ALU)
    self.assertEqual(nout.src[0].op, UOps.DEFINE_VAR)
    self.assertEqual(nout.src[1].op, UOps.CONST)
    self.assertEqual(nout.src[1].arg, 3.0)

  def test_consts_go_last(self):
    a = UOp(UOps.DEFINE_VAR, dtypes.int, (UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1)), arg=Variable('a', 0, 1))
    b = UOp(UOps.DEFINE_VAR, dtypes.int, (UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1)), arg=Variable('b', 0, 1))
    c = UOp(UOps.DEFINE_VAR, dtypes.int, (UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1)), arg=Variable('c', 0, 1))
    d = UOp(UOps.DEFINE_VAR, dtypes.int, (UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1)), arg=Variable('d', 0, 1))
    outs = [2+a, 2+a+d+3+b+c+4, UOp(UOps.ALU, a.dtype, src=(a.const(2), a), arg=BinaryOps.ADD), (4+d)+c+(2+a)+b]
    for out in outs:
      sink = graph_rewrite(out, constant_folder)
      print(sink)
      self.assertEqual(sink.op, UOps.ALU)
      self.assertEqual(sink.src[1].op, UOps.CONST)
      self.assertEqual(len([x for x in sink.sparents if x.op is UOps.CONST]), 3)

class TestUOpGraph(TestUOps):
  def test_add_constant_fold(self):
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    uops = linearize_uop([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.int, (UOp(UOps.CONST, dtypes.int, (), 0), UOp(UOps.CONST, dtypes.int, (), 1)), arg=Variable('tmp', 0, 1))
    c0 = UOp(UOps.CONST, dtypes.int, arg=0)
    vc = UOp(UOps.ALU, dtypes.bool, (v, c0), BinaryOps.CMPNE)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    out = UOp(UOps.ALU, dtypes.float, (vc, c1, c1), TernaryOps.WHERE)
    uops = linearize_uop([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (bf, c1, c2), TernaryOps.WHERE)
    uops = linearize_uop([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    out = UOp(UOps.CAST, dtypes.int, (bf,))
    uops = linearize_uop([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 0)

  def test_noop_vectorize_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(UOps.LOAD, dtypes.float.vec(2), (d0, idx))
    vec = UOp(UOps.VECTORIZE, dtypes.float.vec(2), (ld,))
    x = UOp(UOps.GEP, dtypes.float, (vec, ), arg=0)
    alu = UOp(UOps.ALU, dtypes.float, (x, ), UnaryOps.SQRT)
    out = UOp(UOps.STORE, None, (d0, idx, alu))
    uops = linearize_uop([out])
    self.assertEqual(len([x for x in uops if x.op is UOps.VECTORIZE]), 0)

  def test_gep_vec_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    d1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 1)
    d2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 2)
    idx = UOp.const(dtypes.int, 0)
    def _test_vec(geps, count=4):
      vec = UOp(UOps.VECTORIZE, dtypes.float.vec(count), geps)
      out = UOp(UOps.STORE, None, (d0, idx, vec))
      uops = linearize_uop([out])
      if DEBUG >= 4:
        from tinygrad import Device
        print(Device[Device.DEFAULT].renderer.render("test", uops))
      return uops[-1].src[-1]

    # possible
    val = UOp(UOps.LOAD, dtypes.float.vec(4), (d1, idx))
    xyzw = tuple(UOp(UOps.GEP, dtypes.float, (val,), i) for i in range(4))
    self.assert_equiv_uops(_test_vec(xyzw), val)

    # unaligned
    val = UOp(UOps.LOAD, dtypes.float.vec(4), (d1, idx))
    wzyx = tuple(UOp(UOps.GEP, dtypes.float, (val,), i) for i in reversed(range(4)))
    self.assertIs(_test_vec(wzyx).op, UOps.VECTORIZE)

    # different_size
    val = UOp(UOps.LOAD, dtypes.float.vec(2), (d1, idx))
    xy = tuple(UOp(UOps.GEP, dtypes.float, (val, ), i) for i in range(2))
    self.assertIs(_test_vec(xy+xy).op, UOps.VECTORIZE)
    val = UOp(UOps.LOAD, dtypes.float.vec(4), (d1, idx))
    xy = tuple(UOp(UOps.GEP, dtypes.float, (val, ), i) for i in range(2))
    self.assertIs(_test_vec(xy, count=2).op, UOps.VECTORIZE)

    # different vals
    val1 = UOp(UOps.LOAD, dtypes.float.vec(2), (d1, idx))
    val2 = UOp(UOps.LOAD, dtypes.float.vec(2), (d2, idx))
    xy1 = tuple(UOp(UOps.GEP, dtypes.float, (val1, ), i) for i in range(2))
    xy2 = tuple(UOp(UOps.GEP, dtypes.float, (val2, ), i) for i in range(2))
    self.assertIs(_test_vec(xy1+xy2).op, UOps.VECTORIZE)

  def test_gep_vec_const_fold(self):
    for vec_size in [2, 4, 8]:
      consts = [UOp.const(dtypes.float, float(i)) for i in range(vec_size)]
      vec = UOp(UOps.VECTORIZE, dtypes.float.vec(vec_size), tuple(consts))
      geps = [UOp(UOps.GEP, dtypes.float, (vec,), i) for i in range(vec_size)]
      uops = linearize_uop(geps)
      for uop, const in zip(uops, consts):
        self.assert_equiv_uops(uop, const)

  def test_wmma_vectorize_fold(self):
    for i in [2, 4, 8]:
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i), tuple(UOp.const(dtypes.half, 0.0) for _ in range(i)))
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable('acc', 0.0, 1.0))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = linearize_uop([wmma])
      self.assert_equiv_uops(uops[0], acc)
      self.assertEqual(len(uops), 1)

    for i in [2, 4, 8]:
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i))
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i), tuple(UOp.const(dtypes.half, 0.0) for _ in range(i)))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable('acc', 0.0, 1.0))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = linearize_uop([wmma])
      self.assert_equiv_uops(uops[0], acc)
      self.assertEqual(len(uops), 1)

  def test_wmma_vectorize_no_fold(self):
    for i in [4, 8]:
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 0.0) for _ in range(i//2)) +
                tuple(UOp(UOps.DEFINE_VAR, dtypes.half, arg=Variable(f'tmp{j}', 0.0, 1.0)) for j in range(i//2)))
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable(f'tmp{i}', 0.0, 1.0))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable('acc', 0.0, 1.0))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = linearize_uop([wmma])
      self.assert_equiv_uops(uops[-1], wmma)

    for i in [4, 8]:
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable(f'tmp{i}', 0.0, 1.0))
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 0.0) for _ in range(i//2)) +
                tuple(UOp(UOps.DEFINE_VAR, dtypes.half, arg=Variable(f'tmp{j}', 0.0, 1.0)) for j in range(i//2)))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable('acc', 0.0, 1.0))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = linearize_uop([wmma])
      self.assert_equiv_uops(uops[-1], wmma)

    for i in [2, 4, 8]:
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable(f'tmp{i}', 0.0, 1.0))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable('acc', 0.0, 1.0))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = linearize_uop([wmma])
      self.assert_equiv_uops(uops[-1], wmma)

    for i in [2, 4, 8]:
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable(f'tmp{i}', 0.0, 1.0))
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=Variable('acc', 0.0, 1.0))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = linearize_uop([wmma])
      self.assert_equiv_uops(uops[-1], wmma)

  def test_cast_alu_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.bool), arg=0)
    d1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(UOps.LOAD, dtypes.int, (d1, idx))
    alu = ld.lt(1).cast(dtypes.bool)
    out = UOp(UOps.STORE, None, (d0, idx, alu))
    uops = linearize_uop([out])
    self.assertEqual(len([x for x in uops if x.op is UOps.CAST]), 0)

  def test_double_cast_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0)
    d1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(UOps.LOAD, dtypes.int, (d1, idx))
    alu = ld.cast(dtypes.float).cast(dtypes.float)
    out = UOp(UOps.STORE, None, (d0, idx, alu))
    uops = linearize_uop([out])
    self.assertEqual(len([x for x in uops if x.op is UOps.CAST]), 1)

  def test_depth_2_const_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.int, (UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1)), arg=Variable('tmp', 0, 1))
    c2 = UOp(UOps.CONST, dtypes.int, arg=2)
    c4 = UOp(UOps.CONST, dtypes.int, arg=4)
    vc = UOp(UOps.ALU, dtypes.int, (v, c2), BinaryOps.ADD)
    out = UOp(UOps.ALU, dtypes.int, (vc, c4), BinaryOps.ADD)
    uops = linearize_uop([out])
    self.assertEqual(len(uops), 5)
    out = uops[-1]
    self.assertEqual(out.op, UOps.ALU)
    self.assertEqual(out.arg, BinaryOps.ADD)
    self.assertEqual(out.src[1].op, UOps.CONST)
    self.assertEqual(out.src[1].arg, 6)

  def test_fold_gated_load(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    glbl1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 1)
    glbl2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 2)
    idx = UOp.const(dtypes.int, 0)
    ld0 = UOp(UOps.LOAD, dtypes.int, (glbl1, idx, UOp.const(dtypes.int, 2), UOp.const(dtypes.bool, False)))
    ld1 = UOp(UOps.LOAD, dtypes.int, (glbl2, idx, UOp.const(dtypes.int, 3), UOp.const(dtypes.bool, True)))
    uops = linearize_uop([UOp(UOps.STORE, None, (glbl0, idx, ld1+ld0))])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    self.assert_equiv_uops(ld1, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    self.assert_equiv_uops(ld0, UOp.load(glbl2, idx, dtype=dtypes.int))

  def test_fold_gated_load_local(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    smem = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.int), (), ("temp", 1))
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 16))
    st = UOp(UOps.STORE, None, (smem, lidx, UOp.load(glbl0, lidx, dtype=dtypes.int)))
    barrier = UOp(UOps.BARRIER, None, (st, ))
    ld0 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+1, UOp.const(dtypes.int, 2), UOp.const(dtypes.bool, False), barrier))
    ld1 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+2, UOp.const(dtypes.int, 3), UOp.const(dtypes.bool, True), barrier))
    uops = linearize_uop([UOp(UOps.STORE, None, (glbl0, lidx, ld1+ld0))])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    self.assert_equiv_uops(ld1, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    self.assert_equiv_uops(ld0, UOp.load(smem, lidx+2, barrier, dtype=dtypes.int))

  def test_fold_gated_store(self):
    glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 0)
    val = UOp.const(dtypes.int, 42)
    st0 = UOp(UOps.STORE, None, (glbl, idx0, val, UOp.const(dtypes.bool, False)))
    st1 = UOp(UOps.STORE, None, (glbl, idx1, val, UOp.const(dtypes.bool, True)))
    uops = linearize_uop([st0, st1])
    # only the second store happens
    self.assertEqual(len(uops), 4)
    self.assert_equiv_uops(uops[-1], UOp.store(glbl, idx1, val))

  def test_asserts_bad_gate(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    idx = UOp.const(dtypes.int, 0)
    bad_gate = UOp.const(dtypes.int, 1)
    with self.assertRaises(AssertionError): linearize_uop([UOp(UOps.STORE, None, (glbl0, idx, UOp.const(dtypes.int, 42), bad_gate))])

  def test_switched_range_order(self):
    glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    c0 = UOp.const(dtypes.int, 0)
    c2 = UOp.const(dtypes.int, 2)
    cf = UOp.const(dtypes.float, 0.0)
    r1 = UOp(UOps.RANGE, dtypes.int, (c0, c2), (1, 0, False))
    r2 = UOp(UOps.RANGE, dtypes.int, (c0, c2), (1, 1, False))
    alu = UOp(UOps.ALU, dtypes.int, (r2, r1), BinaryOps.MUL)
    store = UOp(UOps.STORE, None, (glbl, alu, cf))
    uops = linearize_uop([store])
    ranges = [x for x in uops if x.op is UOps.RANGE]
    endranges = [x for x in uops if x.op is UOps.ENDRANGE]
    # ranges are closed in the right order
    self.assertEqual(endranges[-1].src[0], ranges[0])

def expander_rewrite(sink): return graph_rewrite(sink, constant_folder + expander + reducer)
def float4_rewrite(sink): return graph_rewrite(sink, constant_folder + expander + float4_folding)

class TestExpander(unittest.TestCase):
  def test_expand_add_broadcast(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    sink = expander_rewrite(e1+3)
    assert sink.op is UOps.EXPAND and len(sink.src) == 4
    self.assertListEqual([x.arg for x in sink.src], [3,4,5,6])

  def test_contract_simple(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((1,4),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.VECTORIZE and len(sink.src) == 4
    self.assertListEqual([x.arg for x in sink.src], [0,1,2,3])

  def test_contract_axis_1(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(16)), ((1,4),(2,4)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((1,4),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and len(sink.src) == 4 and sink.arg == ((2,4),)
    assert sink.src[0].op is UOps.VECTORIZE and len(sink.src[0].src) == 4
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,4,8,12])
    self.assertListEqual([x.arg for x in sink.src[3].src], [3,7,11,15])

  def test_contract_axis_2(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(16)), ((1,4),(2,4)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((2,4),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and len(sink.src) == 4 and sink.arg == ((1,4),)
    assert sink.src[0].op is UOps.VECTORIZE and len(sink.src[0].src) == 4
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,1,2,3])
    self.assertListEqual([x.arg for x in sink.src[3].src], [12,13,14,15])

  def test_contract_axis_2_big(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(16)), ((1,2),(2,2),(3,2),(4,2)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and sink.arg == ((1, 2), (3, 2), (4, 2))
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,4])
    self.assertListEqual([x.arg for x in sink.src[6].src], [10,14])

  def test_contract_multi_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(16)), ((1,2),(2,2),(3,2),(4,2)))
    sink = expander_rewrite(UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((3,2),(2,2))))
    assert sink.op is UOps.EXPAND and sink.arg == ((1, 2), (4, 2))
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,4,2,6])
    sink = expander_rewrite(UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((2,2),(3,2))))
    assert sink.op is UOps.EXPAND and sink.arg == ((1, 2), (4, 2))
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,2,4,6])

  def test_contract_mid(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(8)), ((1,2),(2,2),(3,2)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and len(sink.src) == 4 and sink.arg == ((1,2),(3,2))
    assert sink.src[0].op is UOps.VECTORIZE and len(sink.src[0].src) == 2
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,2])
    self.assertListEqual([x.arg for x in sink.src[1].src], [1,3])
    self.assertListEqual([x.arg for x in sink.src[2].src], [4,6])
    self.assertListEqual([x.arg for x in sink.src[3].src], [5,7])

  def test_contract_no_expand(self):
    e1 = UOp(UOps.DEFINE_VAR, dtypes.int)
    con = UOp(UOps.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.VECTORIZE and len(sink.src) == 2
    assert sink.src[0] == sink.src[1]

  def test_contract_half_expand(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(8), (e1,), ((1,4), (2,2)))
    sink = expander_rewrite(con)
    assert sink.op is UOps.VECTORIZE and len(sink.src) == 8
    assert sink.src[0] == sink.src[1]
    assert sink.src[0] != sink.src[2]
    assert sink.src[6] == sink.src[7]

  def test_expand_same_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4*x) for x in range(4)), ((1,4),))
    sink = expander_rewrite(e1+e2)
    assert sink.op is UOps.EXPAND and len(sink.src) == 4
    self.assertListEqual([x.arg for x in sink.src], [0,5,10,15])

  def test_expand_different_axis(self, flip=False):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4*x) for x in range(4)), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((2,4),))
    sink = expander_rewrite((e2+e1) if flip else (e1+e2))
    assert sink.op is UOps.EXPAND and len(sink.src) == 16
    assert sink.arg == ((1, 4), (2, 4))
    self.assertListEqual([x.arg for x in sink.src], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

  def test_expand_different_axis_flip(self): self.test_expand_different_axis(True)

  @unittest.skip("no longer supported")
  def test_reduce_known_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    sink = UOp(UOps.REDUCE, dtypes.int, (3*e1,e1), ReduceOps.SUM)
    sink = expander_rewrite(sink)
    assert sink.op is UOps.CONST
    self.assertEqual(sink.arg, 3*(0+1+2+3))

  @unittest.skip("no longer supported")
  def test_reduce_const(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    sink = UOp(UOps.REDUCE, dtypes.int, (UOp.const(dtypes.int, 3), e1), ReduceOps.SUM)
    sink = expander_rewrite(sink)
    assert sink.op is UOps.CONST
    self.assertEqual(sink.arg, 3*4)

  def test_double_expand(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((2,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((2,4),))
    e = UOp(UOps.EXPAND, dtypes.int, (e1, e2), ((1,2),))
    sink = expander_rewrite(e)
    assert sink.op is UOps.EXPAND and len(sink.src) == 8
    assert sink.arg == ((1, 2), (2, 4))
    self.assertListEqual([x.arg for x in sink.src], [0,1,2,3,4,5,6,7])

  def test_double_expand_reverse(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((1,4),))
    e = UOp(UOps.EXPAND, dtypes.int, (e1, e2), ((2,2),))
    sink = expander_rewrite(e)
    assert sink.op is UOps.EXPAND and len(sink.src) == 8
    assert sink.arg == ((1, 4), (2, 2))
    self.assertListEqual([x.arg for x in sink.src], [0, 4, 1, 5, 2, 6, 3, 7])

  def test_double_expand_middle(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,2),(3,2)))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((1,2),(3,2)))
    e = UOp(UOps.EXPAND, dtypes.int, (e1, e2), ((2,2),))
    sink = expander_rewrite(e)
    assert sink.op is UOps.EXPAND and len(sink.src) == 8
    assert sink.arg == ((1, 2), (2, 2), (3, 2))
    self.assertListEqual([x.arg for x in sink.src], [0, 1, 4, 5, 2, 3, 6, 7])

  # does this need to work?
  @unittest.expectedFailure
  @unittest.skip
  def test_reduce_different_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((2,4),))
    sink = UOp(UOps.REDUCE, dtypes.int, (e1,e2), ReduceOps.SUM)
    sink = expander_rewrite(sink)
    print(sink)

class TestLoadStoreFolder(unittest.TestCase):
  def test_simple_load_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i))) for i in range(4)]
    sink = UOp(UOps.EXPAND, dtypes.float, tuple(load), ((0,4),))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 1

  def test_two_load_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i))) for i in range(8)]
    sink = UOp(UOps.EXPAND, dtypes.float, tuple(load), ((0,8),))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 2

  def test_simple_load_fold_gated(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp(UOps.DEFINE_VAR, dtypes.bool)
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate)) for i in range(4)]
    sink = UOp(UOps.EXPAND, dtypes.float, tuple(load), ((0,4),))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 1
    single_load = [x for x in sink.sparents if x.op is UOps.LOAD][0]
    self.assertListEqual([src.arg for src in single_load.src[2].src], [0.0, 1.0, 2.0, 3.0])

  def test_simple_load_dont_fold_different_gated(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp(UOps.DEFINE_VAR, dtypes.bool, arg="g1")
    gate2 = UOp(UOps.DEFINE_VAR, dtypes.bool, arg="g2")
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate if i == 0 else gate2)) for i in range(4)]
    sink = UOp(UOps.EXPAND, dtypes.float, tuple(load), ((0,4),))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 3

  def test_simple_store_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    load = [UOp(UOps.STORE, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i))) for i in range(4)]
    sink = UOp(UOps.SINK, None, tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.STORE]) == 1

  def test_simple_store_fold_gate(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp(UOps.DEFINE_VAR, dtypes.bool, arg="g1")
    load = [UOp(UOps.STORE, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate)) for i in range(4)]
    sink = UOp(UOps.SINK, None, tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.STORE]) == 1
    one_store = [x for x in sink.sparents if x.op is UOps.STORE][0]
    assert len(one_store.src) == 4
    assert str(one_store.src[3]) == str(gate)  # huh, why do i need str here?

  def test_simple_store_dont_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp(UOps.DEFINE_VAR, dtypes.bool, arg="g1")
    gate2 = UOp(UOps.DEFINE_VAR, dtypes.bool, arg="g2")
    load = [UOp(UOps.STORE, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate if i == 0 else gate2)) for i in range(4)]
    sink = UOp(UOps.SINK, None, tuple(load))
    sink = float4_rewrite(sink)
    print(sink)
    assert len([x for x in sink.sparents if x.op is UOps.STORE]) == 3

def gate_rewrite(sink): return graph_rewrite(sink, constant_folder + expander + reducer)

class TestIFUOps(TestUOps):
  def test_create_ifs(self):
    gbuf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    sbuf = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.float), (), ("smem", 4))
    valid = UOp(UOps.SPECIAL, dtypes.int, (), ("gidx0", 10)).lt(5)
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 4))
    gate = valid&(lidx.ne(2))
    idx = UOp.const(dtypes.int, 0)
    st = UOp(UOps.STORE, None, (sbuf, idx, UOp.const(dtypes.float, 42)))
    barrier = UOp(UOps.BARRIER, None, (st,))
    lbuf = UOp(UOps.LOAD, dtypes.float, (sbuf, UOp.const(dtypes.int, 0), barrier))
    store = UOp(UOps.STORE, None, (gbuf, UOp.const(dtypes.int, 0), lbuf, gate))
    sink = UOp(UOps.SINK, None, (store,))
    sink = gate_rewrite(sink)
    if_uops = [u for u in sink.parents if u.op is UOps.IF]
    self.assertEqual(len(if_uops), 1)
    self.assert_equiv_uops(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 3)

  def test_expand_ifs_one_gate(self):
    gbuf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    sbuf = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.float), (), ("smem", 16))
    valid = UOp(UOps.SPECIAL, dtypes.int, (), ("gidx0", 4)).lt(1)
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 16))
    gate = valid&(lidx.ne(2))
    st = UOp(UOps.STORE, None, (sbuf, lidx, UOp.const(dtypes.float, 42)))
    barrier = UOp(UOps.BARRIER, None, (st,))
    lbufs = [UOp(UOps.LOAD, dtypes.float, (sbuf, UOp.const(dtypes.int, i), barrier)) for i in range(4)]
    stores = [UOp(UOps.STORE, None, (gbuf, UOp.const(dtypes.int, i), lbufs[i], gate)) for i in range(4)]
    sink = UOp(UOps.SINK, None, tuple(stores))
    sink = gate_rewrite(sink)
    if_uops = [u for u in sink.parents if u.op is UOps.IF]
    self.assertEqual(len(if_uops), 1)
    self.assert_equiv_uops(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 3)

  # this will be fixed with the merge gated stores bounty
  @unittest.expectedFailure
  def test_expand_ifs_dumb(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    valid = UOp(UOps.SPECIAL, dtypes.int, (), ("gidx0", 10)).lt(5)
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 4))
    gate = valid&(lidx.ne(2))
    stores = [UOp(UOps.STORE, None, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate)) for i in range(4)]
    sink = UOp(UOps.SINK, None, tuple(stores))
    sink = gate_rewrite(sink)
    if_uops = [u for u in sink.parents if u.op is UOps.IF]
    self.assertEqual(len(if_uops), 1)
    self.assert_equiv_uops(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 3)


if __name__ == '__main__':
  unittest.main(verbosity=2)
