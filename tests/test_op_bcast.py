from nnsmith.abstract.op import *


def test_bcast():
    # bcast tests
    p0, p1, p2, p3, p4, p5 = z3.Ints('p0 p1 p2 p3 p4 p5')
    shapes = (2,), (3, 1), (1, 1, 1)
    shapes_var = [ShapeVar(s) for s in shapes]
    assert list(torch.broadcast_shapes(*shapes)
                ) == broadcast_shapes(*shapes_var).shape
    shapes = (p0,), (p1, 3), (1, 1, 1)
    shapes_var = [ShapeVar(s) for s in shapes]
    # print(broadcast_cons(*shapes_var))
    shapes = (p0,), (p1, 3), (1, 1, 2)
    shapes_var = [ShapeVar(s) for s in shapes]
    # print(broadcast_cons(*shapes_var))
    assert z3.is_false(z3.simplify(broadcast_cons(*shapes_var)[0]))

    for x1 in [p0, 1, 3]:
        for x2 in [p1, 1, 3]:
            shapes = (x1,), (x2,)
            shapes_var = [ShapeVar(s) for s in shapes]
            cons1 = broadcast_cons(*shapes_var)
            cons2 = broadcast_cons_2d(*shapes_var)
            s = z3.Solver()
            assert s.check(z3.And(*cons1) != z3.And(*cons2)) == z3.unsat


def test_bcast_add():
    # Add
    a = torch.randn(2, 1, 4, 5)
    b = torch.randn(3, 1, 5)
    c = a + b
    assert c.shape == torch.Size(Add().shape_fn(
        [ShapeVar(list(a.shape)), ShapeVar(list(b.shape))])[0].shape)

    i0, i1, i2, i3 = z3.Ints('i0 i1 i2 i3')
    ash = ShapeVar([i0, i1, 5])
    bsh = ShapeVar([3, i2, 1, i3])
    csh = Add().shape_fn([ash, bsh])[0]
    cons = Add()._requires([ash, bsh])
    cons.extend([i >= 1 for i in ash.shape])
    cons.extend([i >= 1 for i in bsh.shape])
    cons.extend([i >= 1 for i in csh.shape])
    s = z3.Solver()
    s.add(*cons)
    assert s.check() == z3.sat
    # print(s.model())

    s.add(i1 > 3)
    assert s.check() == z3.sat
    # print(s.model())

    s.add(i3 > 3)
    assert s.check() == z3.sat
    # print(s.model())

    s.add(i3 > 5)
    assert s.check() == z3.unsat


test_bcast()
test_bcast_add()
