def dot(x: list[float], y: list[float]) -> float:
    assert len(x) == len(y)
    return sum(x_i * y_i for x_i, y_i in zip(x,y))