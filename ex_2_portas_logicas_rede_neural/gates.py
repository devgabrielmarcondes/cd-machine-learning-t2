from linear_algebra.operations import dot

class Neuronio:
    def __init__(self, weights: list[float] = [1,1], bias: float = 0, theta: float = 1):
        self.theta = theta
        self.weights = weights
        self.bias = bias

    def step_function(self, x: float) -> int:
        return 1 if x >= self.theta else 0

    def neuronio_MCP(self, x: float) -> int:
        vk = dot(self.weights, x) + self.bias
        return self.step_function(vk)
    
    def test_inputs(self, inputs: list[list[float]]) -> list[int]:
        return [self.neuronio_MCP(x) for x in inputs]


# Saída formatada
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

 
andGate = Neuronio(theta=1.5)
andGateResult = andGate.test_inputs(inputs)

nandGate = Neuronio(weights=[-1, -1], theta=-1)
nandGateResult = nandGate.test_inputs(inputs)

orGate = Neuronio(theta=1)
orGateResult = orGate.test_inputs(inputs)

norGate = Neuronio(weights=[-1, -1], theta=0)
norGateResult = norGate.test_inputs(inputs)

for x, r in zip(inputs, andGateResult):
    print(f"{x[0]} AND {x[1]} = {r}")
print()
for x, r in zip(inputs, orGateResult):
    print(f"{x[0]} OR  {x[1]} = {r}")
print()
for x, r in zip(inputs, nandGateResult):
    print(f"{x[0]} NAND {x[1]} = {r}")
print()
for x, r in zip(inputs, norGateResult):
    print(f"{x[0]} NOR  {x[1]} = {r}")