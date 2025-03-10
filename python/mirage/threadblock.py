from .core import *

class TBGraph:
    def __init__(self, graph):
        self.cygraph = graph
        self.use_nvshmem = False

    def new_input(self, dtensor: DTensor, input_map: tuple, forloop_dim: int):
        return self.cygraph.new_input(dtensor, input_map, forloop_dim)

    def new_output(self, stensor: STensor, output_map: tuple, forloop_dim: int = -1, epilogue: str = None):
        if epilogue == "allreduce" or epilogue == "alltoall":
            self.use_nvshmem = True
        return self.cygraph.new_output(stensor, output_map, forloop_dim, epilogue)

    def matmul(self, A: STensor, B: STensor):
        return self.cygraph.matmul(A, B)

    def exp(self, A: STensor):
        return self.cygraph.exp(A)

    def silu(self, A: STensor):
        return self.cygraph.silu(A)

    def gelu(self, A: STensor):
        return self.cygraph.gelu(A)
    
    def relu(self, A: STensor):
        return self.cygraph.relu(A)
    
    def clamp(self, A: STensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)

    def square(self, A: STensor):
        return self.cygraph.square(A)

    def sqrt(self, A: STensor):
        return self.cygraph.sqrt(A)

    def add(self, A: STensor, B: STensor):
        return self.cygraph.add(A, B)

    def mul(self, A: STensor, B: STensor):
        return self.cygraph.mul(A, B)

    def div(self, A: STensor, B: STensor):
        return self.cygraph.div(A, B)

    def reduction(self, A: STensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def rms_norm(self, A: STensor):
        return self.cygraph.rms_norm(A)

    def concat(self, A: STensor, B: STensor, dim: int):
        return self.cygraph.concat(A, B, dim)

    def forloop_accum(self, A: STensor, acc: str = None):
        return self.cygraph.forloop_accum(A, acc)
