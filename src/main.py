from torch import Tensor
from src.data.solve import gaussian_elimination, repr_record

def test_gaussian_elimination():
    system = Tensor([
        [1, 2, 3, 4],
        [2, 1, 3, 7],
        [3, 2, 1, 6]
    ])
    system, record = gaussian_elimination(system)
    print(repr_record(record, ["x", "y", "z"]))


if __name__ == "__main__":
    test_gaussian_elimination()