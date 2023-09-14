from torch import Tensor
from typing import NamedTuple, List


# elementary row operations
def swap_rows(system: Tensor, i: int, j: int) -> Tensor:
    """Swaps rows i and j of the system of equations.
    
    Args:
        system (Tensor): system of equations
        i (int): row index
        j (int): row index
        
    Returns:
        Tensor: system of equations with swapped rows
    """
    system[[i, j]] = system[[j, i]]
    return system


def multiply_row(system: Tensor, i: int, k: float) -> Tensor:
    """Multiplies row i of the system of equations by k.
    
    Args:
        system (Tensor): system of equations
        i (int): row index
        k (float): scalar
        
    Returns:
        Tensor: system of equations with row i multiplied by k
    """
    system[i] *= k
    return system


def reduce_rows(system: Tensor, reduced_row_idx: int, other_row_idx: int, k: float) -> Tensor:
    """Adds k times row j to row i of the system of equations.
    
    Args:
        system (Tensor): system of equations
        reduced_row_idx (int): row index of row to be reduced
        other_row_idx (int): row index of row to be added
        k (float): scalar
        
    Returns:
        Tensor: system of equations with row i added by k times row j
    """
    system[reduced_row_idx] += k * system[other_row_idx]
    return system


def gaussian_elimination(system: Tensor) -> Tensor:
    """Transforms the system of equations to reduced row echelon form 
    by employing elementary row operations.
    
    Args:
        system (Tensor): system of equations
        
    Returns:
        Tensor: system of equations in reduced row echelon form
    """
    pass


class Operation:
    """Represents an operation performed on a system of equations."""
    _name: str

    def __init__(self, name: str):
        """Initializes the operation with a name.
        
        Args:
            name (str): name of the operation
        """
        self._name = name


class SwapRows(Operation):
    """Represents the operation of swapping two rows of a system of equations."""
    _i: int
    _j: int

    def __init__(self, i: int, j: int):
        """Initializes the operation with the row indices of the rows to be swapped.
        
        Args:
            i (int): row index
            j (int): row index
        """
        super().__init__("/swap_rows")
        self._i = i
        self._j = j

    def __repr__(self):
        """Returns a string representation of the operation.
        
        Returns:
            str: string representation of the operation
        """
        return f"{self._name}({self._i},{self._j})"


class MultiplyRow(Operation):
    """Represents the operation of multiplying a row of a system of equations by a scalar."""
    _i: int
    _k: float

    def __init__(self, i: int, k: float):
        """Initializes the operation with the row index of the row to be multiplied and the scalar.
        
        Args:
            i (int): row index
            k (float): scalar
        """
        super().__init__("/multiply_row")
        self._i = i
        self._k = k

    def __repr__(self):
        """Returns a string representation of the operation.
        
        Returns:
            str: string representation of the operation
        """
        return f"{self._name}({self._i},{self._k})"


class ReduceRows(Operation):
    """Represents the operation of adding a multiple of one row to another row of a system of equations."""
    _i: int
    _j: int
    _k: float

    def __init__(self, i: int, j: int, k: float):
        """Initializes the operation with the row indices of the rows to be added and the scalar.
        
        Args:
            i (int): row index
            j (int): row index
            k (float): scalar
        """
        super().__init__("/reduce_rows")
        self._i = i
        self._j = j
        self._k = k

    def __repr__(self):
        """Returns a string representation of the operation.
        
        Returns:
            str: string representation of the operation
        """
        return f"{self._name}({self._i},{self._j},{self._k})"


class SystemOperationPair(NamedTuple):
    """Represents a pair of a system of equations and the corresponding operation."""
    system: Tensor
    operation: Operation


class Record:
    """Records a history of system states and operations."""
    _system_op_pairs: List[SystemOperationPair]

    def __init__(self):
        """Initializes the record with the initial state of the system of equations."""
        self._system_op_pairs = []

    def add(self, system: Tensor, operation: Operation):
        """Adds a pair of a system of equations and the corresponding operation to the record.
        
        Args:
            system (Tensor): system of equations
            operation (str): operation performed on the system of equations
        """
        self._system_op_pairs.append(SystemOperationPair(system, operation))

    def system_op_pairs(self) -> List[SystemOperationPair]:
        """Returns the list of system-operation pairs.
        
        Returns:
            List[SystemOperationPair]: list of system-operation pairs
        """
        return self._system_op_pairs


def repr_system(system: Tensor, names: List[str]) -> str:
    """Returns a string representation of the system of equations.
    
    Args:
        system (Tensor): system of equations
        
    Returns:
        str: string representation of the system of equations
    """
    assert len(system.shape) == 2
    assert len(names) == system.shape[1] - 1
    num_equations: int = system.shape[0]
    num_variables: int = system.shape[1] - 1
    repr_system: str = ""
    for i in range(num_equations):
        for j in range(num_variables):
            if j < num_variables - 1:
                repr_system += f"{system[i, j]}{names[j]} + "
            else:
                repr_system += f"{system[i, j]}{names[j]} "
        repr_system += f"= {system[i, num_variables]}\n"
    return repr_system


def repr_system_op_pair(system_op_pair: SystemOperationPair, names: List[str]) -> str:
    """Returns a string representation of the system of equations and the corresponding operation.
    
    Args:
        system_op_pair (SystemOperationPair): pair of a system of equations and the corresponding operation
        
    Returns:
        str: string representation of the system of equations and the corresponding operation
    """
    return f"{repr_system(system_op_pair.system, names)}{system_op_pair.operation}"


def repr_record(record: Record, names: List[str]) -> str:
    """Returns a string representation of the record.
    
    Args:
        record (Record): record of system states and operations
        
    Returns:
        str: string representation of the record
    """
    repr_record: str = ""
    for system_op_pair in record.system_op_pairs():
        repr_record += f"{repr_system_op_pair(system_op_pair, names)}\n\n"
    return repr_record


if __name__ == "__main__":
    system = Tensor([
        [1, 2, 3],
        [2, 1, 3],
    ])
    names = ["x", "y"]
    print(repr_system(system, names))

    swap_rows(system, 0, 1)
    print(repr_system(system, names))
    swap_operation = SwapRows(0, 1)
    print(swap_operation)
    system_op_pair = SystemOperationPair(system, swap_operation)
    print(repr_system_op_pair(system_op_pair, names))

    print("hello world")

    multiply_row(system, 0, 2)
    print(repr_system(system, names))
    multiply_operation = MultiplyRow(0, 2)
    print(multiply_operation)
    system_op_pair = SystemOperationPair(system, multiply_operation)
    print(repr_system_op_pair(system_op_pair, names))

    reduce_rows(system, 1, 0, -2)
    print(repr_system(system, names))
    reduce_operation = ReduceRows(1, 0, -2)
    print(reduce_operation)
    system_op_pair = SystemOperationPair(system, reduce_operation)
    print(repr_system_op_pair(system_op_pair, names))

    print("start")

    record = Record()
    record.add(system, swap_operation)
    record.add(system, multiply_operation)
    record.add(system, reduce_operation)
    print(repr_record(record, names))