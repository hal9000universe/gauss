#  Copyright (c) 2023. Benjamin Schoofs

import torch
from typing import List

from src.data_engine.row_ops import SystemOperationPair, Record


def repr_system(system: torch.Tensor, names: List[str]) -> str:
    """Returns a string representation of the system of equations.

    Args:
        system (Tensor): system of equations
        names (List[str]): list of variable names

    Returns:
        str: string representation of the system of equations
    """
    assert len(system.shape) == 2
    assert len(names) == system.shape[1] - 1
    num_equations: int = system.shape[0]
    num_variables: int = system.shape[1] - 1
    system_representation: str = ""
    for i in range(num_equations):
        for j in range(num_variables):
            if j < num_variables - 1:
                system_representation += f"{system[i, j]}*{names[j]} + "
            else:
                system_representation += f"{system[i, j]}*{names[j]} "
        system_representation += f"= {system[i, num_variables]} /n"
    return system_representation[:-3]


def repr_system_op_pair(system_op_pair: SystemOperationPair, names: List[str]) -> str:
    """Returns a string representation of the system of equations and the corresponding operation.

    Args:
        system_op_pair (SystemOperationPair): a pair of a system of equations and the corresponding operation
        names (List[str]): list of variable names

    Returns:
        str: string representation of the system of equations and the corresponding operation
    """
    return f"{repr_system(system_op_pair.system, names)}?{system_op_pair.operation}"


def repr_record(record: Record, names: List[str]) -> str:
    """Returns a string representation of the record.

    Args:
        record (Record): record of system states and operations
        names (List[str]): list of variable names

    Returns:
        str: string representation of the record
    """
    record_representation: str = ""
    for system_op_pair in record.system_op_pairs():
        record_representation += f"{repr_system_op_pair(system_op_pair, names)} \n"
    # remove final newline
    record_representation = record_representation[:-1]
    # add end token
    record_representation += "/rref \n"
    return record_representation
