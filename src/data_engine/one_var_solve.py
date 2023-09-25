import torch

from typing import Union, Optional, Tuple, NamedTuple, List
from random import randint, random


class Equation:
    """Represents a single linear equation with one variable."""
    _tensor: torch.Tensor

    def __init__(self, a: Union[int, float], b: Union[int, float], c: Union[int, float], d: Union[int, float],
                 dtype: Optional[torch.dtype] = None):
        """Initializes an Equation object.

        Args:
            a (Union[int, float]): coefficient of the variable
            b (Union[int, float]): constant
            c (Union[int, float]): coefficient of the variable
            d (Union[int, float]): constant
            dtype (Optional[torch.dtype], optional): data type of the equation tensor. Defaults to None.
        """
        if dtype is None:
            dtype = torch.float
        self._tensor = torch.tensor([[a, b], [c, d]], dtype=dtype)

    @classmethod
    def generate_rand_int_equation(cls, max_solution: int, lower_bound: int, upper_bound: int) -> "Equation":
        """Generates a random Equation object with integer coefficients and constants.

        Args:
            max_solution (int, optional): maximum value of the solution.
            lower_bound (int, optional): lower bound for randomly generated coefficients and constants.
            upper_bound (int, optional): upper bound for randomly generated coefficients and constants.

        Returns:
            Equation: random Equation object
        """
        x: int = randint(-max_solution, max_solution)
        left_coefficient: int = randint(lower_bound, upper_bound)
        right_coefficient: int = randint(lower_bound, upper_bound)
        while abs(left_coefficient - right_coefficient) < 1e-10:
            right_coefficient = randint(lower_bound, upper_bound)
        remaining_constant: int = (left_coefficient - right_coefficient) * x
        left_constant: int = randint(lower_bound, upper_bound)
        right_constant: int = left_constant + remaining_constant
        return cls(left_coefficient, left_constant, right_coefficient, right_constant, dtype=torch.float)

    @classmethod
    def generate_rand_float_equation(cls, max_solution: int, lower_bound: int, upper_bound: int) -> "Equation":
        """Generates a random Equation object with float coefficients and constants.

        Args:
            max_solution (int, optional): maximum value of the solution.
            lower_bound (int, optional): lower bound for randomly generated coefficients and constants.
            upper_bound (int, optional): upper bound for randomly generated coefficients and constants.

        Returns:
            Equation: random Equation object
        """
        x: float = max_solution * (2 * random() - 1)
        left_coefficient: float = random() * (upper_bound - lower_bound) + lower_bound
        right_coefficient: float = random() * (upper_bound - lower_bound) + lower_bound
        while abs(left_coefficient - right_coefficient) < 1e-10:
            right_coefficient = random() * (upper_bound - lower_bound) + lower_bound
        remaining_constant: float = (left_coefficient - right_coefficient) * x
        left_constant: float = random() * (upper_bound - lower_bound) + lower_bound
        right_constant: float = left_constant + remaining_constant
        return cls(left_coefficient, left_constant, right_coefficient, right_constant, dtype=torch.float)

    def add_constant(self, constant: Union[int, float]):
        """Adds a constant to the equation.

        Args:
            constant (Union[int, float]): constant
        """
        self._tensor[:, 1] += constant

    def add_coefficient(self, coefficient: Union[int, float]):
        """Adds a coefficient to the equation.

        Args:
            coefficient (Union[int, float]): coefficient
        """
        self._tensor[:, 0] += coefficient

    def subtract_constant(self, constant: Union[int, float]):
        """Subtracts a constant from the equation.

        Args:
            constant (Union[int, float]): constant
        """
        self._tensor[:, 1] -= constant

    def subtract_coefficient(self, coefficient: Union[int, float]):
        """Subtracts a coefficient from the equation.

        Args:
            coefficient (Union[int, float]): coefficient
        """
        self._tensor[:, 0] -= coefficient

    def multiply_constant(self, constant: Union[int, float]):
        """Multiplies the equation by a constant.

        Args:
            constant (Union[int, float]): constant
        """
        self._tensor *= constant

    def divide_constant(self, constant: Union[int, float]):
        """Divides the equation by a constant.

        Args:
            constant (Union[int, float]): constant
        """
        self._tensor /= constant

    def tensor(self) -> torch.Tensor:
        """Returns the equation tensor.

        Returns:
            torch.Tensor: equation tensor
        """
        return self._tensor

    def clone(self) -> "Equation":
        """Returns a clone of the equation.

        Returns:
            Equation: clone of the equation
        """
        return Equation(self._tensor[0, 0].item(), self._tensor[0, 1].item(),
                        self._tensor[1, 0].item(), self._tensor[1, 1].item())

    def __repr__(self) -> str:
        """Returns a string representation of the equation.

        Returns:
            str: string representation of the equation
        """
        representation = f" {self._tensor[0, 0].item()}x + {self._tensor[0, 1].item()} = " \
                         f"{self._tensor[1, 0].item()}x + {self._tensor[1, 1].item()}"
        # replace
        representation = representation.replace(".0", "")
        representation = representation.replace(" 1x", " x")
        representation = representation.replace(" 0x ", " ")
        representation = representation.replace(" -0x", "")
        representation = representation.replace(" -0", "")
        representation = representation.replace(" + 0", "")
        representation = representation.replace("= +", "= ")
        representation = representation.replace("+ =", " =")
        representation = representation.replace("+ -", "-")
        representation = representation.replace("  ", " ")
        return representation


class Transformation:
    """Represents an equation operation."""
    _name: str

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        """Returns a string representation of the transformation.

        Returns:
            str: string representation of the transformation
        """
        representation = repr(self)
        representation = representation.replace(".0", "")
        representation = representation.replace("1x", "x")
        return representation

    def __repr__(self) -> str:
        """Returns a string representation of the transformation.

        Returns:
            str: string representation of the transformation
        """
        return f"{self._name}"


class Add(Transformation):
    """Represents an addition operation."""
    _argument: str

    def __init__(self, argument: str):
        super().__init__("/add")
        self._argument = argument

    def __repr__(self):
        return f"{self._name}({self._argument})"


class Subtract(Transformation):
    """Represents a subtraction operation."""
    _argument: str

    def __init__(self, argument: str):
        super().__init__("/sub")
        self._argument = argument

    def __repr__(self):
        return f"{self._name}({self._argument})"


class Multiply(Transformation):
    """Represents a multiplication operation."""
    _argument: str

    def __init__(self, argument: str):
        super().__init__("/mul")
        self._argument = argument

    def __repr__(self):
        return f"{self._name}({self._argument})"


class Divide(Transformation):
    """Represents a division operation."""
    _argument: str

    def __init__(self, argument: str):
        super().__init__("/div")
        self._argument = argument

    def __repr__(self):
        return f"{self._name}({self._argument})"


class End(Transformation):
    """Represents an end operation."""

    def __init__(self):
        super().__init__("/end")

    def __repr__(self):
        return f"{self._name}"


class EquationTransformationPair(NamedTuple):
    """Represents a pair of an equation and an equation operation."""
    equation: Equation
    operation: Transformation

    def __repr__(self) -> str:
        """Returns a string representation of the equation transformation pair."""
        return f"{self.equation}?{self.operation}"


class Record:
    """Represents a record of equation transformation pairs."""
    _equation_transformation_pairs: List[EquationTransformationPair]

    def __init__(self):
        self._equation_transformation_pairs = []

    def add_equation_transformation_pair(self, equation: Equation, transformation: Transformation):
        """Adds an equation transformation pair to the record.

        Args:
            equation (Equation): equation
            transformation (Transformation): equivalence transformation
        """
        self._equation_transformation_pairs.append(
            EquationTransformationPair(equation, transformation)
        )

    def equation_transformation_pairs(self) -> List[EquationTransformationPair]:
        """Returns the equation transformation pairs.

        Returns:
            List[EquationTransformationPair]: equation transformation pairs
        """
        return self._equation_transformation_pairs

    def __repr__(self) -> str:
        """Returns a string representation of the record.

        Returns:
            str: string representation of the record
        """
        return "\n".join([str(pair) for pair in self._equation_transformation_pairs])


def solve(equation: Equation) -> Tuple[Equation, Record]:
    # initialize record
    record: Record = Record()
    # check left constant
    b = equation.tensor()[0, 1].item()
    if abs(b - 0) > 1e-10:
        # subtract left constant
        record.add_equation_transformation_pair(equation.clone(), Subtract(f"{b}"))
        equation.subtract_constant(b)
    # check right coefficient
    c = equation.tensor()[1, 0].item()
    if abs(c - 0) > 1e-10:
        # subtract by right coefficient
        record.add_equation_transformation_pair(equation.clone(), Subtract(f"{c}x"))
        equation.subtract_coefficient(c)
    # check left coefficient
    a = equation.tensor()[0, 0].item()
    if abs(a - 0) > 1e-10:
        if abs(a - 1) > 1e-10:
            # divide by left coefficient
            record.add_equation_transformation_pair(equation.clone(), Divide(f"{a}"))
            equation.divide_constant(a)
    # end
    record.add_equation_transformation_pair(equation.clone(), End())
    return equation, record
