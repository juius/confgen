from abc import ABC, abstractmethod
from pathlib import Path

from rdkit import Chem


class BaseConformerGenerator(ABC):
    """Base class for conformer generators."""

    def __init__(self, n_cores: int = 1, scr: str = ".") -> None:
        self.n_cores = n_cores
        self.scr = Path(scr)

    def __repr__(self):
        str_options = ", ".join(
            [":".join([str(key), str(val)]) for key, val in self.__dict__.items()]
        )
        return repr(f"{self.__class__.__name__}({str_options})")

    @abstractmethod
    def generate(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:
        pass
