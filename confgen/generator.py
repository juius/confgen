from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from rdkit import Chem

from confgen.utils import combine_conformers


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


class MixedGenerator:
    def __init__(self, generators: List[BaseConformerGenerator]) -> None:
        self.generators = generators

    def generate(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:
        mols = []
        for g in self.generators:
            mol = g.generate(mol, **kwargs)
            mols.append(mol)

        for i, m in enumerate(mols):
            if i == 0:
                mol3d = m
            else:
                mol3d = combine_conformers(mol3d, m)

        return mol3d
