import copy
from typing import Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry.rdGeometry import Point3D

from confgen.generator import BaseConformerGenerator


class ETKDG(BaseConformerGenerator):
    """Conformer Generator using the ETKDG method."""

    def __init__(
        self,
        n_confs: int = 10,
        pruneRmsThresh: float = 0.1,
        useRandomCoords: bool = True,
        ETversion: int = 2,
        **kwargs
    ) -> None:
        self.n_confs = n_confs
        self.pruneRmsThresh = pruneRmsThresh
        self.useRandomCoords = useRandomCoords
        self.ETversion = ETversion
        super().__init__(**kwargs)

    @staticmethod
    def _make_coordmap(
        mol: Chem.Mol, constrained_atoms: Optional[Union[list, dict]] = None
    ) -> dict:
        # Generate coordMap for constrained embedding
        constrained_embed = True if constrained_atoms is not None else False

        coordMap = {}
        if constrained_embed:
            if isinstance(constrained_atoms, dict):
                # loop over values and make sure they are all Point3d objects
                for key, value in constrained_atoms.items():
                    if isinstance(value, Point3D):
                        continue
                    elif isinstance(value, (np.ndarray, list)):
                        value = Point3D(
                            float(value[0]), float(value[1]), float(value[2])
                        )
                        constrained_atoms[key] = value
                    else:
                        raise ValueError(
                            "constrained_atoms must be a dictionary of atomids and Point3D objects"
                        )
                coordMap = constrained_atoms
            elif isinstance(constrained_atoms, list):
                try:
                    conf = mol.GetConformer(0)
                except ValueError:
                    raise ValueError(
                        "No conformers found to apply atom constraints to."
                    )
                for idxI in constrained_atoms:
                    PtI = conf.GetAtomPosition(idxI)
                    coordMap[idxI] = PtI
            else:
                raise ValueError(
                    "constrained_atoms must be either a list of atomids or a coordmap dict."
                )

        return coordMap, constrained_embed

    def generate(
        self, mol: Chem.Mol, constrained_atoms: Optional[Union[list, dict]] = None
    ) -> Chem.Mol:
        """Generate conformers of mol object using the ETKDG method.

        Args:
            mol (Chem.Mol): Mol object
            constrained_atoms (Optional[list|dict]): List of atomids to constrain or coordMap

        Raises:
            ValueError: if mol does not contain a conformer with id 0 which is
                        used to constrain atoms to their respective positions

        Returns:
            Chem.Mol: Mol object with conformers embedded

        Note:
            Conformers are sorted in ascending order of energy.
            Energy is stored in mol.GetConformer(0).GetProp("energy").
        """
        mol3d = copy.deepcopy(mol)
        self.check_mol(mol3d)

        # Generate coordMap for constrained embedding
        coordMap, constrained_embed = self._make_coordmap(mol3d, constrained_atoms)

        cids = AllChem.EmbedMultipleConfs(
            mol=mol3d,
            coordMap=coordMap,
            numThreads=self.n_cores,
            numConfs=self.n_confs,
            pruneRmsThresh=self.pruneRmsThresh,
            useRandomCoords=self.useRandomCoords,
            ETversion=self.ETversion,
            useSmallRingTorsions=True,
        )

        assert len(cids) > 0, "Embed failed."

        # align conformers if constrained embedding
        if constrained_embed:
            if isinstance(constrained_atoms, dict):
                constrained_atoms = list(constrained_atoms.keys())
            _ = AllChem.AlignMolConformers(mol3d, atomIds=constrained_atoms)

        return mol3d
