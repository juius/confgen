import copy
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from confgen.generator import BaseConformerGenerator


class ETKDG(BaseConformerGenerator):
    """Conformer Generator using the ETKDG method."""

    def __init__(
        self,
        n_confs: int = 10,
        pruneRmsThresh: float = 0.1,
        actions: Optional[list] = None,
        useRandomCoords: bool = True,
        ETversion: int = 2,
        **kwargs
    ) -> None:
        self.n_confs = n_confs
        self.pruneRmsThresh = pruneRmsThresh
        self.actions = actions
        self.useRandomCoords = useRandomCoords
        self.ETversion = ETversion
        super().__init__(**kwargs)

    def generate(
        self, mol: Chem.Mol, constrained_atoms: Optional[list] = None
    ) -> Chem.Mol:
        """Generate conformers of mol object using the ETKDG method.

        Args:
            mol (Chem.Mol): Mol object
            constrained_atoms (Optional[list]): List of atomids to constrain

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
        assert (
            len(Chem.GetMolFrags(mol3d)) == 1
        ), "Can not handle multiple fragments yet."

        constrained_embed = True if constrained_atoms is not None else False

        coordMap = {}
        if constrained_embed:
            try:
                conf = mol3d.GetConformer(0)
            except ValueError:
                raise ValueError("No conformers found to apply atom constraints to.")
            for idxI in constrained_atoms:
                PtI = conf.GetAtomPosition(idxI)
                coordMap[idxI] = PtI

        cids = AllChem.EmbedMultipleConfs(
            mol=mol3d,
            coordMap=coordMap,
            numThreads=self.n_cores,
            numConfs=self.n_confs,
            pruneRmsThresh=self.pruneRmsThresh,
            useRandomCoords=self.useRandomCoords,
            ETversion=self.ETversion,
        )

        assert len(cids) > 0, "Embed failed."

        if self.actions:
            for action in self.actions:
                mol3d = action.run(
                    mol3d,
                    n_cores=self.n_cores,
                    scr=self.scr,
                    constrained_atoms=constrained_atoms,
                )

        if constrained_embed:
            _ = AllChem.AlignMolConformers(mol3d, atomIds=constrained_atoms)

        return mol3d
