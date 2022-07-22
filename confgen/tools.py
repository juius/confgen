import copy
from typing import Union

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from confgen.rmsd_utils import rmsd_matrix
from confgen.utils import hartree2kcalmol
from confgen.xtb_utils import xtb_optimize

RDLogger.DisableLog("rdApp.*")


class GeomOptimizer:
    """Geometry optimizer for Chem.Mol objects.

    Args:
        method (str): Method to use for geometry optimization.
                      Options are: UFF, MMFF, and GFNFF, GFN1, GFN2.
        options (dict, optional): Options to use for xtb geometry optimization.
    """

    def __init__(self, method, **kwargs):
        self.method = method
        self.options = kwargs.get("options", {})

    def __repr__(self):
        return f"{self.method.upper()} Optimize"

    def run(
        self, mol: Chem.Mol, n_cores: int = 1, scr: str = ".", **kwargs
    ) -> Chem.Mol:
        """Perform geometry optimization on a Chem.Mol object.

        Args:
            mol (Chem.Mol): Mol object to be optimized
            n_cores (int, optional): Number of cores to use in optimization.
                                     Defaults to 1.
            scr (str, optional): Scratch directory. Defaults to ".".

        Raises:
            Warning: if optimization method is not supported.

        Returns:
            Chem.Mol: Mol object containing conformers with optimized geometry.
        """
        if self.method.lower() == "uff":
            assert AllChem.UFFHasAllMoleculeParams(
                mol
            ), "UFF is not parameterized for this molecule"
            forcefield = AllChem.UFFGetMoleculeForceField(mol)
            results = AllChem.OptimizeMoleculeConfs(mol, forcefield, numThreads=n_cores)
            for i, conf in enumerate(mol.GetConformers()):
                conf.SetDoubleProp("energy", float(results[i][1]))
        elif self.method.lower() == "mmff":
            assert AllChem.MMFFHasAllMoleculeParams(
                mol
            ), "MMFF is not parameterized for this molecule"
            mprobs = AllChem.MMFFGetMoleculeProperties(mol)
            forcefield = AllChem.MMFFGetMoleculeForceField(mol, mprobs)
            results = AllChem.OptimizeMoleculeConfs(mol, forcefield, numThreads=n_cores)
            for i, conf in enumerate(mol.GetConformers()):
                conf.SetDoubleProp("energy", float(results[i][1]))
        elif "gfn" in self.method.lower():
            # set GFN method to use for xTB
            self.options["gfn"] = self.method.lower().split("gfn")[-1]
            gfn = "gfn"
            assert self.options["gfn"].lower() in [
                "ff",
                "1",
                "2",
            ], f"Unsupported method: {self.options[gfn]}"
            mol = xtb_optimize(mol, self.options, n_cores, scr=scr)
        else:
            raise Warning(f"{self.method} is not a valid option.")

        return mol


class Cluster:
    """RMSD clustering on conformers in Chem.Mol object."""

    def __init__(self, threshold: float, keep: str = "lowenergy") -> Chem.Mol:
        self.threshold = threshold
        self.keep = keep

    def __repr__(self):
        return f"RMSD Cluster({self.threshold})"

    def run(self, mol: Chem.Mol, **kwargs):
        """Perform RMSD clustering on conformers of a Chem.Mol object.

        Raises:
            Warning: if 'keep' option is not supported.
                     Options are: 'lowenergy', 'centroid'

        Returns:
            Chem.Mol: Mol object containing conformers from different clusters.
        """

        # Calculate difference matrix
        workers = kwargs.get("n_cores", 1)
        diffmat = rmsd_matrix(mol, n_workers=workers)
        # Cluster conformers
        clt = Butina.ClusterData(
            diffmat,
            mol.GetNumConformers(),
            self.threshold,
            isDistData=True,
            reordering=True,
        )
        # Get centroid conformer of each cluster
        if self.keep.lower() == "centroid":
            confs = [mol.GetConformer(id=c[0]) for c in clt]
        # Get lowest energy conformer of each cluster
        elif self.keep.lower() == "lowenergy":
            ids = []
            for cids in clt:
                energies = [
                    float(mol.GetConformer(idx).GetProp("energy")) for idx in cids
                ]
                ids.append(int(np.argmin(energies)))
            confs = [mol.GetConformer(id=idx) for idx in ids]
        else:
            raise Warning(f"{self.keep} is not a valid option.")
        # Resort conformers by energy
        energies = [
            float(conf.GetProp("energy")) if conf.HasProp("energy") else float("nan")
            for conf in confs
        ]
        confs = [c for _, c in sorted(zip(energies, confs), key=lambda x: x[0])]
        new_mol = copy.deepcopy(mol)
        new_mol.RemoveAllConformers()
        for c in confs:
            new_mol.AddConformer(c, assignId=True)

        return new_mol


class Filter:
    """Energy Filter from lowest energy conformer in kcal/mol.

    Args:
        threshold (int, float): Energy threshold in kcal/mol to filter
                                conformers from lowest energy conformer.
    """

    def __init__(self, ewin: Union[int, float]):
        self.ewin = ewin

    def __repr__(self):
        return f"Energy Filter({self.ewin} kcal/mol)"

    def run(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:
        """Remove all conformers from a Chem.Mol object with energy higher than
        a given threshold from the lowest energy conformer.

        Args:
            mol (Chem.Mol): Mol object

        Returns:
            mol (Chem.Mol): Mol object containing conformers within
                                  a threshold from lowest energy conformer.
        """
        confs = mol.GetConformers()
        energies = np.array(
            [float(conf.GetProp("energy")) * hartree2kcalmol for conf in confs]
        )
        mask = energies < (energies.min() + self.ewin)
        confs = list(np.array(confs)[mask])
        new_mol = copy.deepcopy(mol)
        new_mol.RemoveAllConformers()
        for c in confs:
            new_mol.AddConformer(c, assignId=True)

        return new_mol
