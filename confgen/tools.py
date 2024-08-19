import copy
from typing import Union

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from confgen.rmsd_utils import rmsd_matrix
from confgen.utils import hartree2kcalmol, sort_conformers
from confgen.xtb_utils import xtb_calculate

try:
    from tooltoad.orca import orca_calculate
except ImportError:
    print("tooltoad is not installed. Please install it to use orca_calculate options.")

RDLogger.DisableLog("rdApp.*")


class GeomOptimizer:
    """Geometry optimizer for Chem.Mol objects.

    Args:
        method (str): Method to use for geometry optimization.
                      Options are: UFF, MMFF, and GFNFF, GFN1, GFN2.
        options (dict, optional): Options to use for xtb geometry optimization.
        n_cores (int, optional): Number of cores to use in calculation.
                                     Defaults to 1.
        scr (str, optional): Scratch directory. Defaults to ".".
    """

    def __init__(self, method: str, n_cores: int = 1, scr: str = ".", **kwargs):
        self.method = method
        self.n_cores = n_cores
        self.scr = scr
        self.options = kwargs.get("options", {})

    def __repr__(self):
        return f"{self.method.upper()} Optimize"

    def run(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:
        """Perform geometry optimization on a Chem.Mol object.

        Args:
            mol (Chem.Mol): Mol object to be optimized

        Raises:
            Warning: if optimization method is not supported.

        Returns:
            Chem.Mol: Mol object containing conformers with optimized geometry.
        """
        if self.method.lower() == "uff":
            # assert AllChem.UFFHasAllMoleculeParams(
            #     mol
            # ), "UFF is not parameterized for this molecule"
            forcefield = AllChem.UFFGetMoleculeForceField(mol)
            results = AllChem.OptimizeMoleculeConfs(
                mol, forcefield, numThreads=self.n_cores
            )
            for i, conf in enumerate(mol.GetConformers()):
                conf.SetDoubleProp("energy", float(results[i][1]))
        elif self.method.lower() == "mmff":
            assert AllChem.MMFFHasAllMoleculeParams(
                mol
            ), "MMFF is not parameterized for this molecule"
            mprobs = AllChem.MMFFGetMoleculeProperties(mol)
            forcefield = AllChem.MMFFGetMoleculeForceField(mol, mprobs)
            results = AllChem.OptimizeMoleculeConfs(
                mol, forcefield, numThreads=self.n_cores
            )
            for i, conf in enumerate(mol.GetConformers()):
                conf.SetDoubleProp("energy", float(results[i][1]))
        elif "gfn" in self.method.lower():
            # set GFN method to use for xTB
            _ = self.options.setdefault("opt", None)
            self.options["gfn"] = self.method.lower().split("gfn")[-1]
            gfn = "gfn"
            assert self.options[gfn].lower() in [
                "ff",
                "1",
                "2",
            ], f"Unsupported method: {self.options[gfn]}"
            detailed_input = kwargs.get("detailed_input", None)
            mol = xtb_calculate(
                mol,
                options=self.options,
                detailed_input=detailed_input,
                n_cores=self.n_cores,
                scr=self.scr,
            )
        else:
            raise Warning(f"{self.method} is not a valid option.")

        return mol


class Cluster:
    """RMSD clustering on conformers in Chem.Mol object."""

    def __init__(self, rmsdThreshold: float, keep: str = "lowenergy") -> Chem.Mol:
        self.threshold = rmsdThreshold
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


class SinglePoint:
    """Geometry optimizer for Chem.Mol objects.

    Args:
        method (str): Method to use for geometry optimization.
                      Options are: UFF, MMFF, and GFNFF, GFN1, GFN2.
        options (dict, optional): Options to use for xtb geometry optimization.
        n_cores (int, optional): Number of cores to use in calculation.
                                     Defaults to 1.
        scr (str, optional): Scratch directory. Defaults to ".".
    """

    def __init__(
        self, method: str = "gfn2", n_cores: int = 1, scr: str = ".", **kwargs
    ):
        self.method = method
        self.n_cores = n_cores
        self.scr = scr
        self.options = kwargs.get("options", {})
        for key, value in kwargs.items():
            if key not in ["options"]:
                self.__setattr__(key, value)

    def __repr__(self):
        return "Single Point Calculation"

    def run(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:
        """Perform single point calculation on a Chem.Mol object.

        Args:
            mol (Chem.Mol): Mol object


        Returns:
            Chem.Mol: Mol object containing conformers with optimized geometry.
        """
        # TODO: use tooltoad for all qm calcs
        # set GFN method to use for xTB
        if "gfn" in self.method.lower():
            self.options["gfn"] = self.method.lower().split("gfn")[-1]
            gfn = "gfn"
            assert self.options["gfn"].lower() in [
                "ff",
                "1",
                "2",
            ], f"Unsupported method: {self.options[gfn]}"
            mol = xtb_calculate(
                mol, options=self.options, n_cores=self.n_cores, scr=self.scr
            )
        elif self.method.lower() == "orca":
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            charge = Chem.GetFormalCharge(mol)
            for conf in mol.GetConformers():
                coords = conf.GetPositions()
                results = orca_calculate(
                    atoms=atoms,
                    coords=coords,
                    charge=charge,
                    options=self.options,
                    n_cores=self.n_cores,
                    scr=self.scr,
                    orca_cmd=kwargs.get("orca_cmd", "orca"),
                    set_env=kwargs.get("set_env", ""),
                )
                conf.SetDoubleProp(
                    "energy",
                    (
                        results["electronic_energy"]
                        if results["normal_termination"]
                        else np.inf
                    ),
                )
            mol = sort_conformers(mol, property="energy")

        else:
            raise NotImplementedError(f"{self.method} is not a valid option.")
        return mol
