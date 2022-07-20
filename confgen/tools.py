import copy

import numpy as np
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from confgen.rmsd_utils import rmsd_matrix
from confgen.utils import hartree2kcalmol
from confgen.xtb_utils import xtb_optimize

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")



class GeomOptimizer:
    """Optimize the molecular geometry using UFF, MMFF or a GFNx method"""

    def __init__(self, method, **kwargs):
        self.method = method
        self.options = kwargs.get("options", {})

    def __repr__(self):
        # add representation for xTB method
        return f"{self.method.upper()} Optimize"

    def run(self, mol, **kwargs):
        n_cores = kwargs.get("n_cores", 1)
        scr = kwargs.get("scr", ".")
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
            mol = xtb_optimize(mol, self.options, n_cores, scr=scr)
        else:
            raise Warning(f"{self.method} is not a valid option.")

        return mol


class Cluster:
    """RMSD Clustering"""

    def __init__(self, threshold, keep="centroid"):
        self.threshold = threshold
        self.keep = keep

    def __repr__(self):
        return f"RMSD Cluster({self.threshold})"

    def run(self, mol, **kwargs):
        verbose = kwargs.get("verbose", False)
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
        try:
            energies = [float(conf.GetProp("energy")) for conf in confs]
            confs = [c for _, c in sorted(zip(energies, confs))]
        except:
            pass
        new_mol = copy.deepcopy(mol)
        new_mol.RemoveAllConformers()
        for c in confs:
            new_mol.AddConformer(c, assignId=True)
        if verbose:
            print(
                f"{new_mol.GetNumConformers()} Conformers after Clustering({self.threshold})"
            )

        return new_mol


class Filter:
    """Energy Filter from lowest energy conformer in kcal/mol"""

    def __init__(self, ewin=None):
        self.ewin = ewin

    def __repr__(self):
        return f"Energy Filter({self.ewin} kcal/mol)"

    def run(self, mol, **kwargs):
        verbose = kwargs.get("verbose", False)
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
        if verbose:
            print(
                f"{new_mol.GetNumConformers()} Conformers after Energy Filtering({self.ewin})"
            )

        return new_mol
