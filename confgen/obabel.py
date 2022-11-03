from pathlib import Path
from tempfile import TemporaryDirectory

from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import AllChem

from confgen.generator import BaseConformerGenerator
from confgen.utils import normal_termination, stream


def rdkit2ob(mol):
    with TemporaryDirectory() as tempdir:
        filename = tempdir + "/rdkit2ob.sdf"
        Chem.SDWriter(filename).write(mol)
        obmol = openbabel.OBMol()
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "sdf")
        obConversion.ReadFile(obmol, filename)
    return obmol


def ob2rdkit(obmol, all_confs=True, skip_first=False):
    with TemporaryDirectory() as tempdir:
        dir_ = Path(tempdir)
        n_confs_to_write = obmol.NumConformers() if all_confs else 1
        if skip_first:
            start = 1
            n_confs_to_write += 1
        else:
            start = 0
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "sdf")
        for n in range(start, n_confs_to_write):
            obmol.SetConformer(n)
            obConversion.WriteFile(obmol, str(dir_ / f"rdkit2ob_{n}.sdf"))

        mols = []
        for f in dir_.glob("*.sdf"):
            suppl = Chem.SDMolSupplier(str(f.resolve()), removeHs=False, sanitize=False)
            mols.append(suppl[0])

        for i, m in enumerate(mols):
            if i < 1:
                mol = m
            else:
                mol.AddConformer(m.GetConformer(), assignId=True)
    return mol


class Gen3d(BaseConformerGenerator):
    def __init__(self, n_cores: int = 1, scr: str = ".") -> None:
        super().__init__(n_cores, scr)

    def generate(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:
        with TemporaryDirectory(dir=self.scr) as tempdir:
            input_file = tempdir + "/mol.sdf"
            output_file = tempdir + "/obabel_out.sdf"
            Chem.SDWriter(input_file).write(mol)
            lines = list(
                stream(f"obabel -i sdf {input_file} -o sdf -O {output_file} --gen3d")
            )
            if not normal_termination(lines, "1 molecule converted"):
                raise Warning(f"OpenBabel failed with error:\n{''.join(lines)}")
            suppl = Chem.SDMolSupplier(output_file, removeHs=False, sanitize=False)
        mol3d = suppl[0]

        return mol3d


class Confab(BaseConformerGenerator):
    def __init__(self, n_cores: int = 1, scr: str = ".") -> None:
        super().__init__(n_cores, scr)

    def generate(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:

        assert AllChem.EmbedMolecule(mol) == 0, "Inital embedding failed"
        ff_type = kwargs.get("ff_type", "uff")
        rmsd_cutoff = kwargs.get("rmsd_cutoff", 0.5)
        conf_cutoff = kwargs.get("conf_cutoff", 4000000)
        energy_cutoff = kwargs.get("energy_cutoff", 50.0)

        obmol = rdkit2ob(mol)

        ff = openbabel.OBForceField.FindForceField(ff_type)
        obmol.AddHydrogens()
        if ff.Setup(obmol) == 0:
            print("Could not setup forcefield")
        ff.SetCoordinates(obmol)
        ff.DiverseConfGen(rmsd_cutoff, conf_cutoff, energy_cutoff, True)
        ff.GetConformers(obmol)

        mol3d = ob2rdkit(obmol, all_confs=True)

        return mol3d


class WeightedRotor(BaseConformerGenerator):
    def __init__(self, n_cores: int = 1, scr: str = ".") -> None:
        super().__init__(n_cores, scr)

    def generate(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:

        ff_type = kwargs.get("ff_type", "uff")
        n_confs = kwargs.get("n_confs", 250)
        n_steps = kwargs.get("n_steps", 100)

        obmol = rdkit2ob(mol)

        ff = openbabel.OBForceField.FindForceField(ff_type)
        obmol.AddHydrogens()
        if ff.Setup(obmol) == 0:
            print("Could not setup forcefield")
        ff.SetCoordinates(obmol)
        ff.WeightedRotorSearch(n_confs, n_steps)
        ff.GetConformers(obmol)

        mol3d = ob2rdkit(obmol, all_confs=False, skip_first=True)

        return mol3d
