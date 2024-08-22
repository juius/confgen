import copy
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from confgen.utils import normal_termination, sort_conformers, stream

XTB_CMD = "xtb"
_logger = logging.getLogger("xtb")
_logger.setLevel(logging.INFO)


def set_threads(n_cores):
    """Set threads and procs environment variables."""
    _ = list(stream("ulimit -s unlimited"))
    os.environ["OMP_STACKSIZE"] = "4G"
    os.environ["OMP_NUM_THREADS"] = f"{n_cores},1"
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


def check_xtb(version=None, logger=None):
    """Check xTB executable and version."""
    lines = stream(f"{XTB_CMD} --version")
    lines = list(lines)
    if normal_termination(lines, "normal termination"):
        if version:
            for line in lines:
                if "xtb version" in line:
                    assert line.split()[3] == version, f"Requires xtb version {version}"
            _logger.info(f"xtb version {version}\n")
    else:
        raise Warning("Could not find xTB")


def write_xyz(atoms, coords, scr):
    """Write .xyz file."""
    natoms = len(atoms)
    xyz = f"{natoms} \n \n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    with open(scr / "mol.xyz", "w") as inp:
        inp.write(xyz)
    return scr / "mol.xyz"


def parse_coordline(line):
    line = line.split()
    atom = line[0]
    coord = [float(x) for x in line[-3:]]
    return atom, coord


def add_conformer2mol(mol, atoms, coords, energy=None):
    """Add Conformer to rdkit.mol object."""
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        # assert that same atom type
        assert (
            mol.GetAtomWithIdx(i).GetSymbol() == atoms[i]
        ), "Order of atoms if not the same in CREST output and rdkit Mol"
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    if energy:
        conf.SetDoubleProp("energy", float(energy))
    mol.AddConformer(conf, assignId=True)


def run_xtb(args):
    """Runs xTB command for xyz-file in parent directory and returns optimized
    structure."""
    cmd, xyz_file = args
    lines = stream(f"{cmd}-- {xyz_file.name}", cwd=xyz_file.parent)
    lines = list(lines)
    _logger.debug("".join(lines) + "\n")
    if normal_termination(lines, "normal termination of xtb"):
        return lines
    else:
        _logger.warning("Calculation terminated abnormally.\n")
        return None


def read_opt_structure(lines):
    """Reads optimized structure from xTB output."""
    for i, l in reversed(list(enumerate(lines))):
        if "final structure" in l:
            break

    n_atoms = int(lines[i + 2].rstrip())
    start = i + 4
    end = start + n_atoms

    atoms = []
    coords = []
    for line in lines[start:end]:
        atom, coord = parse_coordline(line)
        atoms.append(atom)
        coords.append(coord)

    return atoms, coords


def read_energy(lines):
    """Reads energy from xTB output."""
    for i, l in reversed(list(enumerate(lines))):
        if "TOTAL ENERGY" in l:
            energy = float(l.split()[-3])
            break
    return energy


def get_detailed_input(detailed_input_dict, force_constant: float = 1.0):
    """Returns a string with the detailed input for xTB."""
    output = ""
    for section in detailed_input_dict.keys():
        output += f"${section}\n"
        output += str(detailed_input_dict[section])
    output += f"$force constant={force_constant}\n"
    output += "$end\n"
    return output


def xtb_calculate(
    mol: Chem.Mol,
    options: dict = None,
    detailed_input: dict = None,
    n_cores: int = 1,
    scr: str = ".",
) -> Chem.Mol:
    """Run xTB calculation on each conformer of rdkit.mol in parallel.

    Args:
        mol (Chem.Mol): Mol object
        options (dict, optional): xtb options, see https://xtb-docs.readthedocs.io/en/latest/commandline.html. Defaults to None.
        detailed_input (dict, optional): detailed input dict, see https://xtb-docs.readthedocs.io/en/latest/xcontrol.html. Defaults to None.
        n_cores (int, optional): Number of cores to use. Defaults to 1.
        scr (str, optional): Working directory. Defaults to ".".

    Returns:
        Chem.Mol: Mol object
    """

    # Check that mol has 3D conformer
    n_confs = mol.GetNumConformers()
    assert n_confs > 0, "Mol has no conformers"
    assert mol.GetConformer().Is3D(), "Mol has no 3D conformer"

    # Check xtb version and executable
    check_xtb()

    # Only use one core for each xTB calculation
    cores_per_calc = max([n_cores // n_confs, 1])
    set_threads(cores_per_calc)

    # Creat TMP directory
    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix="XTBOPT_")
    tmp_scr = Path(tempdir.name)
    _logger.info(f"Working directory:\n{tmp_scr}\n")

    # Write xyz-file for each conformer
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz_files = []
    for i, conf in enumerate(mol.GetConformers()):
        confdir = tmp_scr / "conf{num:0{width}}".format(num=i, width=len(str(n_confs)))
        os.makedirs(confdir)
        coords = conf.GetPositions()
        xyz_file = write_xyz(atoms, coords, confdir)
        xyz_files.append(xyz_file)

    # clean xtb method option
    for k, value in options.items():
        if "gfn" in k.lower():
            if value is not None and value is not True:
                options[k + str(value)] = None
                del options[k]
                break
    # Options to xTB command
    cmd = f"{XTB_CMD} --verbose --parallel {cores_per_calc} --chrg {Chem.GetFormalCharge(mol)} "
    for key, value in options.items():
        if value is None or value is True:
            cmd += f"--{key} "
        else:
            cmd += f"--{key} {str(value)} "
    if detailed_input:
        # write detailed input file
        detailed_string = get_detailed_input(detailed_input)
        with open(tmp_scr / "constrains.inp", "w") as f:
            f.write(detailed_string)
        cmd += "--input ../constrains.inp "
        _logger.info(f"Detailed input:\n{detailed_string}\n")
    _logger.info(f"Command:\n{cmd}\n")

    # run xTB on each conformer in parallel
    args = []
    for xyz_file in xyz_files:
        args.append((cmd, xyz_file))

    n_workers = n_cores // cores_per_calc
    with get_context("fork").Pool(n_workers) as pool:
        xtb_results = pool.map(run_xtb, args)
    xtb_results = list(xtb_results)

    if "opt" in options:
        results = []
        for res in xtb_results:
            atoms, coords = read_opt_structure(res)
            energy = read_energy(res)
            results.append((atoms, coords, energy))
        # Add optimized conformers to mol_opt
        mol_opt = copy.deepcopy(mol)
        # Remove all but first conformers
        confids = [conf.GetId() for conf in mol_opt.GetConformers()]
        _ = [mol_opt.RemoveConformer(i) for i in confids[1:]]

        # Sort results in ascending energy (index 2)
        results.sort(key=lambda res: res[2])
        # Add optimized conformers
        for i, res in enumerate(results):
            if res:
                add_conformer2mol(mol_opt, res[0], res[1], res[2])
            else:
                print(f"Conformer {i} did not converge.")
        # Remove last old conformer
        mol_opt.RemoveConformer(0)
        # Reset confIDs (starting from 0)
        confs = mol_opt.GetConformers()
        for i in range(len(confs)):
            confs[i].SetId(i)
        # align conformers along constrained atoms
        catoms = (
            options.get("detailed_input", dict())
            .get("constrain", dict())
            .get("atoms", None)
        )
        if catoms:
            atomlist = []
            parts = catoms.split(",")
            for part in parts:
                if "-" in part:
                    start, stop = part.split("-")
                    r = list(range(int(start), int(stop) + 1))
                    atomlist.extend(r)
                else:
                    atomlist.append(int(part))
            _ = AllChem.AlignMolConformers(mol_opt, atomIds=atomlist)
    else:
        # read energy and set as conf property
        mol_opt = copy.deepcopy(mol)
        results = []
        for res in xtb_results:
            if res:
                energy = read_energy(res)
            else:
                energy = math.nan
            results.append(energy)
        for conf, energy in zip(mol_opt.GetConformers(), results):
            conf.SetDoubleProp("energy", energy)
        # resort conformers with ascending energy
        mol_opt = sort_conformers(mol_opt, property="energy")

    return mol_opt


@dataclass
class AtomConstraint:
    atom: int

    def __post_init__(self):
        assert isinstance(self.atom, int), "atom must be int"
        self.atom += 1  # xtb starts counting at 1

    def __str__(self) -> str:
        return f"atom: {self.atom}"


@dataclass
class ElementConstraint:
    element: str

    def __post_init__(self):
        assert isinstance(self.element, str), "element must be str"
        self.element = self.element.capitalize()

    def __str__(self) -> str:
        return f"element: {self.element}"


@dataclass
class DistanceConstraint:
    atoms: List or Tuple
    length: float or int or None

    def __post_init__(self):
        assert isinstance(self.atoms, (list, tuple)), "atoms must be a list or tuple"
        assert len(self.atoms) == 2, "DistanceConstraint must have 2 atoms"
        if not self.length:
            self.length = "auto"
        self.atoms = [a + 1 for a in self.atoms]  # xtb starts counting at 1

    def __str__(self) -> str:
        return f"distance: {self.atoms[0]}, {self.atoms[1]}, {self.length}"


@dataclass
class AngleConstraint:
    atoms: List or Tuple
    angle: float or int or None

    def __post_init__(self):
        assert isinstance(self.atoms, (list, tuple)), "atoms must be a list or tuple"
        assert len(self.atoms) == 3, "AngleConstraint must have 3 atoms"
        if self.angle:
            assert self.angle <= 180, "angle must be <= 180 degrees"
        if not self.angle:
            self.angle = "auto"
        self.atoms = [a + 1 for a in self.atoms]  # xtb starts counting at 1

    def __str__(self) -> str:
        return f"angle: {self.atoms[0]}, {self.atoms[1]}, {self.atoms[2]}, {self.angle}"


@dataclass
class DihedralConstraint:
    atoms: List or Tuple
    angle: float or int or None

    def __post_init__(self):
        assert isinstance(self.atoms, (list, tuple)), "atoms must be a list or tuple"
        assert len(self.atoms) == 4, "DihedralConstraint must have 4 atoms"
        if self.angle:
            assert self.angle <= 180, "angle must be <= 180 degrees"
        if not self.angle:
            self.angle = "auto"
        self.atoms = [a + 1 for a in self.atoms]  # xtb starts counting at 1

    def __str__(self) -> str:
        return f"dihedral: {self.atoms[0]}, {self.atoms[1]}, {self.atoms[2]}, {self.atoms[3]}, {self.angle}"


@dataclass
class XtbConstraints:
    atoms: List[AtomConstraint] = None
    elements: List[ElementConstraint] = None
    distances: List[DistanceConstraint] = None
    angles: List[AngleConstraint] = None
    dihedrals: List[DihedralConstraint] = None

    def __post_init__(self):
        if self.atoms:
            for a in self.atoms:
                assert isinstance(
                    a, AtomConstraint
                ), "atoms must be a list of AtomConstraint"
        if self.elements:
            for e in self.elements:
                assert isinstance(
                    e, ElementConstraint
                ), "elements must be a list of ElementConstraint"
        if self.distances:
            for d in self.distances:
                assert isinstance(
                    d, DistanceConstraint
                ), "distances must be a list of DistanceConstraint"
        if self.angles:
            for ang in self.angles:
                assert isinstance(
                    ang, AngleConstraint
                ), "angles must be a list of AngleConstraint"
        if self.dihedrals:
            for dih in self.dihedrals:
                assert isinstance(
                    dih, DihedralConstraint
                ), "dihedrals must be a list of DihedralConstraint"

    def __str__(self) -> str:
        string = ""
        if self.atoms:
            for a in self.atoms:
                string += f"{a}\n"
        if self.elements:
            for e in self.elements:
                string += f"{e}\n"
        if self.distances:
            for d in self.distances:
                string += f"{d}\n"
        if self.angles:
            for ang in self.angles:
                string += f"{ang}\n"
        if self.dihedrals:
            for dih in self.dihedrals:
                string += f"{dih}\n"
        return string
