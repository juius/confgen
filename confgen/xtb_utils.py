import copy
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from confgen.utils import normal_termination, stream

XTB_CMD = "xtb"


def set_threads(n_cores):
    """Set threads and procs environment variables"""
    os.environ["OMP_NUM_THREADS"] = f"{n_cores},1"
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


def check_xtb(version=None, logger=None):
    """Check xTB executable and version"""
    lines = stream(f"{XTB_CMD} --version")
    lines = list(lines)
    if normal_termination(lines, "normal termination"):
        if version:
            for l in lines:
                if "xtb version" in l:
                    assert l.split()[3] == version, f"Requires xtb version {version}"
            # if "Mac" in l:
            #     raise Warning("Does not work with Mac compiled xTB")
    else:
        raise Warning("Could not find xTB")


def write_xyz(atoms, coords, scr):
    """Write .xyz file"""
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
    """Add Conformer to rdkit.mol object"""
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        # assert that same atom type
        assert (
            mol.GetAtomWithIdx(i).GetSymbol() == atoms[i]
        ), f"Order of atoms if not the same in CREST output and rdkit Mol"
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    if energy:
        conf.SetDoubleProp("energy", float(energy))
    mol.AddConformer(conf, assignId=True)


import time


def run_xtb_opt(args):
    """Runs xTB command for xyz-file in parent directory and returns optimized structure"""
    cmd, xyz_file = args
    start = time.time()
    lines = stream(f"{cmd}-- {xyz_file.name}", cwd=xyz_file.parent)
    lines = list(lines)
    end = time.time()
    print("time:", end - start)
    if normal_termination(lines, "normal termination of xtb"):
        return read_opt_structure_and_energy(lines)
    else:
        return None


def read_opt_structure_and_energy(lines):
    """Reads optimized structure and energy from xTB output"""
    for i, l in reversed(list(enumerate(lines))):
        if "TOTAL ENERGY" in l:
            energy = float(l.split()[-3])
        elif "final structure" in l:
            break

    n_atoms = int(lines[i + 2].rstrip())
    start = i + 4
    end = start + n_atoms

    atoms = []
    coords = []
    for l in lines[start:end]:
        atom, coord = parse_coordline(l)
        atoms.append(atom)
        coords.append(coord)

    n_atoms = int(lines[i + 2].rstrip())
    start = i + 4
    end = start + n_atoms

    atoms = []
    coords = []
    for l in lines[start:end]:
        atom, coord = parse_coordline(l)
        atoms.append(atom)
        coords.append(coord)
    return atoms, coords, energy


def xtb_optimize(mol, options, n_cores, scr="."):
    """Optimizes each conformer of rdkit.mol in parallel"""

    # Only use one core for each xTB calculation
    set_threads(1)
    if "opt" not in options:
        options["opt"] = None

    # Creat TMP directory
    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix=f"XTBOPT_")
    tmp_scr = Path(tempdir.name)

    # Write xyz-file for each conformer
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz_files = []
    n_confs = mol.GetNumConformers()
    for i, conf in enumerate(mol.GetConformers()):
        confdir = tmp_scr / "CONF{num:0{width}}".format(num=i, width=len(str(n_confs)))
        os.makedirs(confdir)
        coords = conf.GetPositions()
        xyz_file = write_xyz(atoms, coords, confdir)
        xyz_files.append(xyz_file)

    # Options to xTB command
    cmd = f"{XTB_CMD} "
    for key, value in options.items():
        if value is None:
            cmd += f"--{key} "
        else:
            if not "constrain" in key:
                cmd += f"--{key} {str(value)} "

    # write constrains
    if "constrain_atoms" in options and len(options["constrain_atoms"]) > 0:
        with open(tmp_scr / "constrains.inp", "w") as f:
            f.write("$fix\n")
            f.write(
                f"    atoms: {','.join([str(a+1) for a in options['constrain_atoms']])}\n"
            )
            f.write("$end")
        cmd += "--input ../constrains.inp "

    # xTB optimize each conformer in parallel
    args = []
    for xyz_file in xyz_files:
        args.append((cmd, xyz_file))

    with Pool(n_cores) as pool:
        results = pool.map(run_xtb_opt, args)
    results = list(results)

    # Add optimized conformers to mol_opt
    mol_opt = copy.deepcopy(mol)
    n_confs = mol_opt.GetNumConformers()
    # Remove all but first conformers
    _ = [mol_opt.RemoveConformer(i) for i in range(1, n_confs)]
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
    if "constrain_atoms" in options and len(options["constrain_atoms"]) > 0:
        rms = AllChem.AlignMolConformers(mol_opt, atomIds=options["constrain_atoms"])
    return mol_opt
