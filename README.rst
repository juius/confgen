
ConformerGenerator
==================

Example using ETKDG
-------------------

.. code-block:: python

    from confgen.etkdg import ETKDG

    # Initialize ConformerGenerator
    cgen_etkdg = ETKDG(n_confs=100, pruneRmsThresh=0.1, n_cores=2)

    # Initialize rdkit.mol object without any conformers
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)NCC(=O)O")
    mol = Chem.AddHs(mol)

    # Generate Conformers
    mol3d = cgen_etkdg.generate(mol)

    mol3d.GetNumConformers()
    # >> 14


Atoms can be constrained in the embedding process by passing a list of atom indices to the ``constrained_atoms`` argument.
This requires the presence of one 3D conformer in the molecule, the atoms will the constrained to their respective position in this conformer.

Alternatively, a coordMap dictionary with atom indices as keys and either lists, numpy arrays or Point3Ds as values can be passed to ``constrained_atoms``.

.. code-block:: python

    # constrain atoms with the following indices to their respective position in the first conformer
    constrained_atoms = [2,3,4,5,16]

    # constrain atoms with the following indices to the positions given in the dictionary
    constrained_atoms = {2: [0,0,0], 3: [2,0,0], 5: [2,2,0]}

    mol3d = cgen_etkdg.generate(mol, constrained_atoms=constrained_atoms)


Example using CREST
-------------------
CREST keywords can be found `here <https://xtb-docs.readthedocs.io/en/latest/crestcmd.html>`_


.. code-block:: python

    from confgen.crest import CREST

    # Initialize ConformerGenerator
    cgen_crest = CREST(gfn=2, ewin=6, mquick=True, mdlen='x0.1', n_cores=2)

    # Initialize rdkit.mol object without any conformers
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)NCC(=O)O")
    mol = Chem.AddHs(mol)

    # Generate Conformers
    mol3d = cgen_crest.generate(mol)

    mol3d.GetNumConformers()
    # >> 9

Constrains can be parsed to CREST as a list of atom and/or bond indices:

.. code-block:: python

    pattern = Chem.MolFromSmarts('NC(=O)CN')
    match = list(mol.GetSubstructMatch(pattern))
    # >> [5, 3, 4, 1, 2]

    mol3d = cgen_crest.generate(mol, constrained_atoms=match, constrained_bonds=[(1,4),(4,5)])


Tools
===============

Geometry Optimization
---------------------

Availabel methods: ``uff``, ``mmff`` and ``gfnff``, ``gfn1`` and ``gfn2``
The method ``run`` will return a new rdkit.mol object with the optimized conformers sorted with ascending energies.


.. code-block:: python

    from confgen.tools import GeomOptimizer

    gfn2_opt = GeomOptimizer(method='gfn2', n_cores=2)

    mol3d_opt = gfn2_opt.run(mol3d)

    conformer_energies = [conf.GetDoubleProp('energy') for conf in mol3d_opt.GetConformers()]


RMSD Clustering
---------------

The ``keep`` argument specifies which conformer of each cluster should be retained, either the lowest energy conformer (``lowenergy``) or the centroid (``centroid``).

.. code-block:: python

    from confgen.tools import Cluster

    rmsd_cluster = Cluster(rmsdThreshold=0.5, keep='lowenergy')

    clustered = rmsd_cluster.run(mol3d_opt)

    clustered.GetNumConformers()
    # >> 8


Energy Filter
---------------

The ``ewin`` argument specifies the energy window in which conformers should be retained (up from the lowest energy conformer found).

.. code-block:: python

    from confgen.tools import Filter

    filtered = Filter(ewin=2).run(mol3d_opt)
    filtered.GetNumConformers()
    # >> 6
