
ConformerGenerator
===============

Example using ETKDG
---------------

.. code-block:: python

    from confgen import ConformerGenerator
    from confgen.tools import GeomOptimizer, Filter, Cluster

    # ETKDG options
    etkdg_options = {
        'n_confs': 1000,
        'embed_threshold': 0.05,
        'opt_flow': [
            # GFN-FF optimization of all conformers in methanol
            GeomOptimizer('gfnff', options={'alpb': 'methanol'}),
            # keep conformers within 3 kcal/mol of lowest energy one
            Filter(3),
            # cluster conformers based on RMSD and keep lowest energy
            # conformer of each cluster (or 'centroid')
            Cluster(0.25, keep='lowenergy'),
            # GFN2 optimization
            GeomOptimizer('gfn2', options={'alpb': 'methanol'}),
            ]
    }

    # Initialize ConformerGenerator
    cgen = ConformerGenerator(method='etkdg', options=etkdg_options, n_cores=2, verbose=True)

    # Initialize rdkit.mol object without any conformers
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)NCC(=O)O")
    mol = Chem.AddHs(mol)

    # Generate Conformers
    mol = cgen.generate(mol)

    mol.GetNumConformers() # --> 48 Conformers

Atoms can be constrained in the embedding and optimization

.. code-block:: python

    constrain_atoms = [2,3,4,5,16]

    # ETKDG options
    etkdg_options = {
        ...,
        'constrain_atoms': constrain_atoms,
        'opt_flow': [
            GeomOptimizer('gfnff', options={'alpb': 'methanol', 'constrain_atoms': constrain_atoms}),
            ...
            ]
    }

    ...


Example using CREST
---------------
CREST keywords can be found `here <https://xtb-docs.readthedocs.io/en/latest/crestcmd.html>`_


.. code-block:: python

    # CREST options
    crest_options = {'gfn': 1,
                     'ewin': 3,
                     'mdlen': 'x0.5',
                     'mquick': None}

    # Initialize ConformerGenerator
    cgen = ConformerGenerator(method='crest', options=crest_options, n_cores=2)

    # Initialize rdkit.mol object without any conformers
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)NCC(=O)O")
    mol = Chem.AddHs(mol)

    # Generate Conformers
    mol = cgen.generate(mol)

    mol.GetNumConformers() # --> 9 Conformers

Constrains can be parsed to CREST as a list of atom and/or bond indices:

.. code-block:: python

    pattern = Chem.MolFromSmarts('NC(=O)CN')
    match = list(mol.GetSubstructMatch(pattern))
    # [5, 3, 4, 1, 2]

    # CREST options
    crest_options = {'constrain_atoms': match,
                     'constrain_bonds': [(1,4),(4,5)]
                     ...
                     }
