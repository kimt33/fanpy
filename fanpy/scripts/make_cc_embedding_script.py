"""Code generating script."""
import os
import textwrap
import re

from fanpy.scripts.utils import check_inputs, parser


def make_script(  # pylint: disable=R1710,R0912,R0915
    script_filenames,
    atom_system_inds,
    ao_inds_file,
    filename=None,
    memory=None,
    constraint=None,
    wfn_noise=0,
    loc_type="pm",
    orbital_optimization=True,
):
    """Make a script for running calculations.

    Parameters
    ----------
    script_filenames : list of str
        Locations of the scripts that are used as references to copy over wavefunction
        initialization.
    atom_system_inds : list of int
        Indices of the system to which each atom belongs.

    """
    for script in script_filenames:
        if not os.path.isfile(script):
            raise ValueError("Given script must exist")
    if len(script_filenames) != len(set(atom_system_inds)):
        raise ValueError("Number of systems must be equal to the number of scripts")
    if set(atom_system_inds) != set(range(max(atom_system_inds) + 1)):
        raise ValueError("indices for the systems must not skip and values and must start from 0")

    # use first script as a reference
    with open(script_filenames[0], 'r') as f:
        output = f.read()

    output = re.sub(
            "from fanpy.upgrades import speedup_sign",
            """from fanpy.upgrades import speedup_sign
import fanpy.tools.slater as slater
from fanpy.wfn.cc.embedding import EmbeddedCC""",
            output
    )

    # FIXME: double check if indentation needed
    if loc_type == "svd":
        indices_list_assign = """indices_list[ao_ind].append(i)
indices_list[ao_ind].append(slater.spatial_to_spin_indices(i, nspin // 2, to_beta=True))"""
    else:
        indices_list_assign = """indices_list[system_inds[ao_ind]].append(i)
    indices_list[system_inds[ao_ind]].append(slater.spatial_to_spin_indices(i, nspin // 2, to_beta=True))"""

    wavefunction_preamble = f"""# System indices
system_inds = {atom_system_inds}

# Orbital labels (index of the atom to which each localized spatial orbital is assigned)
ao_inds = np.load('{ao_inds_file}')
indices_list = [[] for _ in range(max(system_inds) + 1)] 
for i, ao_ind in enumerate(ao_inds):
    {indices_list_assign}
indices_list = [sorted(i) for i in indices_list]

print("Spin orbital indices of each subsystem:")
print(indices_list)

# Number of electrons in each system
# assuming ground state corresponds to first N/2 alpha and N/2 beta orbitals occupied
npairs = nelec // 2
nspatial = nspin // 2
""" 
    # FIXME: hardcode electonrs

    wavefunction_preamble += "nelecs = []\n"
    for k in range(len(script_filenames)):
        wavefunction_preamble += f"""nelec{k+1} = len([i for i in indices_list[{k}] if i < npairs or nspatial <= i < nspatial + npairs])
nelecs.append(nelec{k+1})
print('Number of Electrons in System {k+1}: {{}}'.format(nelec{k+1}))
"""
    if orbital_optimization:
        wavefunction_preamble += fr"""# Add single excitation between systems
import itertools as it
from fanpy.tools.slater import ground, occ_indices, vir_indices
ground_sd = ground(nelec, nspin)
indices_occ = set(occ_indices(ground_sd))

inter_exops = []
for indices1, indices2 in it.combinations(indices_list, 2):
    # FIXME: split into occupied and virtual
    for i_occ in indices1:
        if i_occ not in indices_occ:
            continue
        for j_vir in indices2:
            if j_vir in indices_occ:
                continue
            inter_exops.append((i_occ, j_vir))
    for j_occ in indices2:
        if j_occ not in indices_occ:
            continue
        for i_vir in indices1:
            if i_vir in indices_occ:
                continue
            inter_exops.append((j_occ, i_vir))
inter_exops = list(set(inter_exops))
""" 
    else:
        wavefunction_preamble += fr"""inter_exops = []\n"""

    wavefunction_preamble += "\n# Initialize wavefunction 1"
    output = re.sub(r"# Initialize wavefunction", wavefunction_preamble, output)
    output = re.sub(r"wfn = ([\w\d]+)\(nelec, nspin", r"wfn1 = \1(nelec1, len(indices_list[0])", output)
    output = re.sub(f"wfn.assign_params\(.+?\n", "", output)
    output = re.sub(r"print\('Wavefunction: (.+?)'\)", r"print('Wavefunction in System 1: \1')\nwfn_list = [wfn1]", output)

    for i, script in enumerate(script_filenames[1:]):
        i += 2
        with open(script, 'r') as f:
            script_text = f.read()
        start = script_text.find("# Initialize wavefunction")
        end = script_text.find("# Initialize Hamiltonian")
        wfn_init = script_text[start: end]

        wfn_init = re.sub(r"# Initialize wavefunction", f"# Initialize wavefunction {i}", wfn_init)
        wfn_name = re.search(r"wfn = ([\w\d]+)\(nelec,", wfn_init).group(1)
        wfn_init = re.sub(fr"wfn = {wfn_name}\(nelec, nspin", fr"wfn{i} = {wfn_name}(nelec{i}, len(indices_list[{i-1}])", wfn_init)
        wfn_init = re.sub(f"wfn.assign_params\(.+?\n", "", wfn_init)
        wfn_init = re.sub(r"print\('Wavefunction: (.+?)'\)", fr"print('Wavefunction in System {i}: \1')\nwfn_list.append(wfn{i})", wfn_init)
        wfn_init += "# Initialize Hamiltonian"

        output = re.sub(r"# Initialize Hamiltonian", wfn_init, output)

        wfn_import = re.search(fr"\n(.+?{wfn_name})\n", script_text).group(1)
        output = re.sub("(import \w+Hamiltonian)", fr"\1\n{wfn_import}", output)

    nelecs = ", ".join([f"nelec{i}" for i in range(1, len(script_filenames) + 1)])
    output = re.sub(
        r"""# Initialize Hamiltonian""",
#        fr"""# Initialize wavefunction
#from fanpy.wfn.cc.embedding import EmbeddedCC
#wfn = EmbeddedCC([nelec1, nelec2], [len(indices) for indices in indices_list], indices_list, wfn_list, memory='6gb', params_list=None, ranks_list=ranks_list, exop_indices_list=None,
#                 inter_exops=inter_exops, refwfn_list=None,
#                 exop_combinations=None, refresh_exops=50000)
#print('Wavefunction: Embedded Fixed Electrons')
#
## Initialize Hamiltonian""", 
        fr"""# Initialize wavefunction
from fanpy.wfn.cc.embedding import EmbeddedCC
wfn = EmbeddedCC(nelecs, [len(indices) for indices in indices_list], indices_list, wfn_list, memory='6gb',
                 inter_exops=inter_exops, exop_combinations=None, refresh_exops=50000)
wfn.assign_params(wfn.params + {wfn_noise} * 2 * (np.random.rand(*wfn.params.shape) - 0.5))
print('Wavefunction: Embedded Fixed Electrons')

# Initialize Hamiltonian""", 
        output,
    )

    output = re.sub(r"(nproj = .+)wfn.nparams", r"\1sum(i.nparams for i in wfn_list)", output)

    if filename is None:  # pragma: no cover
        print(output)
    # NOTE: number was used instead of string (eg. 'return') to prevent problems arising from
    #       accidentally using the reserved string/keyword.
    elif filename == -1:
        return output
    else:
        with open(filename, "w") as f:  # pylint: disable=C0103
            f.write(output)


def main():  # pragma: no cover
    """Run script for run_calc using arguments obtained via argparse."""
    parser.description = "Optimize a wavefunction and/or Hamiltonian."
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        required=False,
        help="Name of the file that contains the output of the script.",
    )
    args = parser.parse_args()
    make_script(
        args.nelec,
        args.one_int_file,
        args.two_int_file,
        args.wfn_type,
        nuc_nuc=args.nuc_nuc,
        optimize_orbs=args.optimize_orbs,
        pspace_exc=args.pspace_exc,
        nproj=args.nproj,
        objective=args.objective,
        solver=args.solver,
        solver_kwargs=args.solver_kwargs,
        wfn_kwargs=args.wfn_kwargs,
        ham_noise=args.ham_noise,
        wfn_noise=args.wfn_noise,
        load_orbs=args.load_orbs,
        load_ham=args.load_ham,
        load_ham_um=args.load_ham_um,
        load_wfn=args.load_wfn,
        load_chk=args.load_chk,
        save_chk=args.save_chk,
        filename=args.filename,
        memory=args.memory,
    )