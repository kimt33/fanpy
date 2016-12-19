from __future__ import absolute_import, division, print_function
from nose.plugins.attrib import attr
from wfns.proj.solver import solve
from wfns.proj.apfsg import APfsG
from wfns.wrapper.horton import gaussian_fchk

@attr('slow')
def test_apfsg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # FCI Value :      -1.87832550029
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

    nelec = 2
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # Reproduce HF energy
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apfsg.params[:-1] = apfsg.template_coeffs.flatten()
    apfsg.cache = {}
    apfsg.d_cache = {}
    assert abs(apfsg.compute_energy(include_nuc=False, ref_sds=apfsg.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(apfsg, jac=False)
    print('HF energy', -1.84444667247)
    print('new energy', apfsg.compute_energy())
    print('FCI value', -1.87832550029)
    print(apfsg.params)
    assert -1.84444667247 > apfsg.compute_energy() > -1.87832550029
    assert abs(apfsg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7


@attr('slow')
def test_apfsg_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # FCI Value :      -8.96741814557
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')

    nelec = 4
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # Reproduce HF energy
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apfsg.params[:-1] = apfsg.template_coeffs.flatten()
    apfsg.cache = {}
    apfsg.d_cache = {}
    assert abs(apfsg.compute_energy(include_nuc=False, ref_sds=apfsg.default_ref_sds) - (-8.9472891719)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(apfsg, jac=False)
    print('HF energy', -8.9472891719)
    print('new energy', apfsg.compute_energy())
    print('FCI value', -8.96741814557)
    assert -8.9472891719 > apfsg.compute_energy() > -8.96741814557
    assert False
