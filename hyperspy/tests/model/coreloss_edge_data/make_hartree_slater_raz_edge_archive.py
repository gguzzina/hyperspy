import numpy as np
from pathlib import Path
import hyperspy.api as hs
from hyperspy.misc import elements


shell_dict = {'K': 1, 'L':2, 'M': 3, 'N':4, 'O':5}

def generate_edge(Z, shell, alpha, beta, E0, npoints):
    element = elements.atomic_number2name[Z]
    edges = []
    edges_db = elements.elements_db[element]['Atomic_properties']['Binding_energies']
    for e in edges_db:
        edge_name = e[0]
        if edge_name[0] == shell:
            edges.append(e)
    energies = [e[1][' onset_energy (eV)'] for e in edges]
    emin = int(round(min(energies) * 0.95))
    emax = int(round(max(energies) * 1.1))

    spectrum = np.ones(npoints)
    sig = hs.signals.EELSSpectrum(spectrum)
    sig.axes_manager[0].offset = emin
    sig.axes_manager[0].scale = (emax - emin) / npoints

    sig.set_microscope_parameters(convergence_angle=alpha, collection_angle=beta, beam_energy=E0)
    try:
        sig.add_elements( (element,) )
        mod = sig.create_model(auto_background=False, GOS='Hartree-Slater')
        s_mod = mod.as_signal()
    except Exception as e:
        print(e)
    return s_mod, (emin, emax)


def simulate_edges_for_element(Z, E0, alpha, beta, npoints):
    element = elements.atomic_number2name[Z]
    try:
        edges_db = elements.elements_db[element]['Atomic_properties']['Binding_energies']
        edge_archive = []
        shells = []
        for e in edges_db:
            if e[0][0] not in shells :
                shells.append(e[0][0])
        for s in shells:
            s_mod, (emin, emax) = generate_edge(Z, s, alpha, beta, E0, npoints)
            edgename = '{}_{}'.format(element, s)
            metadata = {'convergence_angle':alpha, 'collection_angle':beta, 'beam_energy':E0}
            edge_dict = {
                    'name': edgename,
                    'emin': emin,
                    'emax': emax,
                    'metadata': metadata,
                    'spectrum': s_mod.data}
            edge_archive.append(edge_dict)
        return edge_archive

    except AttributeError:
        print('Skip {}'.format(Z))
        pass


def make_list_of_edge_data(Zs, E0=100, alpha=5, beta=10, npoints=256):
    edges_archive = []
    for Z in Zs:
        edges_archive_partial = simulate_edges_for_element(
                Z=Z,
                E0=E0,
                alpha=alpha,
                beta=beta,
                npoints=npoints,
        )
        if edges_archive_partial is not None:
            for e in edges_archive_partial:
                edges_archive.append(e)
    return edges_archive


def make_file_with_h_s_spectra(
        filename=None,
        z_range=None,
        h_s_gos_file_path=None,
    ):
    if filename is None:
        filename = "hartree_slater_raz_edge_archive.npz"
    if z_range is None:
        z_range = range(3, 84)
    if h_s_gos_file_path is None:
        gos_path = Path(hs.preferences.EELS.eels_gos_files_path)
        if not gos_path.is_dir():
            raise Exception(
                "H-S GOS not found, specify the path with the "
                "h_s_gos_file_path parameter"
            )
    else:
        hs.preferences.EELS.eels_gos_files_path = h_s_gos_file_path

    edges_archive = make_list_of_edge_data(z_range)
    spectrum_list = []
    emin_list = []
    emax_list = []
    name_list = []
    for edge in edges_archive:
        spectrum = edge["spectrum"]
        emin = edge["emin"]
        emax = edge["emax"]
        name = edge["name"]
        spectrum_list.append(spectrum)
        emin_list.append(emin)
        emax_list.append(emax)
        name_list.append(name)

    convergence_angle = edge["metadata"]["convergence_angle"]
    collection_angle = edge["metadata"]["collection_angle"]
    beam_energy = edge["metadata"]["beam_energy"]
    np.savez(
        file=filename,
        name_list=name_list,
        emin_list=emin_list,
        emax_list=emax_list,
        spectrum_list=spectrum_list,
        convergence_angle=convergence_angle,
        collection_angle=collection_angle,
        beam_energy=beam_energy,
    )

make_file_with_h_s_spectra()
