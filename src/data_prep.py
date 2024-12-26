import os
import requests
import contextlib
from urllib.request import urlopen
import torch
from .data import AminoAcidDataset

def download_pdb(pdb_id, save_file=False, output_dir=None):
    """
    Download a PDB file from the Protein Data Bank.

    Args:
        pdb_id (str): The PDB ID of the protein.
        save_file (bool): Whether to save the file locally.
        output_dir (str): Directory to save the file if save_file is True.

    Returns:
        str: The content of the PDB file.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        if save_file:
            with open(os.path.join(output_dir, f"{pdb_id}.pdb"), "w") as f:
                f.write(response.text)
        return response.text
    else:
        print(f"Failed to download {pdb_id}")

def list_all_pdbids():
    """
    List all PDB IDs available in the Protein Data Bank.

    Returns:
        list: A list of all PDB IDs.
    """
    url = "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
    with contextlib.closing(urlopen(url)) as handle:
        all_pdbids = [line[:4].decode() for line in handle.readlines()[2:] if len(line) > 4]
    return all_pdbids

def get_amino_acids(pdb: str):
    """
    Extract amino acids from a PDB file content.

    Args:
        pdb (str): The content of the PDB file.

    Returns:
        list: A list of amino acid lines.
    """
    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", 
                   "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", 
                   "MET", "PHE", "PRO", "SER", "THR", "TRP", 
                   "TYR", "VAL"]
    amino_acid_lengths = {"ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6, "GLN": 9, 
                          "GLU": 9, "GLY": 4, "HIS": 10, "ILE": 8, "LEU": 8, "LYS": 9, 
                          "MET": 8, "PHE": 11, "PRO": 7, "SER": 6, "THR": 7, "TRP": 14, 
                          "TYR": 12, "VAL": 7}
    out_list = []
    previous_aa_num = None
    counter = 0
    previous_aa_name = None

    for line in pdb.split("\n"):
        if line.startswith("ATOM") and line[17:20] in amino_acids and line[13:16] != "OXT":
            current_aa_num = line[22:26]
            counter += 1
            if current_aa_num != previous_aa_num:
                if previous_aa_name and counter != amino_acid_lengths[previous_aa_name]:
                    out_list = out_list[:-counter]
                    print(f"Deleted {previous_aa_name} with {counter} entries")
                counter = 0
                if previous_aa_num:
                    out_list.append("END\n")
                previous_aa_num = current_aa_num
                previous_aa_name = line[17:20]
            out_list.append(line + "\n")
    out_list.append("END\n")
    return out_list

def run():
    """
    Main function to download PDB files, extract amino acids, and save them to a file.
    """
    all_pdbids = list_all_pdbids()
    n = 300
    aas = []
    for i, pdb_id in enumerate(all_pdbids[:n]):
        print(f"Downloading {pdb_id} {i+1}/{len(all_pdbids[:n])}", end="\r")
        pdb = download_pdb(pdb_id)
        aas += get_amino_acids(pdb)

    while aas[0] == "END\n":
        aas.pop(0)

    elements_to_remove = []
    for i in range(len(aas) - 1):
        if aas[i] == "END\n" and aas[i + 1] == "END\n":
            elements_to_remove.append(i)

    for i in elements_to_remove[::-1]:
        aas.pop(i)

    with open("amino_acids.pdb", "w") as f:
        f.write("".join(aas))

def get_amino_acids_from_file(pdb_file, out_file=None, write=False):
    """
    Extract amino acids from a PDB file and optionally write to an output file.

    Args:
        pdb_file (str): Path to the PDB file.
        out_file (str): Path to the output file.
        write (bool): Whether to write the output to a file.

    Returns:
        list: A list of amino acid lines.
    """
    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", 
                   "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", 
                   "MET", "PHE", "PRO", "SER", "THR", "TRP", 
                   "TYR", "VAL"]
    with open(pdb_file) as f:
        lines = f.readlines()
        out_list = []
        previous_aa_num = None
        for line in lines:
            if line.startswith("ATOM") and line[17:20] in amino_acids and line[13:16] != "OXT":
                current_aa_num = line[22:26]
                if current_aa_num != previous_aa_num:
                    out_list.append("END\n")
                    previous_aa_num = current_aa_num
                out_list.append(line)
        out_list.append("END\n")

    if write:
        with open(out_file, "w") as f:
            f.write("".join(out_list))
    return out_list


def center_residue_to_first_atom(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Center the coordinates to the first atom.

    Args:
        coordinates (torch.Tensor): Tensor of atom coordinates.

    Returns:
        torch.Tensor: Centered coordinates.
    """
    # center the coordinates to the first atom
    first_atom = coordinates[0]
    centered_coordinates = coordinates - first_atom
    
    return centered_coordinates

def rotate_residue(coordinates: torch.Tensor, phi: float, psi: float, theta: float) -> torch.Tensor:
    """
    Rotate the residue by given angles around the x, y, and z axes.

    Args:
        coordinates (torch.Tensor): Tensor of atom coordinates.
        phi (float): Rotation angle around the x-axis in degrees.
        psi (float): Rotation angle around the y-axis in degrees.
        theta (float): Rotation angle around the z-axis in degrees.

    Returns:
        torch.Tensor: Rotated coordinates.
    """
    phi, psi, theta = torch.deg2rad(torch.tensor(phi)), torch.deg2rad(torch.tensor(psi)), torch.deg2rad(torch.tensor(theta))
    # rotate the molecule by phi degrees around the x-axis
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, torch.cos(phi), -torch.sin(phi)],
                                    [0, torch.sin(phi), torch.cos(phi)]])
    rotated_residue = torch.matmul(coordinates, rotation_matrix)
    # rotate the molecule by psi degrees around the y-axis
    rotation_matrix = torch.tensor([[torch.cos(psi), 0, torch.sin(psi)],
                                    [0, 1, 0],
                                    [-torch.sin(psi), 0, torch.cos(psi)]])
    rotated_residue = torch.matmul(rotated_residue, rotation_matrix)
    # rotate the molecule by theta degrees around the z-axis
    rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                    [torch.sin(theta), torch.cos(theta), 0],
                                    [0, 0, 1]])
    rotated_residue = torch.matmul(rotated_residue, rotation_matrix)

    return rotated_residue

def augment_dataset(dataset: AminoAcidDataset, output_file: str, n_orientations: int = 1) -> None:
    """
    Augment the dataset by generating multiple orientations of each residue.

    Args:
        dataset (AminoAcidDataset): The dataset to augment.
        output_file (str): The file to save the augmented dataset.
        n_orientations (int, optional): Number of orientations to generate for each residue. Defaults to 1.
    """
    residue_to_name = {
        0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS', 5: 'GLN', 6: 'GLU',
        7: 'GLY', 8: 'HIS', 9: 'ILE', 10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE',
        14: 'PRO', 15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL'
    }
    element_to_name = ['C', 'N', 'O', 'S']

    augmented_lines = []
    for i in range(len(dataset)):
        coordinates, elements, residue = dataset[i]
        for _ in range(n_orientations):
            centered_residue = center_residue_to_first_atom(coordinates)
            rotated_coordinates = rotate_residue(
                centered_residue,
                torch.randint(0, 360, (1,)).item(),
                torch.randint(0, 360, (1,)).item(),
                torch.randint(0, 360, (1,)).item()
            )
            for j in range(len(rotated_coordinates)):
                augmented_lines.append(
                    f'ATOM {j + 1:6d} {element_to_name[int(torch.argmax(elements[j]))]:>2}   '
                    f'{residue_to_name[int(torch.argmax(residue))]:>3}     1    '
                    f'{rotated_coordinates[j][0]:8.3f}{rotated_coordinates[j][1]:8.3f}'
                    f'{rotated_coordinates[j][2]:8.3f}    1.00  0.00\n'
                )
            augmented_lines.append('END\n')
    with open(output_file, 'w') as f:
        f.writelines(augmented_lines)


