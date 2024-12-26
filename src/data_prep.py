import os
import requests
import contextlib
from urllib.request import urlopen

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


