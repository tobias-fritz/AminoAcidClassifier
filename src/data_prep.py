import os
import requests
import contextlib
from urllib.request import urlopen
import torch
import concurrent.futures
from itertools import islice
import gzip
import time
from .dataset import AminoAcidDataset

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
        if line.startswith("ATOM") and line[17:20] in amino_acids and line[13:16] != "OXT" and line[13:14] != "H" and line[13:14] in ["N", "C", "O", "S"]:
            current_aa_num = line[22:26]
            counter += 1
            if current_aa_num != previous_aa_num:
                if previous_aa_name and counter != amino_acid_lengths[previous_aa_name]:
                    out_list = out_list[:-counter]
                    #print(f"Deleted {previous_aa_name} with {counter} entries")
                counter = 0
                if previous_aa_num:
                    out_list.append("END\n")
                previous_aa_num = current_aa_num
                previous_aa_name = line[17:20]
            out_list.append(line + "\n")
    out_list.append("END\n")
    return out_list

def run(pdb_info: list, out_path: str, download=True):
    """Main function to download PDB files, extract amino acids, and save them to a file.

    Args:   
        pdb_ids (list): List of PDB

    Returns:
        None
    """
    
    
    aas = []
    for i, (pdb_id, pdb_file) in enumerate(pdb_info):
        print(f"Processing {pdb_id} {i+1}/{len(pdb_info)}", end="\r")
        if not download:
            pdb = download_pdb(pdb_id)
        else:
            pdb = read_pdb(pdb_file)
        aas += get_amino_acids(pdb)

    while aas[0] == "END\n":
        aas.pop(0)

    elements_to_remove = []
    for i in range(len(aas) - 1):
        if aas[i] == "END\n" and aas[i + 1] == "END\n":
            elements_to_remove.append(i)

    for i in elements_to_remove[::-1]:
        aas.pop(i)

    print(f"Saving to {out_path}")
    with open(out_path, "w") as f:
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
    # Rotate the molecule by phi degrees around the x-axis
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, torch.cos(phi), -torch.sin(phi)],
                                    [0, torch.sin(phi), torch.cos(phi)]])
    rotated_residue = torch.matmul(coordinates, rotation_matrix)
    # Rotate the molecule by psi degrees around the y-axis
    rotation_matrix = torch.tensor([[torch.cos(psi), 0, torch.sin(psi)],
                                    [0, 1, 0],
                                    [-torch.sin(psi), 0, torch.cos(psi)]])
    rotated_residue = torch.matmul(rotated_residue, rotation_matrix)
    # Rotate the molecule by theta degrees around the z-axis
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
    print(f"Augmented dataset saved to {output_file}", end="\r")
    with open(output_file, 'w') as f:
        f.writelines(augmented_lines)

def process_pdb_res_download(pdbid: str) -> tuple:
    """Download a pdb file and extract resolution

    Arguments:
        pdbid (str): The pdb id to process

    Returns:
        tuple: The pdbid and resolution
    """

    # Download the pdb file
    pdb = download_pdb(pdbid) 
    # Extract the resolution
    try:
      resolution = get_resolution(pdb) 
    except:
      return pdbid, None
    # Return the pdbid and resolution
    if resolution:
        try:
            resolution = float(resolution)
            return pdbid, resolution
        except:
            return pdbid, None
    return pdbid, None


def get_resolution(pdb:str) -> str:
    """ Get the resolution of a PDB file.

    Args:
        pdb: PDB file content as a string

    Returns:
        Resolution of the PDB file
    """
    
    for line in pdb.split("\n")[:100]:
        #print(line)
        if line.startswith("REMARK   2 RESOLUTION."):
            resolution = line.split()[3]
            #print(resolution)
            return resolution
    return None

def read_pdb(fname: str) -> str:
    """ Read a PDB file.

    Args:
        fname: Path to the PDB file in .ent.gz format

    Returns:
        PDB file content as a string
    """

    with gzip.open(fname, 'rb') as file:
        content = file.read()
        pdb = content.decode('utf-8')
    return pdb

def process_pdb(pdb_info: tuple) -> tuple:
    """ Process a PDB file to get the resolution.

    Args:
        pdb_info: Tuple with PDB ID and PDB file path

    Returns:
        Tuple with PDB ID and resolution
    """

    pdbid, pdb_file = pdb_info
    pdb = read_pdb(pdb_file)
    #print(pdb)
    try:
      resolution = get_resolution(pdb)
    except:
      return pdbid, None
    if resolution:
        try:
            resolution = float(resolution)
            return pdbid, resolution
        except:
            return pdbid, None
    return pdbid, None

def batched(iterable: iter, n: int) -> iter:
    """ Yield batches of size n from an iterable.

    Args:
        iterable: Iterable object
        n: Batch size

    Returns:
        Batch of size n
    """
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

def estimate_time(batch_num: int, total: int, time_per_batch: int, num_batches: int) -> None:
    """ Print the estimated time remaining for the batch processing.

    Args:
        batch_num: Current batch number
        total: Total number of batches
        time_per_batch: Time taken per batch
        num_batches: Total number

    Returns:
        None
    """
    print(f"Batch {batch_num}/{total} done in {round(time_per_batch,1)}s. Estimated time remaining (min:s): \
          {time.strftime('%M:%S', time.gmtime(time_per_batch * (num_batches - batch_num)))}", end="\r")
    
def get_id_and_ressolution(subset_ids: list, subset_files: list, batch_size: int=40) -> dict:
    """ Get the resolution for a subset of PDBs

    Args:
        subset_ids: List of PDB IDs
        subset_files: List of PDB files
        batch_size: Number of PDBs to process in parallel

    Returns:
        Dictionary with PDB ID as key and resolution as value
    """
    pdb_resolution = {}

    # Process the PDBs in batches
    for batch_num, batch in enumerate(batched(zip(subset_ids, subset_files), batch_size)):
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_pdb, batch))
        estimate_time(batch_num, len(subset_ids)//batch_size, time.time() - start_time, len(subset_ids)//batch_size)
        # Store the results
        for pdbid, resolution in results:
            pdb_resolution[pdbid] = resolution
    return pdb_resolution


def download_all_pdbs(output_dir: str) -> None:
    """ Download all PDB files from the Protein Data Bank.

    This script is based on a bash script published on the PDB website and written by T. Solomon

    Args:
        output_dir: Directory to save the PDB files

    Returns:
        None
    """
    
    # Output directory
    MIRRORDIR = output_dir
    # Log file
    LOGFILE = os.path.join(output_dir, 'logs.txt')
    # Rsync location
    RSYNC = '/usr/bin/rsync'

    # Server and port
    SERVER = 'rsync.wwpdb.org::ftp'
    PORT = 33444

    # Run the rsync command
    os.system(f'{RSYNC} -rlpt -v -z --delete --port={PORT} {SERVER}/data/structures/divided/pdb/ {MIRRORDIR} > {LOGFILE} 2>/dev/null')

def get_all_files_and_ids():
    pdb_files = []
    base_dir = "data/all_pdbs"

    # get all pdb files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".ent.gz"):
                pdb_files.append(os.path.join(root, file))

    # extract the pdb ids
    pdb_ids = [f.split("/")[-1].split(".")[0][3:].upper() for f in pdb_files]

    return pdb_files, pdb_ids