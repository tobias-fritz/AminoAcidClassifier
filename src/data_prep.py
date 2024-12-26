#

# Download a pdb from the protein data base

import os
import requests
import contextlib
from urllib.request import urlopen

def download_pdb(pdb_id, save_file= False,output_dir=None):
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
    url = "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
    with contextlib.closing(urlopen(url)) as handle:
        all_pdbids = [line[:4].decode() for line in handle.readlines()[2:] if len(line) > 4]
    return all_pdbids



def get_amino_acids(pdb:str):

    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", 
                   "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", 
                   "MET", "PHE", "PRO", "SER", "THR", "TRP", 
                   "TYR", "VAL"]
    amino_acid_lengths = {"ALA":5, "ARG":11, "ASN":8, "ASP":8, "CYS":6, "GLN":9, 
                          "GLU":9, "GLY":4, "HIS":10, "ILE":8, "LEU":8, "LYS":9, 
                          "MET":8, "PHE":11, "PRO":7, "SER":6, "THR":7, "TRP":14, 
                          "TYR":12, "VAL":7}
    out_list = []
    previous_aa_num = None
    counter = 0
    previous_aa_name = None
    
    for line in pdb.split("\n"):
        if line.startswith("ATOM") and line[17:20] in amino_acids and line[13:16] != "OXT":
            current_aa_num = line[22:26] 
            counter += 1
            if current_aa_num != previous_aa_num:
                # check if the counter matches the length of the previous aa, if not, delete the previous aa (last couner entries)
                if previous_aa_name and counter != amino_acid_lengths[previous_aa_name]:
                    out_list = out_list[:-counter]
                    print(f"Deleted {previous_aa_name} with {counter} entries")
                counter = 0
                if previous_aa_num:
                    out_list.append("END\n")
                previous_aa_num = current_aa_num
                previous_aa_name = line[17:20]
                
            out_list.append(line+"\n")
    out_list.append("END\n")
    return out_list


def run():
    # get all pdb names:
    all_pdbids = list_all_pdbids()
    n = 300
    aas= []
    for i, pdb_id in enumerate(all_pdbids[:n]):
        print(f"Downloading {pdb_id} {i+1}/{len(all_pdbids[:n])}", end="\r")
        pdb = download_pdb(pdb_id)
        # for each pdb extract the amino acids
        aas += get_amino_acids(pdb)

    # remove all leading "END\n" lines
    while aas[0] == "END\n":
        aas.pop(0)

    # whenever there's 2 "END\n" lines, remove the first one
    elements_to_remove = []

    for i in range(len(aas)-1):
        if aas[i] == "END\n" and aas[i+1] == "END\n":
            elements_to_remove.append(i)

    #remove the elements in reverse order
    for i in elements_to_remove[::-1]:
        aas.pop(i)

    # write aas to a file
    with open("amino_acids.pdb", "w") as f:
        f.write("".join(aas))

@deprecated
def get_amino_acids_from_file(pdb_file, out_file=None, write=False):

    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", 
                   "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", 
                   "MET", "PHE", "PRO", "SER", "THR", "TRP", 
                   "TYR", "VAL"]
    # extract all aa with coordinates from the pdb file
    with open(pdb_file) as f:
        lines = f.readlines()
        out_list = []
        previous_aa_num = None
        for line in lines:
            # make sure the AA is a valid AA and atom type is not OXT
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


