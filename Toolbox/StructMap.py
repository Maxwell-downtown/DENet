from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# This script calculates the contact map of the target protein, and stored it in
# the output directory under the 0.csv file. This file is
# used later in the GCN module of DENet to upgrade node embeddings. 

# Path to the PDB file of the target protein
pdb_file_path = './warehouse/KRAS_AF.pdb'
output_dir = './warehouse/KRAS_AF'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def calculate_contact_map(pdb_file, threshold=8.0):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('protein_structure', pdb_file)
    model = structure[0]  
    # Get the list of C-alpha atoms
    calphas = [atom for atom in model.get_atoms() if atom.get_name() == 'CA']
    # Initialize the contact map matrix
    num_residues = len(calphas)
    dis_map = np.zeros((num_residues, num_residues), dtype=int)
    contact_map = np.zeros((num_residues, num_residues), dtype=int)
    # Calculate the contacts
    for i, atom_i in enumerate(calphas):
        for j, atom_j in enumerate(calphas[i+1:], start=i+1):
            distance = (atom_i - atom_j)
            dis_map[i, j] = distance
            dis_map[j, i] = distance  
            if distance < threshold:
                contact_map[i][j] = 1
                contact_map[j][i] = 1
    return contact_map

# Threshold distance is set to 8 Ångströms in default)
threshold_distance = 8.0

cm0 = calculate_contact_map(pdb_file_path, threshold_distance)
df0 = pd.DataFrame(cm0)
out_file = output_dir + '/0.csv'
df0.to_csv(out_file, index=False, header=None)