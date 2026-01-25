import io
import urllib
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import os
import logging

import torch
from Bio import SeqIO
from Bio.PDB import PDBList
from transformers import EsmPreTrainedModel, PreTrainedTokenizer

from .protein_embedding import embed_sequence
from .prtein_feature_extractor import (
    load_structure, run_dssp, extract_residue_level_features, per_chain_relative_distance_matrices, ENMGreenFunction,
    get_chain_sequence
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AMINO_ACID_MASSES = {
    'A': 71.0788, 'R': 156.1875, 'N': 114.1038, 'D': 115.0886, 'C': 103.1388,
    'E': 129.1155, 'Q': 128.1292, 'G': 57.0519, 'H': 137.1411, 'I': 113.1594,
    'L': 113.1594, 'K': 128.1741, 'M': 131.1926, 'F': 147.1766, 'P': 97.1167,
    'S': 87.0782, 'T': 101.1051, 'W': 186.2139, 'Y': 163.1760, 'V': 99.1326
}

AMINO_ACID_DIC = {
    'A': [0, 0, 0, 0, 0], 'R': [0, 0, 0, 0, 1], 'N': [0, 0, 0, 1, 0], 'D': [0, 0, 0, 1, 1], 'C': [0, 0, 1, 0, 0],
    'E': [0, 0, 1, 0, 1], 'Q': [0, 0, 1, 1, 0], 'G': [0, 0, 1, 1, 1], 'H': [0, 1, 0, 0, 0], 'I': [0, 1, 0, 0, 1],
    'L': [0, 1, 0, 1, 0], 'K': [0, 1, 0, 1, 1], 'M': [0, 1, 1, 0, 0], 'F': [0, 1, 1, 0, 1], 'P': [0, 1, 1, 1, 0],
    'S': [0, 1, 1, 1, 1], 'T': [1, 0, 0, 0, 0], 'W': [1, 0, 0, 0, 1], 'Y': [1, 0, 0, 1, 1], 'V': [1, 0, 1, 1, 1],
    'ALA': [0, 0, 0, 0, 0],
    'ARG': [0, 0, 0, 0, 1],
    'ASN': [0, 0, 0, 1, 0],
    'ASP': [0, 0, 0, 1, 1],
    'CYS': [0, 0, 1, 0, 0],
    'GLU': [0, 0, 1, 0, 1],
    'GLN': [0, 0, 1, 1, 0],
    'GLY': [0, 0, 1, 1, 1],
    'HIS': [0, 1, 0, 0, 0],
    'ILE': [0, 1, 0, 0, 1],
    'LEU': [0, 1, 0, 1, 0],
    'LYS': [0, 1, 0, 1, 1],
    'MET': [0, 1, 1, 0, 0],
    'PHE': [0, 1, 1, 0, 1],
    'PRO': [0, 1, 1, 1, 0],
    'SER': [0, 1, 1, 1, 1],
    'THR': [1, 0, 0, 0, 0],
    'TRP': [1, 0, 0, 0, 1],
    'TYR': [1, 0, 0, 1, 1],
    'VAL': [1, 0, 1, 1, 1]
}

SEC_CODE = {"H": [0, 0], "G": [0, 0], "I": [0, 0],        # helix
            "E": [0, 1], "B": [0, 1],                # strand
            "T": [1, 0], "S": [1, 0],                # turn / bend
            "-": [1, 1]}                        # loop / unknown

# Approximate contour length per amino acid in unfolded state (in nm)
# This is a typical value used in WLC modeling
CONTOUR_LENGTH_PER_AA = 0.36 # nm

def calculate_contour_length(amino_acid_sequence: str) -> float:
    """
    Calculates the approximate contour length of a protein sequence
    in its fully extended (unfolded) state.

    Args:
        amino_acid_sequence (str): The protein sequence string.

    Returns:
        float: The approximate contour length in nanometers.
    """
    if not isinstance(amino_acid_sequence, str) or not amino_acid_sequence:
        logging.warning("Invalid or empty amino acid sequence provided for contour length calculation.")
        return 0.0
    # Basic check for valid amino acids (case-insensitive)
    valid_aa_sequence = ''.join(c for c in amino_acid_sequence.upper() if c in AMINO_ACID_MASSES)
    if len(valid_aa_sequence) != len(amino_acid_sequence):
         logging.warning(f"Sequence contains unexpected characters. Using valid AA count. Original: {amino_acid_sequence}, Valid: {valid_aa_sequence}")

    num_residues = len(valid_aa_sequence)
    return num_residues * CONTOUR_LENGTH_PER_AA

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data from a specified file path using pandas.
    Assumes data is in a tabular format (CSV, Excel, etc.).

    Args:
        file_path (str): Path to the raw data file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
        Exception: For other loading errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        # Add support for other formats as needed
        else:
            raise ValueError(f"Unsupported raw data file format: {os.path.basename(file_path)}")
        logging.info(f"Successfully loaded raw data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading raw data from {file_path}: {e}")
        raise

def save_processed_data(data, file_path: str):
    """
    Saves processed data to a specified file path.
    Supports saving pandas DataFrames to CSV and numpy arrays to .npy.

    Args:
        data: The processed data (pandas DataFrame or numpy array).
        file_path (str): Path to save the processed data.

    Raises:
        ValueError: If the data type or file format is unsupported.
        Exception: For other saving errors.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
    try:
        if isinstance(data, pd.DataFrame) and file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif isinstance(data, np.ndarray) and file_path.endswith('.npy'):
             np.save(file_path, data)
        elif isinstance(data, (list, np.ndarray)) and file_path.endswith('.pkl'):
             # Saving list/array to pickle can be convenient for arbitrary structures
             pd.to_pickle(data, file_path)
        # Add support for other formats as needed
        else:
            raise ValueError(f"Unsupported data type ({type(data)}) or file format ({os.path.basename(file_path)}) for saving processed data.")
        logging.info(f"Successfully saved processed data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving processed data to {file_path}: {e}")
        raise


def create_cath_maps(cath_ids: List[str]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Creates mapping dictionaries from CATH IDs to integer indices for each level.

    Args:
        cath_ids: A list of CATH ID strings (e.g., ['1.10.490.10', '1.10.490.20']).

    Returns:
        A tuple of four dictionaries for C, A, T, and H levels.
    """
    all_c, all_a, all_t, all_h = set(), set(), set(), set()
    for cath_id in cath_ids:
        parts = cath_id.split('.')
        if len(parts) == 4:
            all_c.add(parts[0])
            all_a.add(parts[1])
            all_t.add(parts[2])
            all_h.add(parts[3])

    c_map = {id: i for i, id in enumerate(sorted(list(all_c)))}
    a_map = {id: i for i, id in enumerate(sorted(list(all_a)))}
    t_map = {id: i for i, id in enumerate(sorted(list(all_t)))}
    h_map = {id: i for i, id in enumerate(sorted(list(all_h)))}

    return c_map, a_map, t_map, h_map


def _ensure_mmCIF(pdb_id_or_path: str, feature_path: str) -> str:
    """
    Return a local path to mmCIF given a file path or a 4-char PDB id.
    """
    in_path = Path(pdb_id_or_path)
    if in_path.exists():
        return str(in_path)

    pdb_id = pdb_id_or_path.lower()
    os.makedirs(feature_path, exist_ok=True)
    # Bio.PDB downloads as .../pdb_id.cif (or subdir). We return the file it reports.
    cif_path = PDBList().retrieve_pdb_file(pdb_id, pdir=feature_path, file_format="mmCif", overwrite=False)
    return str(Path(cif_path).resolve())


def _slice_chain_submatrix(df: pd.DataFrame, full_map: np.ndarray, chain: str) -> np.ndarray:
    """
    Select L×L submatrix for one chain based on residue order in df.
    Assumes df rows match the contact-map residue order returned together.
    """
    mask = (df["chain"].astype(str) == str(chain))
    idx = np.nonzero(mask.values)[0]
    if len(idx) == 0:
        raise ValueError(f"Chain '{chain}' not found in DSSP/feature table.")
    return full_map[np.ix_(idx, idx)]  # L×L for this chain


def extract_features_from_pdb(pdb_id_or_path: str,
                              chain: str,
                              feature_path: str = './',
                              cutoff: float = 7.3, # A
                              dssp_exec: str = r"C:\Program Files (x86)\mkdssp\bin\mkdssp.exe",
                              device: str = "cpu",
                              save_enm: bool = True,
                              model: EsmPreTrainedModel = None,
                              tokenizer: PreTrainedTokenizer = None,
                              k0: float = 1.0,
                              covalent_ratio: float = 3.0,
                              k_eigs: int = 32):
    """
    End-to-end: mmCIF -> residue table (csv), per-chain distance/contact map (npy),
    ENM/Green’s function products (L, Rn, dCn), and optional embedding placeholder.

    Returns:
        df_chain:        (L, F) residue-level features for the chain
        distance_map:    (L, L) Cα distance matrix
        Lmat:            (1, L, L)
        Rn:              (1, L, L)
        dCn:             (1, L, L)
        embedding:       placeholder (None)
    """
    # 0) file layout
    cif_path = _ensure_mmCIF(pdb_id_or_path, feature_path)
    pdb_id   = Path(cif_path).stem[:4].lower()
    out_dir  = Path(feature_path) / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    residue_csv      = out_dir / "residue_features.csv"
    chain_mech_map_npy = out_dir / f"res_mech_map_{chain}.npy"
    embedding_npy = out_dir / f"embedding_{chain}.npy"

    # 1) parse structure + DSSP once
    if not residue_csv.exists() or not chain_mech_map_npy.exists():
        structure = load_structure(cif_path)
        dssp = run_dssp(structure, cif_path, dssp_exec=dssp_exec)

    # 2) residue-level features + full-structure contact map
    if not residue_csv.exists():
        df = extract_residue_level_features(structure, dssp)
        df.to_csv(residue_csv, index=False)
        df_all = df
    else:
        df_all = pd.read_csv(residue_csv)

    # 3) Get the df of the chain (in the same order as the submatrices)
    df_chain = df_all[df_all["chain"].astype(str) == str(chain)].reset_index(drop=True)

    # 4) Get the distance map of the chain
    if not chain_mech_map_npy.exists():
        distance_maps, _ = per_chain_relative_distance_matrices(structure, return_residue_index=False)
        map_chain = None

        # Save distance maps and Lmat, Rn, dCn
        enm = ENMGreenFunction(k0=k0, c=covalent_ratio, k_eigs=k_eigs).to(device)
        for c in distance_maps.keys():
            dis = torch.from_numpy(distance_maps[c]).to(device)
            dis = dis.unsqueeze(0)  # (1, L, L)
            G, Rn, dCn = enm(dis)  # (1, L, L) each

            map = torch.cat((dis, G, Rn, dCn), dim=0)
            np.save(out_dir / f"res_mech_map_{c}.npy", map.cpu().numpy())

            if c == chain:
                map_chain = map

    else:
        map_chain = np.load(chain_mech_map_npy)

    # 5) ESM Embedding
    if not embedding_npy.exists():
        if model is None or tokenizer is None:
            embedding = None  # placeholder; plug in your ESM/PLM embedding if available
        else:
            seq = get_chain_sequence(df_all, chain)
            embedding = embed_sequence(seq, model, tokenizer)
    else:
        embedding = np.load(embedding_npy)

    return {
        "protein_embed": embedding,
        "res_feat": df_chain,
        "res_map": map_chain,
    }


def get_sequence_from_pdb_id(pdb_id):
    """
    Protein sequences were obtained from the RCSB PDB using PDB IDs.
    """
    try:
        # Standard URL format for PDB FASTA files
        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"

        # Retrieving data using urllib.request
        with urllib.request.urlopen(fasta_url) as response:
            # Parsing FASTA files with SeqIO
            text_stream = io.TextIOWrapper(response, encoding='utf-8')
            for record in SeqIO.parse(text_stream, "fasta"):
                return str(record.seq)

    except Exception as e:
        print(f"Error fetching sequence for {pdb_id}: {e}")
        return None
