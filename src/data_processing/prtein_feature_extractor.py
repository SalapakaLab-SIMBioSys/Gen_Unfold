# file: extract_pdb_residue_features.py
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Any

import Bio
import torch
from Bio.PDB import PDBParser, DSSP, NeighborSearch, MMCIFParser, Atom
from Bio.PDB.Polypeptide import is_aa, Polypeptide, one_to_index, index_to_three, index_to_one, three_to_index
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import seq1
from Bio import SeqIO
from Bio import AlignIO
from Bio.Align import AlignInfo
from Bio.Align.Applications import ClustalwCommandline
import numpy as np
import pandas as pd
import math
import os
import tempfile
import networkx as nx
from matplotlib import pyplot as plt
from torch import nn, Tensor
from tqdm import tqdm

# Aromatic and charged residues
AROMATICS = {'PHE', 'TYR', 'TRP', 'HIS'}
POSITIVE = {'ARG', 'LYS', 'HIS'}
NEGATIVE = {'ASP', 'GLU'}
HYDROPHOBIC = {'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'}
ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',
    'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY',
    'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
    'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}
SECONDARY_STRUCTURE = {
    'H': 0,  # Alpha helix (4-12)
    'B': 1,  # Isolated beta-bridge residue
    'E': 2,  # Strand
    'G': 0,  # 3-10 helix
    'I': 0,  # Pi helix
    'T': 5,  # Turn
    'S': 6,  # Bend
    '-': 7,  # loop or irregular
}

# AAindex placeholder for physicochemical properties
AAINDEX = {
    'ALA': {'hydro': 1.8, 'volume': 88.6, 'charge': 0},
    'ARG': {'hydro': -4.5, 'volume': 173.4, 'charge': 1},
    'ASN': {'hydro': -3.5, 'volume': 114.1, 'charge': 0},
    'ASP': {'hydro': -3.5, 'volume': 111.1, 'charge': -1},
    'CYS': {'hydro': 2.5, 'volume': 108.5, 'charge': 0},
    'GLU': {'hydro': -3.5, 'volume': 138.4, 'charge': -1},
    'GLN': {'hydro': -3.5, 'volume': 143.8, 'charge': 0},
    'GLY': {'hydro': -0.4, 'volume': 60.1, 'charge': 0},
    'HIS': {'hydro': -3.2, 'volume': 153.2, 'charge': 1},
    'ILE': {'hydro': 4.5, 'volume': 166.7, 'charge': 0},
    'LEU': {'hydro': 3.8, 'volume': 166.7, 'charge': 0},
    'LYS': {'hydro': -3.9, 'volume': 168.6, 'charge': 1},
    'MET': {'hydro': 1.9, 'volume': 162.9, 'charge': 0},
    'PHE': {'hydro': 2.8, 'volume': 189.9, 'charge': 0},
    'PRO': {'hydro': -1.6, 'volume': 112.7, 'charge': 0},
    'SER': {'hydro': -0.8, 'volume': 89.0, 'charge': 0},
    'THR': {'hydro': -0.7, 'volume': 116.1, 'charge': 0},
    'TRP': {'hydro': -0.9, 'volume': 227.8, 'charge': 0},
    'TYR': {'hydro': -1.3, 'volume': 193.6, 'charge': 0},
    'VAL': {'hydro': 4.2, 'volume': 140.0, 'charge': 0}
}


def load_structure(pdb_path):
    parser = (MMCIFParser if 'cif' in pdb_path else PDBParser)(QUIET=True)
    pdb_path = Path(pdb_path).expanduser().resolve()
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    return structure


def run_dssp(structure,
             pdb_path: str | Path,
             *,
             dssp_exec: str | Path = "mkdssp",
             acc_mode: str = "Sander"):
    """
    Parse a PDB/mmCIF file and compute DSSP.

    Parameters
    ----------
    pdb_path   : file path (accepts .gz)
    dssp_exec  : mkdssp / dssp binary (name in PATH or full path)
    acc_mode   : "Sander" (legacy) or "Tien";

    Returns
    -------
    structure  : Bio.PDB.Structure.Structure
    dssp       : Bio.PDB.DSSP.DSSP (dict-like)
    """

    model = structure[0]

    dssp = DSSP(model, str(pdb_path),
                dssp=str(dssp_exec),
                acc_array=acc_mode)

    return dssp


def cif_chain_sequences(cif_path: str | Path,
                        *,
                        write_fasta: str | Path | None = None) -> dict[str, str]:
    """
    Extract one-letter amino-acid sequences for every chain in an mmCIF file.

    Parameters
    ----------
    cif_path     : path to .cif / .cif.gz file
    write_fasta  : optional FASTA output path

    Returns
    -------
    dict {chain_id: sequence}
    """
    cif_path = Path(cif_path).expanduser().resolve()
    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, cif_path)

    seqs: dict[str, list[str]] = {}
    model = structure[0]  # mmCIF almost always single model

    for chain in model:
        chain_id = chain.id
        aa_list: list[str] = []
        for res in chain:
            if is_aa(res, standard=True):
                resname = res.resname.strip()  # e.g. "LYS"
                try:
                    aa_list.append(seq1(resname))
                except KeyError:
                    aa_list.append("X")  # non-canonical
        if aa_list:
            seqs[chain_id] = "".join(aa_list)

    if write_fasta:
        recs = [SeqRecord(Seq(s), id=cid, description="") for cid, s in seqs.items()]
        SeqIO.write(recs, Path(write_fasta), "fasta")


    return seqs


def get_chain_sequence(df: pd.DataFrame, chain_id: str) -> str:
    """
    Extract one-letter AA sequence for a given chain from residue features DataFrame.

    Args:
        df (pd.DataFrame): residue_features.csv loaded into DataFrame.
        chain_id (str): target chain ID (e.g., 'A').

    Returns:
        str: one-letter amino acid sequence.
    """
    chain_df = df[df["chain"].astype(str) == chain_id].sort_values("res_idx")

    seq = "".join([index_to_one(three_to_index(res)) if len(res) == 3 else res
                   for res in chain_df["res_name"]])
    return seq


def get_residue_bfactor(structure, chain_id, res_id):
    try:
        for atom in structure[0][chain_id][(' ', res_id, ' ')]:
            return atom.get_bfactor()
    except Exception as e:
        return np.nan


def detect_salt_bridges(structure):
    model = structure[0]
    ns = NeighborSearch(list(model.get_atoms()))
    salt_bridges = set()
    for chain in model:
        for residue in chain:
            if not is_aa(residue):
                continue
            resname = residue.get_resname()
            if resname not in POSITIVE | NEGATIVE:
                continue
            for atom in residue:
                close_atoms = ns.search(atom.coord, 4.0, level='A')
                for neighbor in close_atoms:
                    res2 = neighbor.get_parent()
                    if res2 == residue or not is_aa(res2):
                        continue
                    res2name = res2.get_resname()
                    if resname in POSITIVE and res2name in NEGATIVE:
                        salt_bridges.add((residue.get_id()[1], res2.get_id()[1]))
                    elif resname in NEGATIVE and res2name in POSITIVE:
                        salt_bridges.add((residue.get_id()[1], res2.get_id()[1]))
    return salt_bridges


def detect_pi_stack(structure):
    model = structure[0]
    stack_pairs = set()
    ns = NeighborSearch(list(model.get_atoms()))
    for chain in model:
        for res in chain:
            if not is_aa(res) or res.get_resname() not in AROMATICS:
                continue
            for atom in res:
                neighbors = ns.search(atom.coord, 5.0, level='R')
                for nres in neighbors:
                    if nres == res or not is_aa(nres):
                        continue
                    if nres.get_resname() in AROMATICS:
                        stack_pairs.add((res.get_id()[1], nres.get_id()[1]))
    return stack_pairs


def calculate_entropy(msa_path):
    alignment = AlignIO.read(msa_path, 'clustal')
    summary_align = AlignInfo.SummaryInfo(alignment)
    entropy_list = summary_align.information_content(per_residue=True)
    return entropy_list


def extract_residue_features(structure, dssp, salt_bridges, pi_stack):
    features = []
    for key in dssp.keys():
        chain_id, res_id = key[0], dssp[key][0] - 1
        res_name = dssp[key][1]
        sec_type = dssp[key][2]
        sasa = dssp[key][3]
        phi, psi = dssp[key][4], dssp[key][5]
        hbond_density = sum([dssp[key][i] for i in range(7, 14, 2)])
        bfactor = get_residue_bfactor(structure, chain_id, res_id)
        if res_name in ONE_TO_THREE:
            res_name = ONE_TO_THREE[res_name]
        else:
            print(res_name)
            continue
        hydro = AAINDEX.get(res_name, {}).get('hydro', 0)
        volume = AAINDEX.get(res_name, {}).get('volume', 0)
        charge = AAINDEX.get(res_name, {}).get('charge', 0)

        features.append({
            'chain': chain_id,
            'res_idx': res_id,
            'res_name': res_name,
            'sec_type': sec_type,
            'sasa': sasa,
            'phi': phi,
            'psi': psi,
            'b_factor': bfactor,
            'hbond_density': hbond_density,
            'hydropathy': hydro,
            'volume': volume,
            'charge': charge,
            'hydrophobic': int(res_name in HYDROPHOBIC),
            'aromatic': int(res_name in AROMATICS),
            'positive': int(res_name in POSITIVE),
            'negative': int(res_name in NEGATIVE),
            'pro_or_gly': int(res_name in {'PRO', 'GLY'}),
            'salt_bridge': int(any(res_id in pair for pair in salt_bridges)),
            'pi_stack': int(any(res_id in pair for pair in pi_stack)),
        })
    return pd.DataFrame(features)

def residue_contact_map(structure_or_path,
                        cutoff: float = 8.0,
                        atom_level: str = "CA"):
    """
    Compute binary residue–residue contact map.

    Parameters
    ----------
    structure_or_path : Bio.PDB.Structure.Structure | str | Path
    cutoff            : Ångström distance threshold
    atom_level        : "CA" | "CB" | "heavy"

    Returns
    -------
    contact_map       : np.ndarray (N, N) dtype=uint8
    residue_index     : list[(chain_id, res_num)]
    """

    def representative_atoms(residue, mode: str):
        """Return list of Atom objects according to selection mode."""
        if mode == "heavy":
            return [a for a in residue if a.element != "H"]
        if mode == "CB":
            if "CB" in residue:
                return [residue["CB"]]
            # glycine fallback
            if "CA" in residue:
                return [residue["CA"]]
        # default "CA"
        return [residue["CA"]] if "CA" in residue else residue[0]


    if isinstance(structure_or_path, (str, Path)):
        structure = load_structure(structure_or_path)
    else:
        structure = structure_or_path

    model = structure[0]
    residues, atoms = [], []

    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            rep_atoms = representative_atoms(res, atom_level)
            if not rep_atoms:
                warnings.warn(f"Residue {chain.id}:{res.id[1]} lacks representative atoms – skipped")
                continue
            residues.append((chain.id, res.id[1]))
            atoms.append(rep_atoms)

    N = len(residues)
    contact_map = np.zeros((N, N), dtype=np.uint8)

    # Flatten atoms with residue index bookkeeping
    flat_atoms, flat_owner = [], []
    for idx, atom_list in enumerate(atoms):
        flat_atoms.extend(atom_list)
        flat_owner.extend([idx] * len(atom_list))

    ns = NeighborSearch(flat_atoms)

    for idx, atom in enumerate(flat_atoms):
        close_atoms = ns.search(atom.coord, cutoff, level="A")
        for nbr in close_atoms:
            i, j = flat_owner[idx], flat_owner[flat_atoms.index(nbr)]
            if i != j:
                contact_map[i, j] = contact_map[j, i] = 1

    return contact_map


def per_chain_relative_distance_matrices(
    structure_or_path,
    *,
    atom_level: str = "CA",      # "CA" | "CB" | "heavy"
    as_similarity: bool = False, # If True, return 1 - normalized_distance
    return_residue_index: bool = True,
    normalize: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    """
    Compute per-chain relative distance matrices from a protein structure.

    Parameters
    ----------
    structure_or_path : Bio.PDB.Structure.Structure | str | Path
        Protein structure object or path to PDB/mmCIF file.
    atom_level : str
        Atom selection mode: "CA" (default), "CB", or "heavy".
    as_similarity : bool
        If True, convert relative distance to similarity (1 - rel_dist).
    return_residue_index : bool
        If True, also return residue indices for each chain.
    normalize: bool,
        If True, normalize relative distance to [0, 1].

    Returns
    -------
    relD_by_chain : dict
        Mapping {chain_id: (L, L) relative distance matrix}.
    res_idx_by_chain : dict
        Mapping {chain_id: [residue numbers]} if return_residue_index=True,
        otherwise an empty dict.
    """
    # Load structure if given as file path
    if isinstance(structure_or_path, (str, Path)):
        structure = load_structure(structure_or_path)
    else:
        structure = structure_or_path

    def representative_atoms(residue, mode: str):
        """Return the list of representative atoms according to selection mode."""
        if mode == "heavy":
            return [a for a in residue if a.element != "H"]
        if mode == "CB":
            if "CB" in residue:
                return [residue["CB"]]
            if "CA" in residue:  # fallback for Glycine
                return [residue["CA"]]
            return []
        return [residue["CA"]] if "CA" in residue else []

    relD_by_chain: Dict[str, np.ndarray] = {}
    res_idx_by_chain: Dict[str, List[int]] = {}

    model = structure[0]
    for chain in model:
        chain_id = chain.id
        coords: List[np.ndarray] = []
        res_ids: List[int] = []

        # Collect one representative coordinate per residue
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            atoms = representative_atoms(res, atom_level)
            if not atoms:
                continue
            coords.append(atoms[0].coord.astype(np.float32))
            res_ids.append(res.id[1])  # PDB residue number

        L = len(coords)
        if L < 2:
            # Not enough residues to form a matrix
            relD_by_chain[chain_id] = np.zeros((L, L), dtype=np.float32)
            res_idx_by_chain[chain_id] = res_ids
            continue

        C = np.stack(coords, axis=0)  # (L, 3)

        # Compute pairwise Euclidean distances (full tensor form)
        diff = C[:, None, :] - C[None, :, :]  # (L, L, 3)
        D = np.linalg.norm(diff, axis=-1)     # (L, L)

        # Normalize distances within this chain
        if normalize:
            dmax = float(np.max(D))
            if dmax > 0:
                D_rel = D / dmax
            else:
                D_rel = np.zeros_like(D, dtype=np.float32)
            D_rel = D_rel.astype(np.float32)

            if as_similarity:
                D_rel = 1.0 - D_rel

        else:
            D_rel = D

        relD_by_chain[chain_id] = D_rel
        res_idx_by_chain[chain_id] = res_ids

    if return_residue_index:
        return relD_by_chain, res_idx_by_chain
    else:
        return relD_by_chain, {}


def build_hbond_energy_by_chain(dssp):
    """
    Build hydrogen bond energy matrices for each chain from DSSP results.

    Parameters
    ----------
    dssp : Bio.PDB.DSSP.DSSP
        DSSP output object (dict-like), indexed by (chain_id, residue_id).

    Returns
    -------
    energy_by_chain : dict
        Mapping {chain_id: (energy_matrix, residue_index_list)}
        - energy_matrix: (L, L) np.ndarray, hydrogen bond energies for that chain
        - residue_index_list: list[(chain_id, residue_number)] in DSSP order for that chain
    """
    # Prepare per-chain storage
    chain_keys = {}
    for key in dssp.keys():
        chain_id = key[0]
        chain_keys.setdefault(chain_id, []).append(key)

    energy_by_chain = {}
    res_idx_by_chain = {}

    for chain_id, keys in chain_keys.items():
        N = len(keys)
        energy_matrix = np.zeros((N, N), dtype=np.float32)

        for i, key in enumerate(keys):
            hbonds = [
                (6, 7),   # NH->O_1
                (8, 9),   # O->NH_1
                (10, 11), # NH->O_2
                (12, 13)  # O->NH_2
            ]
            for rel_idx_field, energy_field in hbonds:
                rel_idx = dssp[key][rel_idx_field]
                energy = dssp[key][energy_field]
                if rel_idx != 0:  # 0 means no bond partner
                    j = i + rel_idx
                    if 0 <= j < N:
                        # Keep most stabilizing (lowest) energy if multiple exist
                        if energy_matrix[i, j] == 0 or energy < energy_matrix[i, j]:
                            energy_matrix[i, j] = energy
                            energy_matrix[j, i] = energy  # symmetric

        res_index = [(k[0], k[1][1]) for k in keys]
        energy_by_chain[chain_id] = energy_matrix
        res_idx_by_chain[chain_id] = res_index

    return energy_by_chain, res_idx_by_chain



class ENMGreenFunction(nn.Module):
    """
    Supports two inputs:
      - binary contact map C_{ij} in {0,1}
      - distance map r_{ij} in Å (pfGNM weighting: w_ij = k0 * r_ij^{-gamma})
    """
    def __init__(
        self,
        k0: float = 1.0,          # fixed spring constant
        c: float = 3.0,           # sequence-neighbor spring multiplier
        k_eigs: int = 32,
        beta_r: float = 1.0,
        beta_c: float = 0.5,
        eps: float = 1e-6,
        # --- new for pfGNM ---
        input_is_distance: bool = True,  # True if you pass a distance map
        gamma: float = 2.0,               # pfGNM: r^{-gamma}
        r_cut: float | None = None,       # optional max cutoff for sparsity (Å)
        min_dist: float = 1e-3            # avoid division by zero
    ):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(c)))
        self.register_buffer("k0", torch.tensor(float(k0)))
        self.k_eigs = int(k_eigs)
        self.beta_r = nn.Parameter(torch.tensor(beta_r, dtype=torch.float32))
        self.beta_c = nn.Parameter(torch.tensor(beta_c, dtype=torch.float32))
        self.eps = float(eps)

        # new flags/params
        self.input_is_distance = bool(input_is_distance)
        self.gamma = float(gamma)
        self.r_cut = None if r_cut is None else float(r_cut)
        self.min_dist = float(min_dist)

    @torch.no_grad()
    def forward(self, graph: torch.Tensor, anchors: torch.Tensor | None = None) -> torch.Tensor:
        # Accept (N,1,L,L)
        if graph.dim() == 4:
            graph = graph[:, 0]  # (N,L,L)

        N, L, _ = graph.shape
        device = graph.device
        dtype = graph.dtype
        eye = torch.eye(L, device=device, dtype=dtype).bool().unsqueeze(0)

        # sequence neighbors (i,i±1) mask
        seq_adj_mask = torch.eye(L, device=device).bool().unsqueeze(0).repeat(N,1,1)
        seq_adj_mask = seq_adj_mask.roll(1,1) | seq_adj_mask.roll(-1,1)
        seq_adj_mask[:, -1, 0] = False
        seq_adj_mask[:,  0,-1] = False

        if not self.input_is_distance:
            # --- original ENM on binary contact map ---
            C = graph.float()
            C = torch.where(eye, torch.zeros_like(C), C)        # zero diagonal
            W = self.k0.to(dtype=dtype) * C
            # strengthen covalent neighbors multiplicatively under pf-style
            W[seq_adj_mask] = W[seq_adj_mask] * self.c
        else:
            # --- pfGNM on distance map ---
            R = graph.float()
            # optional sparsification
            valid = (R > 0)
            if self.r_cut is not None:
                valid = valid & (R <= self.r_cut)
            # safe inverse power
            Rinvg = torch.where(valid, torch.clamp(R, min=self.min_dist).pow(-self.gamma), torch.zeros_like(R))
            Rinvg = torch.where(eye, torch.zeros_like(Rinvg), Rinvg)  # zero diagonal
            W = self.k0.to(dtype=dtype) * Rinvg
            # strengthen covalent neighbors
            W[seq_adj_mask] = W[seq_adj_mask] * self.c

        # symmetrize
        W = 0.5 * (W + W.transpose(1,2))

        # Laplacian (a.k.a. Kirchhoff up to sign): L = D - W
        D = torch.diag_embed(W.sum(-1))
        Lmat = D - W + self.eps * torch.eye(L, device=device, dtype=dtype).unsqueeze(0)

        # eigen-decomp + Green’s function, compliance & effective resistance
        evals, evecs = torch.linalg.eigh(Lmat)                                                # (N,L), (N,L,L)
        k = min(self.k_eigs, max(L - 1, 1))
        lam = evals[:, 1:1+k]                                                                  # skip the first
        U   = evecs[:, :, 1:1+k]
        G   = torch.bmm(U * (1.0/lam).unsqueeze(1), U.transpose(1,2))                         # (N,L,L)

        diagG = torch.diagonal(G, dim1=1, dim2=2)                                             # (N,L)
        R_eff = diagG[:, :, None] + diagG[:, None, :] - 2.0 * G                                # (N,L,L)

        if anchors is None:
            s_idx = torch.zeros(N, dtype=torch.long, device=device)
            t_idx = torch.full((N,), L-1, dtype=torch.long, device=device)
        else:
            assert anchors.shape == (N,2)
            s_idx, t_idx = anchors[:,0].long(), anchors[:,1].long()

        b = torch.zeros(N, L, device=device, dtype=dtype)
        b.scatter_(1, t_idx[:,None],  1.0)
        b.scatter_(1, s_idx[:,None], -1.0)
        cvec = torch.bmm(G, b.unsqueeze(-1)).squeeze(-1)                                       # (N,L)
        dC   = (cvec[:, :, None] - cvec[:, None, :]).abs()                                     # (N,L,L)

        return G, R_eff, dC

    @staticmethod
    def _zscore_offdiag(X: torch.Tensor) -> torch.Tensor:
        """
        Per-sample z-score that EXCLUDES the diagonal and sets it to zero.
        X: (N, L, L)
        Returns: Z with same shape; diagonal is exactly 0.
        """
        N, L, _ = X.shape
        device = X.device
        dtype = X.dtype

        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)  # (N, L, L) bool
        off = ~eye

        # masked mean over off-diagonal entries
        count = off.sum(dim=(1, 2), keepdim=True).to(X.dtype)  # (N,1,1)
        mean = (X.masked_fill(~off, 0.0).sum(dim=(1, 2), keepdim=True)) / count  # (N,1,1)

        # masked std over off-diagonal entries
        var = ((X - mean).masked_fill(~off, 0.0).pow(2).sum(dim=(1, 2), keepdim=True)) / count
        std = var.sqrt().clamp_min(1e-6)

        Z = (X - mean) / std
        Z = Z.masked_fill(eye, 0.0)  # zero the diagonal explicitly
        return Z

    @staticmethod
    def _minmax_offdiag(X):
        N, L, _ = X.shape
        device = X.device
        dtype = X.dtype

        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)  # (N, L, L) bool
        off = ~eye
        # Mask out the diagonal
        X_off = X.masked_fill(~off, 0.0)
        # Compute per-sample min and max over off-diagonal entries
        X_min = X_off.masked_fill(~off, float('inf')).amin(dim=(1, 2), keepdim=True)
        X_max = X_off.masked_fill(~off, float('-inf')).amax(dim=(1, 2), keepdim=True)
        # Avoid division by zero
        denom = (X_max - X_min).clamp_min(1e-6)
        # Normalize to [0,1] and set diagonal to zero
        Xn = (X - X_min) / denom
        Xn = Xn.masked_fill(eye, 0.0)
        return Xn



Segment = namedtuple(
    "Segment",
    ["chain", "idx", "sec_code", "start_idx", "end_idx"]  # indices are DSSP order (0-based)
)


def get_sec_seq(dssp) -> List[list[str]]:
    """
    Return DSSP secondary-structure code sequence in PDB order.
    H/G/I → H, E/B → B, the rest keep original for granularity.
    """
    mapping = {"H": "H", "G": "H", "I": "H",
               "E": "B", "B": "B"}
    sec_seq = []
    for key in dssp.keys():
        sec_seq.append([mapping.get(dssp[key][2], dssp[key][2]), key[0]])
    return sec_seq


def segment_sec_seq(sec_seq: List[str]) -> List[Segment]:
    """
    Collapse consecutive identical codes into segments.
    """
    segments: List[Segment] = []
    if not sec_seq:
        return segments

    seg_start = 0
    current = sec_seq[0][0]
    idx = 0
    chain = sec_seq[0][1]
    for i, array in enumerate(sec_seq[1:], start=1):
        code, chain = array[0], array[1]
        if code != current:
            segments.append(Segment(chain, idx, current, seg_start, i - 1))
            idx += 1
            seg_start, current = i, code
    # last segment
    segments.append(Segment(chain, idx, current, seg_start, len(sec_seq) - 1))
    return segments


def aggregate_segment_features(
    segments: List[Segment],
    residue_df: pd.DataFrame,
    numeric_cols: Tuple[str, ...] = (
        "sasa", "phi", "psi", "b_factor", "hbond_density",
        "hydropathy", "volume", "charge")
) -> pd.DataFrame:
    """
    For each segment compute mean (numeric) & mode (categorical) features.
    """
    seg_records = []
    for seg in segments:
        seg_slice = residue_df.iloc[seg.start_idx: seg.end_idx + 1]
        record = {
            "chain": seg.chain,
            "seg_idx": seg.idx,
            "sec_code": seg.sec_code,
            "length": seg.end_idx - seg.start_idx + 1
        }
        # numeric aggregations
        record.update({c: seg_slice[c].mean(skipna=True) for c in numeric_cols if c in seg_slice})
        # binary flags – treat as proportion of residues with flag
        for flag in ["hydrophobic", "aromatic", "positive", "negative",
                     "pro_or_gly", "salt_bridge", "pi_stack"]:
            if flag in seg_slice:
                record[flag + "_ratio"] = seg_slice[flag].mean()
        seg_records.append(record)
    return pd.DataFrame(seg_records)


def _collect_residue_objects(structure, dssp) -> list[Bio.PDB.Residue.Residue]:
    """
    Return residues in the exact DSSP order to keep indices consistent.
    """
    residue_objs = []
    for key in dssp.keys():
        chain_id, res_id = key[0], key[1][1]
        if dssp[key][1] in ONE_TO_THREE:
            residue_objs.append(structure[0][chain_id][(' ', res_id, ' ')])
    return residue_objs


def build_segment_contact_adjacency(
    structure,
    dssp,
    segments: list[Segment],
    cutoff: float = 8.0
) -> np.ndarray:
    """
    Compute an N × N adjacency matrix where entry (i, j)=1
    if any residue pair across segments i and j has two atoms
    closer than `cutoff` Å.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
    dssp      : DSSP object (for residue ordering)
    segments  : list of Segment namedtuples (start/end indices in DSSP order)
    cutoff    : distance threshold in Å (default 5.0)

    Returns
    -------
    adj : np.ndarray of shape (N, N)
    """
    n = len(segments)
    adj = np.zeros((n, n), dtype=int)

    # (1) gather residues in DSSP order
    residue_objs = _collect_residue_objects(structure, dssp)

    # (2) pre-gather atom lists per segment
    seg_atoms: list[list[Atom.Atom]] = []
    for seg in segments:
        atoms = []
        for ridx in range(seg.start_idx, seg.end_idx + 1):
            atoms.extend(residue_objs[ridx].get_atoms())
        seg_atoms.append(atoms)

    # (3) build adjacency using Bio.PDB.NeighborSearch
    for i in range(n - 1):
        ns_i = NeighborSearch(seg_atoms[i])         # KD-tree over segment-i atoms
        for j in range(i + 1, n):
            for atom in seg_atoms[j]:
                if ns_i.search(atom.coord, cutoff, level='R'):
                    adj[i, j] = adj[j, i] = 1
                    break            # once contact is found → next pair

    return adj


# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────
def _principal_axis(coords: np.ndarray) -> np.ndarray:
    """Return first principal component unit vector (Cα axis)."""
    coords_centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
    axis = vh[0]
    return axis / np.linalg.norm(axis)


def _segment_axis(structure, seg: Segment) -> np.ndarray | None:
    """
    Extract Cα coordinates belonging to `seg` and compute axis vector.
    Returns None if fewer than 3 Cα atoms found.
    """
    ca_coords = []
    # DSSP preserves structure model order; map DSSP index→(chain,resid):
    dssp_keys = list(structure[0].get_residues())  # quick but order may differ
    # safer: iterate structure residues alongside DSSP enumeration
    idx = -1
    for chain in structure[0]:
        for res in chain:
            if not is_aa(res) or 'CA' not in res:
                continue
            idx += 1
            if seg.start_idx <= idx <= seg.end_idx:
                ca_coords.append(res['CA'].coord)
    if len(ca_coords) < 3:
        return None
    return _principal_axis(np.array(ca_coords))


def compute_segment_geometry(
    structure,
    segments: List[Segment]
) -> pd.DataFrame:
    """
    Calculate geometry between consecutive segments:
    * helix–helix or strand–strand axis angle (degrees)
    """
    geom_records = []
    axes_cache: Dict[int, np.ndarray | None] = {}
    for seg in segments:
        if seg.sec_code in {"H", "B"}:       # only helices / strands
            axes_cache[seg.idx] = _segment_axis(structure, seg)

    for a, b in zip(segments[:-1], segments[1:]):
        if (a.idx in axes_cache) and (b.idx in axes_cache):
            v1, v2 = axes_cache[a.idx], axes_cache[b.idx]
            if v1 is not None and v2 is not None:
                cos_ang = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle_deg = math.degrees(math.acos(cos_ang))
                geom_records.append({
                    "seg_i": a.idx,
                    "seg_j": b.idx,
                    "pair_type": f"{a.sec_code}-{b.sec_code}",
                    "axis_angle_deg": angle_deg
                })
    return pd.DataFrame(geom_records)

# ─────────────────────────────────────────────────────────────
# Domain level
# ─────────────────────────────────────────────────────────────

def load_domain_table(csv_path: str | Path) -> pd.DataFrame:
    """
    Expect columns: domain_id, chain, start, end
    start/end are integer residue numbers (inclusive).
    """
    df = pd.read_csv(csv_path)
    return df.astype({"start": int, "end": int})


# ───────────────────────── aggregation ──────────────────────────
def aggregate_domain_features(res_df: pd.DataFrame,
                              domain_df: pd.DataFrame,
                              numeric_cols: Tuple[str, ...] = (
                                  "sasa", "phi", "psi", "b_factor",
                                  "hbond_density", "hydropathy",
                                  "volume", "charge")) -> pd.DataFrame:
    records = []
    for dom_id, grp in domain_df.groupby("domain_id"):
        feats = res_df[(res_df["chain"].isin(grp["chain"]))
                       & (res_df["res_id"].between(grp["start"].min(), grp["end"].max()))]
        rec = {
            "domain_id": dom_id,
            "chain": grp["chain"].iloc[0],
            "start": grp["start"].min(),
            "end": grp["end"].max(),
            "length": len(feats)
        }
        rec.update({c: feats[c].mean() for c in numeric_cols if c in feats})
        # binary flag ratio
        for flag in ["hydrophobic", "aromatic", "positive", "negative",
                     "pro_or_gly", "salt_bridge", "pi_stack"]:
            if flag in feats:
                rec[flag + "_ratio"] = feats[flag].mean()
        records.append(rec)
    return pd.DataFrame(records)


# ───────────────────────── contact graph ─────────────────────────
def build_domain_contact_graph(structure,
                               domain_df: pd.DataFrame,
                               cutoff: float = 5.0) -> Tuple[np.ndarray, List[str]]:
    """
    Returns adjacency matrix (N×N) and list of domain_id in same order.
    """
    model = structure[0]
    dom_ids = domain_df["domain_id"].tolist()
    atoms_by_domain: Dict[str, List] = {d: [] for d in dom_ids}

    # collect atoms
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            rid = res.id[1]
            chain_id = chain.id
            # find matching domain(s)
            dom_match = domain_df[(domain_df.chain == chain_id)
                                  & (domain_df.start <= rid)
                                  & (domain_df.end >= rid)]
            for dom_id in dom_match["domain_id"]:
                atoms_by_domain[dom_id].extend([a for a in res if a.element != "H"])

    # build adjacency
    N = len(dom_ids)
    adj = np.zeros((N, N), np.uint8)
    for i, di in enumerate(dom_ids):
        ns_i = NeighborSearch(atoms_by_domain[di])
        for j in range(i + 1, N):
            dj = dom_ids[j]
            for atom in atoms_by_domain[dj]:
                if ns_i.search(atom.coord, cutoff, level="A"):
                    adj[i, j] = adj[j, i] = 1
                    break
    return adj, dom_ids


# ───────────────────────── linker flexibility ────────────────────
def compute_linker_flexibility(res_df: pd.DataFrame,
                               domain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consecutive domains (sorted by start) on same chain.
    """
    records = []
    for chain_id, grp in domain_df.sort_values("start").groupby("chain"):
        grp = grp.sort_values("start")
        for a, b in zip(grp.iloc[:-1].itertuples(), grp.iloc[1:].itertuples()):
            linker_mask = (res_df["chain"] == chain_id) & \
                          (res_df["res_id"] > a.end) & \
                          (res_df["res_id"] < b.start)
            linkers = res_df[linker_mask]
            if not linkers.empty:
                records.append({
                    "domain_i": a.domain_id,
                    "domain_j": b.domain_id,
                    "length": len(linkers),
                    "mean_B": linkers["b_factor"].mean()
                })
    return pd.DataFrame(records)



# ─────────────────────────────────────────────────────────────
# Public wrapper
# ─────────────────────────────────────────────────────────────
def extract_residue_level_features(structure, dssp):
    salt_bridges = detect_salt_bridges(structure)
    pi_stack = detect_pi_stack(structure)
    df = extract_residue_features(structure, dssp, salt_bridges, pi_stack)
    return df


def extract_secondary_structure_features(structure, dssp, residue_df, cutoff=4.0):
    """
    Main utility combining all secondary-structure level outputs.

    Returns
    -------
    sec_seq        : list[list[str]]      (one code and chain per residue)
    seg_df         : DataFrame      (aggregated segment features)
    adj_matrix     : np.ndarray     (N×N adjacency)
    geom_df        : DataFrame      (angles between neighbour segments)
    """
    sec_seq = get_sec_seq(dssp)
    segments = segment_sec_seq(sec_seq)
    seg_df = aggregate_segment_features(segments, residue_df)
    adj_matrix = build_segment_contact_adjacency(structure, dssp, segments, cutoff)
    geom_df = compute_segment_geometry(structure, segments)
    return sec_seq, seg_df, adj_matrix, geom_df


def extract_domain_level_features(pdb_or_cif: str | Path,
                                  residue_csv: str | Path,
                                  domain_csv: str | Path,
                                  cutoff: float = 5.0):
    structure = load_structure(pdb_or_cif)
    res_df = pd.read_csv(residue_csv)          # residue-level features table
    domain_df = load_domain_table(domain_csv)  # domain annotation

    dom_df = aggregate_domain_features(res_df, domain_df)
    adj, dom_order = build_domain_contact_graph(structure, domain_df, cutoff)
    linker_df = compute_linker_flexibility(res_df, domain_df)

    return dom_df, adj, linker_df


def extract_features_from_pdb(pdb_path, output_path: str = None, cutoff=4.0,
                              dssp_exec=r"C:\Program Files (x86)\mkdssp\bin\mkdssp.exe"):
    if output_path and os.path.exists(output_path):
        return None
    print(f'Extracting structure features from {pdb_path}')
    structure = load_structure(pdb_path)
    dssp = run_dssp(structure, pdb_path, dssp_exec=dssp_exec)

    df = extract_residue_level_features(structure, dssp)
    res_contact_map = residue_contact_map(structure, dssp)
    sec_seq, seg_df, adj_matrix, geom_df = extract_secondary_structure_features(structure, dssp, df,
                                                                            cutoff=cutoff)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        residue_feature_path = os.path.join(output_path, f"residue_features.csv")
        segment_feature_path = os.path.join(output_path, f"segment_features.csv")
        res_map_path = os.path.join(output_path, f"res_map")
        adj_matrix_path = os.path.join(output_path, f"seg_map")

        df.to_csv(residue_feature_path, index=False)
        seg_df.to_csv(segment_feature_path, index=False)
        print(seg_df.shape)
        np.save(adj_matrix_path, adj_matrix)
        np.save(res_map_path, res_contact_map)

    return df, seg_df, adj_matrix, geom_df


def process_all(pdb_path, func, output_path, max_threads=16, **kwargs):
    success_count = 0
    filenames = os.listdir(pdb_path)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(func,
                            os.path.join(pdb_path, file),
                            os.path.join(output_path, file[:4]),
                            **kwargs): file
            for file in filenames
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
            except Exception as exc:
                result = None
                print('error', exc)
            if result:
                success_count += 1

    print(f"[Done] {success_count}/{len(pdb_path)} saved.")









