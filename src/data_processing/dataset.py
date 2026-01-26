import concurrent.futures
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union, Iterable, Sequence

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
import logging

from tqdm import tqdm

from .utils import AMINO_ACID_DIC, SEC_CODE, create_cath_maps, extract_features_from_pdb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_cath_ids(
        cath_ids: List[str],
        c_map: Dict,
        a_map: Dict,
        t_map: Dict,
        h_map: Dict
) -> torch.Tensor:
    """
    Converts a list of CATH ID strings into a tensor of integer indices.

    Args:
        cath_ids: A list of CATH ID strings.
        c_map, a_map, t_map, h_map: The mapping dictionaries.

    Returns:
        A tensor of shape (batch_size, 4) with integer indices.
    """
    indexed_data = []
    for cath_id in cath_ids:
        parts = cath_id.split('.')
        if len(parts) == 4:
            indexed_data.append([
                c_map[parts[0]],
                a_map[parts[1]],
                t_map[parts[2]],
                h_map[parts[3]]
            ])
    return torch.tensor(indexed_data, dtype=torch.long)


class DataScaler:
    """
    A versatile data scaler for numpy arrays that supports different scaling
    methods and can handle data of various dimensions.

    The scaler computes statistics along the feature axis (the last axis).

    Attributes:
        method (str): The scaling method to use ('min-max' or 'std-mean').
        min_ (np.ndarray): The minimum values for each feature. Only used for 'min-max'.
        max_ (np.ndarray): The maximum values for each feature. Only used for 'min-max'.
        mean_ (np.ndarray): The mean for each feature. Only used for 'std-mean'.
        std_ (np.ndarray): The standard deviation for each feature. Only used for 'std-mean'.
    """

    def __init__(self, method: str = 'min-max'):
        """
        Initializes the DataScaler.

        Args:
            method (str): The scaling method to use.
                          Supported methods: 'min-max', 'std-mean'.
                          Defaults to 'min-max'.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        if method not in ['min-max', 'std-mean']:
            raise ValueError(f"Unsupported method '{method}'. Choose from 'min-max' or 'std-mean'.")
        self.method = method
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray):
        """
        Computes the necessary statistics (min/max or mean/std) from the data.

        The statistics are calculated along all axes except the last one (feature axis).

        Args:
            data (np.ndarray): The input data to fit the scaler on.
                               Shape can be (n_samples, n_features) or
                               (n_samples, n_timesteps, n_features), etc.
        """
        # Determine the axes to compute statistics over (all except the last one)
        axes = tuple(range(data.ndim - 1))

        if self.method == 'min-max':
            self.min_ = np.min(data, axis=axes)
            self.max_ = np.max(data, axis=axes)
        elif self.method == 'std-mean':
            self.mean_ = np.mean(data, axis=axes)
            self.std_ = np.std(data, axis=axes)

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Scales the data using the previously computed statistics.

        Args:
            data (np.ndarray): The data to transform.

        Returns:
            np.ndarray: The transformed (scaled) data.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if self.method == 'min-max':
            if self.min_ is None or self.max_ is None:
                raise RuntimeError("Scaler has not been fitted. Call fit() before transforming.")
            # Add a small epsilon to avoid division by zero for features with no variance
            denominator = self.max_ - self.min_
            epsilon = 1e-8
            return (data - self.min_) / (denominator + epsilon)

        elif self.method == 'std-mean':
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Scaler has not been fitted. Call fit() before transforming.")
            # Add a small epsilon to avoid division by zero for features with zero variance
            epsilon = 1e-8
            return (data - self.mean_) / (self.std_ + epsilon)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverts the scaled data back to its original representation.

        Args:
            data (np.ndarray): The scaled data to inverse transform.

        Returns:
            np.ndarray: The data in its original scale.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if self.method == 'min-max':
            if self.min_ is None or self.max_ is None:
                raise RuntimeError("Scaler has not been fitted. Call fit() before inverse transforming.")
            denominator = self.max_ - self.min_
            epsilon = 1e-8
            return data * (denominator + epsilon) + self.min_

        elif self.method == 'std-mean':
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Scaler has not been fitted. Call fit() before inverse transforming.")
            epsilon = 1e-8
            return data * (self.std_ + epsilon) + self.mean_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        A convenience method that fits the scaler to the data and then transforms it.

        Args:
            data (np.ndarray): The input data.

        Returns:
            np.ndarray: The transformed (scaled) data.
        """
        self.fit(data)
        return self.transform(data)

    def features(self):
        return pd.DataFrame({
            'min': self.min_,
            'max': self.max_,
            'mean': self.mean_,
            'std': self.std_,
            'method': self.method,
        })

    @classmethod
    def create_from_pd(cls, df: pd.DataFrame):
        obj = cls.__new__(cls)
        obj.min_ = np.array(df.min)
        obj.max_ = np.array(df.max)
        obj.mean_ = np.array(df.mean)
        obj.std_ = np.array(df.std)
        obj.method = df.method
        return obj


class FEDataset(Dataset):
    """
    A dataset class for handling Force-Extension (F-E) curves, protein sequences,
    and experimental conditions.

    This dataset treats F-E curves as the primary data source and can optionally
    load associated protein sequences and experimental conditions. It supports
    data normalization and lazy loading for sequences.
    """

    def __init__(self,
                 fe_curves_path: str,
                 sequences_path: Optional[str] = None,
                 data_table_path: Optional[str] = None,
                 conditions_path: Optional[str] = None,
                 feature_first: bool = True,
                 lazy: bool = False,
                 **kwargs
                 ):
        """
        Initializes the FEDataset.

        Args:
            fe_curves_path (str): Path to the F-E curve data file. This is a required argument.
            sequences_path (str, optional): Path to the protein sequence data file or directory (for lazy loading).
            data_table_path (str, optional): Path to a CSV file containing metadata (e.g., PDB_ID), required for lazy loading.
            conditions_path (str, optional): Path to the experimental conditions data file.
            feature_first (bool): If True, tensors will have the shape (batch_size, feature_dim, seq_len).
            lazy (bool): If True, sequence data will be lazily loaded from individual files on demand.
        """
        super().__init__()

        # Validate core paths
        if not fe_curves_path or not Path(fe_curves_path).exists():
            raise FileNotFoundError(f"Required F-E curve file not found: {fe_curves_path}")

        if lazy and (not data_table_path or not sequences_path):
            raise ValueError("In lazy loading mode, `data_table_path` and `sequences_path` must be provided.")

        self.feature_first = feature_first
        self.lazy = lazy
        self.fe_curves_path = Path(fe_curves_path)
        self.sequences_path = Path(sequences_path) if sequences_path else None

        # --- Data Loading ---
        # F-E curves are the primary data and are always loaded
        self.fe_curves, self.fe_scaler = self._load_and_process_data(
            file_path=fe_curves_path,
            data_type='F-E curve',
            normalize=True,
            to_tensor=True
        )
        self.num_samples = len(self.fe_curves)

        # Load other data sources on demand
        self.data_table = pd.read_csv(data_table_path) if data_table_path else None

        self.sequences, self.seq_scaler = (None, None) if self.lazy else self._load_and_process_data(
            file_path=sequences_path,
            data_type='sequences',
            normalize=False,  # Sequence data (e.g., one-hot) usually doesn't need normalization
            to_tensor=True
        )

        self.conditions, self.cond_scaler = self._load_and_process_data(
            file_path=conditions_path,
            data_type='conditions',
            normalize=True,
            to_tensor=True
        )

        # --- Data Consistency Check ---
        #self._validate_data_consistency()

    def _load_and_process_data(self, file_path: Optional[str], data_type: str, normalize: bool, to_tensor: bool) -> \
    Tuple[Optional[Any], Optional[DataScaler]]:
        """
        A unified helper function to load, normalize, and convert data to tensors.
        """
        if not file_path or not Path(file_path).exists():
            return None, None

        data = self._load_from_file(Path(file_path))
        if data is None:
            logging.warning(f"Could not load {data_type} data from {file_path}, unsupported file format.")
            return None, None

        scaler = None
        if normalize:
            scaler = DataScaler()
            data = scaler.fit_transform(data)

        if to_tensor:
            data = self._to_tensor(data)

        return data, scaler

    @staticmethod
    def _load_from_file(path: Path) -> Optional[np.ndarray]:
        """Load data based on file extension."""
        try:
            if path.suffix == '.pkl':
                # Assuming the pickle file contains a DataFrame
                return pd.read_pickle(path).values
            elif path.suffix == '.csv':
                return pd.read_csv(path).values
            elif path.suffix == '.npy':
                return np.load(path)
            else:
                return None
        except Exception as e:
            logging.error(f"Error loading file {path}: {e}")
            raise IOError(f"Failed to load file: {path}") from e

    def _to_tensor(self, data: np.ndarray) -> torch.FloatTensor:
        """Convert a NumPy array to a PyTorch tensor, handling dimension permutation."""
        tensor = torch.from_numpy(data).float()
        # Only permute if the tensor is 3-dimensional
        if self.feature_first and tensor.ndim == 3:
            tensor = tensor.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        return tensor

    def _validate_data_consistency(self):
        """
        Check if the number of samples is consistent across all loaded datasets.
        """
        datasets = {
            'sequences': self.sequences,
            'conditions': self.conditions,
            'data_table': self.data_table
        }
        for name, data in datasets.items():
            if data is not None and len(data) != self.num_samples:
                raise ValueError(
                    f"Data mismatch: F-E curves have {self.num_samples} samples, "
                    f"but {name} has {len(data)} samples."
                )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a data sample by index.
        """
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"Index {idx} is out of range for dataset size {self.num_samples}.")

        sample = {'fe_curve': self.fe_curves[idx]}

        # --- Handle sequence data ---
        if self.lazy:
            pdb_id = self.data_table['PDB_ID'].iloc[idx]
            sequence_path = self.sequences_path / f"{pdb_id}.npy"
            try:
                sequence_data = np.load(sequence_path)
                # Note: Lazily loaded data should also be transformed/normalized if needed.
                # Here, we assume lazy-loaded data is pre-processed and only needs tensor conversion.
                sample['sequence_data'] = self._to_tensor(sequence_data)
            except FileNotFoundError:
                logging.error(f"Lazy loading failed: sequence file not found at {sequence_path}")
                # Return an empty tensor or handle the error as needed
                sample['sequence_data'] = torch.empty(0)
        elif self.sequences is not None:
            sample['sequence_data'] = self.sequences[idx]

        # --- Handle condition data ---
        if self.conditions is not None:
            sample['conditions'] = self.conditions[idx]

        return sample

    @staticmethod
    def move_to_device(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Moves all tensors in a sample to a specified device (e.g., GPU).

        This method is static because it doesn't depend on any state of the dataset instance,
        making it a general-purpose utility.
        """
        return {key: tensor.to(device) for key, tensor in sample.items() if isinstance(tensor, torch.Tensor)}

    # --- Deprecated or Modified Methods ---
    # The get_condition method has been removed.
    # Its function (tensor concatenation) is a model-specific preprocessing step
    # and should not be part of the dataset's responsibilities.
    # This kind of operation is best handled in the training loop or a custom collate_fn.

    # --- Methods to be Implemented ---
    def save_pickle(self):
        """(Not Implemented) Serialize the dataset object to a file."""
        # TODO: Implement the saving logic for the dataset
        pass

    def load_pickle(self):
        """(Not Implemented) Deserialize the dataset object from a file."""
        # TODO: Implement the loading logic for the dataset
        pass


class FEDatasetStructure(FEDataset):
    """
    A dataset for protein structures supporting two distinct data handling pipelines:
    1. Eager Mode: Processes all data upon initialization and can save/load from a single pickle file.
    2. Hybrid Lazy Mode: Processes most features eagerly but lazy-loads `res_map` during `__getitem__`.
    """

    def __init__(self,
                 sequences_path: str,
                 data_table_path: str,
                 fe_curves_path: Optional[str] = None,
                 conditions_path: Optional[str] = None,
                 feature_first: bool = True,
                 pickle_path: Optional[str] = None,
                 lazy: bool = False,
                 max_len: Optional[int] = 250,
                 max_workers: Optional[int] = None,
                 lazy_res_map: bool = True,
                 lazy_esm_seq: bool = False,
                 **kwargs
                 ):
        """
        Initializes the dataset based on the chosen pipeline.

        Args:
            sequences_path (str): Base directory for structure data.
            data_table_path (str): Path to the main metadata CSV.
            fe_curves_path (str, optional): Path to F-E curve data.
            conditions_path (str, optional): Path to experimental conditions data.
            feature_first (bool): Determines tensor dimension order.
            pickle_path (str, optional): Path to save/load the pre-processed dataset.
                                         If file exists, it will be loaded, bypassing all processing.
                                         If file does not exist and `lazy_res_map` is False,
                                         the processed data will be saved here.
            lazy (bool): If True, activates the hybrid lazy loading mode for `res_map`.
                                 This is ignored if loading from an existing pickle file.
            max_len (int): Maximum length of sequences.
        """
        super().__init__(fe_curves_path=fe_curves_path,
                         sequences_path=sequences_path,
                         data_table_path=data_table_path,
                         conditions_path=conditions_path,
                         feature_first=feature_first,
                         lazy=True)
        self.feature_first = feature_first  # Needed by padding helpers
        self.max_len = max_len
        self.max_workers = max_workers if max_workers else os.cpu_count() * 2

        self.lazy_res_map = lazy_res_map
        self.lazy_esm_seq = lazy_esm_seq

        if pickle_path and Path(pickle_path).exists():
            logging.info(f"Loading pre-processed data from pickle file: {pickle_path}")
            self._load_from_pickle(pickle_path)
        else:
            self._initialize_data(pickle_path_to_save=pickle_path)


    def _process_single_item(self, index: int) -> Optional[Tuple]:
        """
        Worker function to process a single data sample.
        This function is executed in parallel by the thread pool.
        """
        try:
            pdb_id, chain = self.data_table.iloc[index][['PDB_ID', 'Chain']]
            chain = str(chain)

            file_path = self.sequences_path / pdb_id
            protein_embedding_path = file_path / f"embedding_{chain}.npy"
            residue_feature_path = file_path / "residue_features.csv"
            res_map_path = file_path / f"res_mech_map_{chain}.npy"

            if not all(p.exists() for p in [protein_embedding_path, residue_feature_path, res_map_path]):
                logging.warning(f"Skipping index {index} ({pdb_id}_{chain}): missing one or more files.")
                return None

            # Load data from disk
            protein_embedding = np.load(protein_embedding_path)
            residue_df = pd.read_csv(residue_feature_path)

            # Process features
            chain_df = residue_df[residue_df['chain'].astype(str) == chain]
            res_feat_arr = self._convert_to_array(chain_df)

            res_map_data: Union[np.ndarray, Path]
            if self.lazy_res_map:
                res_map_data = res_map_path  # Return path for lazy mode
            else:
                # Load res_map and filter for eager mode
                res_map_data = np.load(res_map_path)

            fe_curve = self.fe_curves[index] if self.fe_curves is not None else None

            return (protein_embedding, res_feat_arr, res_map_data, fe_curve, pdb_id, chain)

        except Exception as e:
            logging.error(f"Error processing index {index} ({pdb_id}_{chain}): {e}", exc_info=True)
            return None

    def _initialize_data(self, pickle_path_to_save: Optional[str]):
        """
        Initializes dataset by processing all items in parallel using a thread pool.
        """
        if self.lazy_res_map:
            logging.info("Initializing in Hybrid Lazy Mode with multi-threading.")
        else:
            logging.info("Initializing in Eager Mode with multi-threading.")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a map of futures
            future_to_index = {executor.submit(self._process_single_item, i): i for i in range(self.num_samples)}

            # Use tqdm to show progress as futures complete
            pbar = tqdm(concurrent.futures.as_completed(future_to_index), total=self.num_samples,
                        desc="Processing data")
            for future in pbar:
                results.append(future.result())

        # Filter out failed items and unzip the results
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            raise RuntimeError("No data could be processed. Check file paths and formats.")

        protein_embeddings, residue_features, res_maps_or_paths, fe_curves_filtered, \
            pdb_ids_filtered, chains_filtered = zip(*valid_results)

        # Post-process and store tensors
        self.protein_embeddings = torch.from_numpy(np.array(protein_embeddings)).float()
        self.residue_features, self.residue_mask, self.residue_scaler = self._pad_and_normalize_features(
            list(residue_features))
        self.fe_curves = torch.stack([c for c in fe_curves_filtered if c is not None])
        self.pdb_ids = list(pdb_ids_filtered)
        self.chains = list(chains_filtered)
        self.num_samples = len(self.pdb_ids)

        if self.lazy_res_map:
            self.res_map_paths = list(res_maps_or_paths)
            self.res_maps = None
        else:
            self.res_maps = self._pad_graphs(list(res_maps_or_paths))
            self.res_map_paths = None

        if pickle_path_to_save:
            self.save_pickle(pickle_path_to_save)


    def _init_eager(self, pickle_path_to_save: Optional[str]):
        """Pipeline 1: Processes all data and stores it in memory."""
        protein_embeddings, residue_features, res_maps, fe_curves_filtered = [], [], [], []
        pdb_ids_filtered, chains_filtered, lengths_filtered = [], [], []

        pdb_ids = self.data_table['PDB_ID'].tolist()
        chains = self.data_table['Chain'].tolist()
        lengths = self.data_table['N'].tolist()

        for i, (pdb_id, chain) in enumerate(tqdm(zip(pdb_ids, chains), total=len(pdb_ids), desc="Eager Processing")):
            try:
                file_path = self.sequences_path / pdb_id
                protein_embedding_path = file_path / f"embedding_{chain}.npy"
                residue_feature_path = file_path / "residue_features.csv"
                res_map_path = file_path / f"res_mech_map_{chain}.npy"

                if not all(p.exists() for p in [protein_embedding_path, residue_feature_path, res_map_path]):
                    logging.warning(f"Skipping {pdb_id}_{chain}: missing one or more files.")
                    continue

                # Load data
                protein_embedding = np.load(protein_embedding_path)
                res_map = np.load(res_map_path)
                residue_df = pd.read_csv(residue_feature_path)

                # Filter and process
                chain_df = residue_df[residue_df['chain'].astype(str) == str(chain)]

                res_feat_arr = self._convert_to_array(chain_df)

                # Append to lists
                protein_embeddings.append(protein_embedding)
                residue_features.append(res_feat_arr[:self.max_len])
                res_maps.append(res_map[:, :self.max_len, :self.max_len])
                fe_curves_filtered.append(self.fe_curves[i])
                pdb_ids_filtered.append(pdb_id)
                chains_filtered.append(chain)
                lengths_filtered.append(lengths[i])

            except Exception as e:
                logging.error(f"Error processing {pdb_id}_{chain}: {e}")

        # Post-process and store tensors
        self.protein_embeddings = torch.from_numpy(np.array(protein_embeddings)).float()
        self.residue_features, self.residue_mask, self.residue_scaler = self._pad_and_normalize_features(
            residue_features)
        self.res_maps = self._pad_graphs(res_maps)
        self.fe_curves = torch.stack(fe_curves_filtered)
        self.pdb_ids = pdb_ids_filtered
        self.chains = chains_filtered
        self.lengths = lengths_filtered
        self.num_samples = len(self.pdb_ids)

        if pickle_path_to_save:
            self.save_pickle(pickle_path_to_save)

    def _init_lazy_res_map(self):
        """Pipeline 2: Processes most data but keeps paths for `res_map`."""
        self.lazy_res_map = True
        protein_embeddings, residue_features, fe_curves_filtered = [], [], []
        pdb_ids_filtered, chains_filtered = [], []
        self.res_map_paths = []  # Store paths instead of data

        pdb_ids = self.data_table['PDB_ID'].tolist()
        chains = self.data_table['Chain'].tolist()

        for i, (pdb_id, chain) in enumerate(tqdm(zip(pdb_ids, chains), total=len(pdb_ids), desc="Hybrid Lazy Init")):
            # Similar loop, but `res_map` is handled differently
            try:
                file_path = self.sequences_path / pdb_id
                protein_embedding_path = file_path / f"embedding_{chain}.npy"
                residue_feature_path = file_path / "residue_features.csv"
                res_map_path = file_path / f"res_mech_map_{chain}.npy"

                # Only check for existence of res_map_path, don't load it
                if not all(p.exists() for p in [protein_embedding_path, residue_feature_path, res_map_path]):
                    extract_features_from_pdb(pdb_id_or_path=pdb_id,
                                              chain=chain,
                                              feature_path=self.sequences_path, )
                    if not all(p.exists() for p in [protein_embedding_path, residue_feature_path, res_map_path]):
                        logging.warning(f"Skipping {pdb_id}_{chain}: missing one or more files.")
                        continue

                protein_embedding = np.load(protein_embedding_path)
                residue_df = pd.read_csv(residue_feature_path)
                chain_df = residue_df[residue_df['chain'].astype(str) == str(chain)]
                res_feat_arr = self._convert_to_array(chain_df)

                protein_embeddings.append(protein_embedding)
                residue_features.append(res_feat_arr)
                self.res_map_paths.append(res_map_path)  # Store the path
                fe_curves_filtered.append(self.fe_curves[i])
                pdb_ids_filtered.append(pdb_id)
                chains_filtered.append(chain)

            except Exception as e:
                logging.error(f"Error processing {pdb_id}_{chain}: {e}")

        # Post-process eagerly loaded data
        self.protein_embeddings = torch.from_numpy(np.array(protein_embeddings)).float()
        self.residue_features, self.residue_mask, self.residue_scaler = self._pad_and_normalize_features(
            residue_features)
        self.fe_curves = torch.stack(fe_curves_filtered)
        self.pdb_ids = pdb_ids_filtered
        self.chains = chains_filtered
        self.num_samples = len(self.pdb_ids)
        self.res_maps = None  # Ensure this is None in lazy mode

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < self.num_samples:
            raise IndexError("Index out of range")

        if self.lazy_res_map:
            # Hybrid Lazy Mode: load res_map on the fly
            res_map_path = os.path.join(self.sequences_path, self.pdb_ids[idx], f"res_mech_map_{self.chains[idx]}.npy")
            try:
                res_map = np.load(res_map_path)
                map = np.zeros((res_map.shape[0], self.max_len, self.max_len), dtype=res_map.dtype)
                n = min(res_map.shape[-1], self.max_len)
                map[:, :n, :n] = res_map[:, :n, :n]
                res_map = torch.from_numpy(map).float()

            except Exception as e:
                raise f"Error processing {res_map_path}: {e}"

        else:
            # Eager Mode: all data is already in tensors
            if self.res_maps is not None:
                res_map = self.res_maps[idx]
            else:
                res_map = torch.zeros(1)

        if self.lazy_esm_seq:
            esm_seq_path = os.path.join(self.sequences_path, self.pdb_ids[idx],
                                   f"residue_embedding_{self.chains[idx]}.npy")
            try:
                esm_seq = np.load(esm_seq_path)
                seq = np.zeros((self.max_len, esm_seq.shape[1]), dtype=esm_seq.dtype)
                n = min(esm_seq.shape[0], self.max_len)
                seq[:n, :] = esm_seq[:n, :]
                esm_seq = torch.from_numpy(seq).float()

            except Exception as e:
                raise f"Error processing {esm_seq}: {e}"

        else:
            esm_seq = torch.zeros(1)

        sample = {
            "pdb_id": self.pdb_ids[idx],
            "fe_curve": self.fe_curves[idx],
            "protein_embed": self.protein_embeddings[idx],
            "residue_embed": esm_seq,
            "res_feat": self.residue_features[idx],
            "res_mask": self.residue_mask[idx],
            "res_map": res_map,
        }
        return sample

    # --- Data Processing and Persistence Helpers ---

    def _pad_and_normalize_features(self, arr_list: List[np.ndarray], pad_val: float = 0.0, scaler = None) -> Tuple[
        torch.Tensor, torch.Tensor, DataScaler]:
        """Pads features to max length in dataset, normalizes them, and returns scaler."""
        max_len = self.max_len
        feat_dim = arr_list[0].shape[1]

        padded = np.full((len(arr_list), max_len, feat_dim), pad_val, np.float32)
        mask = np.zeros((len(arr_list), max_len), bool)

        for i, arr in enumerate(arr_list):
            L = min(arr.shape[0], max_len)
            padded[i, :L, :] = arr[:L, :]
            mask[i, :L] = True

        if scaler is None:
            scaler = DataScaler()
            # Normalize along batch and sequence length dimensions
            padded_normalized = scaler.fit_transform(padded)
        else:
            padded_normalized = scaler.transform(padded)

        return torch.from_numpy(padded_normalized).float(), torch.from_numpy(mask), scaler

    def _pad_graphs(self, graph_list: List[np.ndarray]) -> torch.Tensor:
        """Pads a list of 2D graph arrays to the max dimension in the dataset."""
        feat_dim = graph_list[0].shape[0]
        out = np.zeros((len(graph_list), feat_dim, self.max_len, self.max_len), np.float32)
        for i, g in enumerate(graph_list):
            n = g.shape[-1]
            out[i, :, :n, :n] = g
        return torch.from_numpy(out).float()

    def save_pickle(self, path: str | Path):
        """Saves the fully processed (eager) dataset to a pickle file."""
        logging.info(f"Saving processed data to {path}...")
        obj = {
            "fe_curves": self.fe_curves, "fe_scaler": self.fe_scaler.features(),
            "protein_embeddings": self.protein_embeddings,
            "residue_features": self.residue_features, "residue_mask": self.residue_mask,
            "residue_scaler": self.residue_scaler.features(), "res_maps": self.res_maps,
            "res_map_paths": self.res_map_paths,
            "conditions": self.conditions, "data_table": self.data_table,
            "pdb_ids": self.pdb_ids, "chains": self.chains
        }
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        print("Save complete.")

    def _load_from_pickle(self, path: str | Path):
        """Loads state from a pickle file, bypassing processing."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        self.fe_curves = data["fe_curves"]
        self.fe_scaler = DataScaler.create_from_pd(pd.DataFrame(data["fe_scaler"]))
        self.protein_embeddings = data["protein_embeddings"]
        self.residue_features = data["residue_features"]
        self.residue_mask = data["residue_mask"]
        self.residue_scaler = DataScaler.create_from_pd(pd.DataFrame(data["residue_scaler"]))
        self.res_maps = data["res_maps"]
        self.res_map_paths = data["res_map_paths"]
        self.conditions = data["conditions"]
        self.data_table = data["data_table"]
        self.pdb_ids = data["pdb_ids"]
        self.chains = data["chains"]
        self.num_samples = len(self.pdb_ids)
        #self.lazy_res_map = True if self.res_maps is None else False

    @staticmethod
    def collate_fn_for_lazy_mode(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        A collate_fn specifically for the hybrid lazy mode. It pads the `res_map`.
        This is only needed if `lazy_res_map=True`.
        """
        batch = [sample for sample in batch if sample is not None]
        if not batch: return {}

        # Everything except res_map is already padded and stacked
        res_maps = [s.pop('res_map') for s in batch]

        # Dynamically pad the res_maps for this batch
        max_n = 250
        dim = res_maps[0].shape[0]
        padded_res_maps = torch.zeros(len(batch), dim, max_n, max_n, dtype=torch.float32)
        for i, g in enumerate(res_maps):
            n = min(g.shape[0], max_n)
            try:
                padded_res_maps[i, :, :n, :n] = g[:, :n, :n]
            except Exception as e:
                print(e, g.shape)


        # Collate the rest of the data
        collated_batch = {key: torch.stack([s[key] for s in batch]) for key in batch[0]}
        collated_batch['res_map'] = padded_res_maps

        return collated_batch

    @staticmethod
    def _convert_to_array(df: pd.DataFrame, drop_cols: Tuple[str, ...] = ("chain", "res_id", "res_idx")) -> np.ndarray:
        df = df.copy()

        if "res_name" in df:
            aa_one_hot_df = df["res_name"].map(AMINO_ACID_DIC).apply(pd.Series)
            aa_one_hot_df.columns = [f"aa_code_{i}" for i in range(5)]
            df = pd.concat([df, aa_one_hot_df], axis=1)
            df[[f"aa_code_{i}" for i in range(5)]] = df[
                [f"aa_code_{i}" for i in range(5)]].fillna(1)

        if "sec_type" in df:
            sec_one_hot_df = df["sec_type"].map(SEC_CODE).apply(pd.Series)
            sec_one_hot_df.columns = [f"sec_code_{i}" for i in range(2)]
            df = pd.concat([df, sec_one_hot_df], axis=1)
            df[[f"sec_code_{i}" for i in range(2)]] = df[
                [f"sec_code_{i}" for i in range(2)]].fillna(1)

        if "phi" in df:
            df['sin_phi'] = np.sin(df['phi'])
            df['cos_phi'] = np.cos(df['phi'])

        if "psi" in df:
            df['sin_psi'] = np.sin(df['psi'])
            df['cos_psi'] = np.cos(df['psi'])

        if "b_factor" in df:
            if df['b_factor'].isnull().all():
                median_value = 20.0
            else:
                median_value = df['b_factor'].median()

            df['b_factor'] = df['b_factor'].fillna(median_value)


        numeric_cols = [c for c in df.columns
                        if c not in (*drop_cols, "res_name", "sec_type", "sec_code", "chain", "res_id", "psi", "phi")]

        return df[numeric_cols].to_numpy(dtype=np.float32)

    def build_condition(self, features: dict, chain, res_start_idx: int = None, res_end_idx: int = None):
        # Set default residue indexes
        if res_start_idx is None:
            res_start_idx = 0
        if res_end_idx is None:
            res_end_idx = self.max_len

        # Convert protein embedding to tensor
        protein_embed = torch.as_tensor(features["protein_embed"]).unsqueeze(0)

        # Scale and pad res_feat
        residue_df = features["res_feat"]
        try:
            chain_df = residue_df[residue_df['chain'].astype(str) == str(chain)][res_start_idx: res_end_idx]
        except Exception as e:
            raise f'Error: {e}. Please check your residue index'

        res_feat = self._convert_to_array(chain_df)
        res_feat = np.expand_dims(res_feat, axis=0)
        res_feat, res_mask, _ = self._pad_and_normalize_features(res_feat)

        # Pad res_map
        res_map = features["res_map"]
        map = np.zeros((res_map.shape[0], self.max_len, self.max_len), dtype=res_map.dtype)
        n = min(res_map.shape[-1], self.max_len)
        map[:, :n, :n] = res_map[:, :n, :n]
        res_map = torch.from_numpy(map).unsqueeze(0).float()

        return {
            "protein_embed": protein_embed,
            "res_feat": res_feat,
            "res_mask": res_mask,
            "res_map": res_map,
        }

    def _get_dataset_pdb_ids(self) -> List[str]:
        """
        Fetch the per-item PDB_ID sequence (same length as the dataset).
        Priority:
          1) dataset.pdb_ids (list/tuple with len == len(dataset))
          2) dataset.data_table['PDB_ID'] with len == len(dataset)
          3) Fallback: read each item via __getitem__ and extract sample['pdb_id'] (may be slow)
        """
        # 1) Prefer a pre-computed list on the dataset
        if hasattr(self, "pdb_ids") and isinstance(self.pdb_ids, (list, tuple)):
            if len(self.pdb_ids) != len(self):
                raise ValueError("dataset.pdb_ids length does not match len(dataset).")
            return [str(x) for x in self.pdb_ids]

        # 2) Try a data table
        if hasattr(self, "data_table") and hasattr(self.data_table, "columns") \
                and "PDB_ID" in self.data_table.columns:
            ids = list(self.data_table["PDB_ID"])
            if len(ids) != len(self):
                raise ValueError("data_table['PDB_ID'] length does not match len(dataset).")
            return [str(x) for x in ids]

        # 3) Fallback: item-by-item lookup
        ids = []
        for i in range(len(self)):
            sample = self[i]
            if "pdb_id" not in sample:
                raise ValueError("Cannot find 'pdb_id' in sample and no pdb_ids/data_table available.")
            ids.append(str(sample["pdb_id"]))
        return ids

    def split_by_pdb_id(
            dataset,
            ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            test_only_ids: Sequence[str] = ('1emb', '1wlh', '1g1c', '1t1t'),
            seed: int = 42,
            shuffle: bool = True,
    ) -> Tuple[Subset, Subset, Subset, Dict[str, Any]]:
        """
        Group-aware split by unique PDB_ID:
          - All samples with the same PDB_ID are assigned to the same split.
          - Specific IDs in `test_only_ids` are forced into the test split (if present).
          - The split is based on the number of unique PDB_IDs (not raw sample count) to prevent leakage.

        Args
        ----
        dataset : torch.utils.data.Dataset-like object
            Must be indexable and either provide `dataset.pdb_ids`, or a
            `dataset.data_table['PDB_ID']` column, or return {'pdb_id': ...} in __getitem__.
        ratios : (float, float, float)
            Target (train, val, test) ratios over unique PDB_IDs. Must sum to 1.0.
            Note: exact counts may deviate due to rounding and forced test IDs.
        test_only_ids : Sequence[str]
            These PDB_IDs will appear only in the test split (if they exist in the dataset).
        seed : int
            Random seed for shuffling unique PDB_IDs.
        shuffle : bool
            Whether to shuffle the remaining PDB_IDs before allocation.

        Returns
        -------
        (train_subset, val_subset, test_subset, info_dict)
            Subsets are torch.utils.data.Subset over the original dataset.
            `info_dict` includes split diagnostics (unique counts, missing forced IDs, etc.).
        """
        assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

        pdb_ids_per_item = dataset._get_dataset_pdb_ids()
        n_items = len(pdb_ids_per_item)

        # Build PDB_ID -> list of item indices
        id2idx = defaultdict(list)
        for idx, pid in enumerate(pdb_ids_per_item):
            id2idx[str(pid)].append(idx)

        unique_ids = list(id2idx.keys())

        # Separate out the forced-test IDs
        test_only_ids = [str(x) for x in test_only_ids]
        present_forced = [pid for pid in test_only_ids if pid in id2idx]
        missing_forced = [pid for pid in test_only_ids if pid not in id2idx]

        # Remaining PDB_IDs to be allocated by ratio
        remaining_ids = [pid for pid in unique_ids if pid not in present_forced]
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(remaining_ids)

        # Target counts by unique-ID ratio
        U = len(unique_ids)
        n_train = int(round(U * ratios[0]))
        n_val = int(round(U * ratios[1]))
        n_test = U - n_train - n_val  # ensure total equals U

        # Start test with all forced IDs
        test_ids = list(present_forced)

        # Allocate the remaining IDs
        # Note: if forced test already exceeds the nominal test budget, we still keep all in test
        # and put the rest of remaining IDs into train/val/test in that order to keep sets disjoint.
        remaining_pointer = 0

        # Train allocation
        train_ids = remaining_ids[remaining_pointer: remaining_pointer + n_train]
        remaining_pointer += n_train

        # Val allocation
        val_ids = remaining_ids[remaining_pointer: remaining_pointer + n_val]
        remaining_pointer += n_val

        # Test allocation (whatever is left + any shortfall to reach nominal n_test)
        shortfall = max(0, n_test - len(test_ids))
        test_ids.extend(remaining_ids[remaining_pointer: remaining_pointer + shortfall])
        remaining_pointer += shortfall

        # If there are still leftovers (e.g., due to rounding/forced IDs), put all leftovers into test
        test_ids.extend(remaining_ids[remaining_pointer:])

        # Helper to gather per-item indices from ID groups
        def gather_indices(ids: Iterable[str]) -> List[int]:
            out = []
            for pid in ids:
                out.extend(id2idx[pid])
            return out

        train_idx = gather_indices(train_ids)
        val_idx = gather_indices(val_ids)
        test_idx = gather_indices(test_ids)

        # Sanity checks: disjointness
        def _disjoint(a, b) -> bool:
            return set(a).isdisjoint(set(b))

        assert _disjoint(train_idx, val_idx), "train and val indices are not disjoint."
        assert _disjoint(train_idx, test_idx), "train and test indices are not disjoint."
        assert _disjoint(val_idx, test_idx), "val and test indices are not disjoint."

        # Ensure forced test IDs are not leaking
        leakage_train = set(train_ids) & set(present_forced)
        leakage_val = set(val_ids) & set(present_forced)
        assert not leakage_train and not leakage_val, "Forced test IDs leaked into train/val."

        info = {
            "n_items": n_items,
            "n_unique_pdb": U,
            "missing_forced_test_ids": missing_forced,
            "ids": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
            "counts": {
                "train_items": len(train_idx),
                "val_items": len(val_idx),
                "test_items": len(test_idx),
                "train_unique": len(set(train_ids)),
                "val_unique": len(set(val_ids)),
                "test_unique": len(set(test_ids)),
            }
        }

        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx), info





