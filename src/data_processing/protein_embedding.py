import os

import numpy as np
import pandas as pd
import tqdm
from transformers import EsmModel, EsmTokenizer
import torch
from .prtein_feature_extractor import cif_chain_sequences

def load_model(model_name="facebook/esm2_t33_650M_UR50D", cache_dir="D:\\pretained_model\\esm\\"):
    model_name = model_name
    tokenizer = EsmTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False, cache_dir=cache_dir)
    model.eval()
    return model, tokenizer

def embed_sequence(sequence, model, tokenizer):
    inputs = tokenizer(sequence, return_tensors="pt").to(torch.device("cuda"))
    with torch.no_grad():
        outputs = model(**inputs)
    residue_embeddings = outputs.last_hidden_state[0, 1:-1]  # shape: [L, 1280]
    #sequence_embedding = residue_embeddings.mean(dim=0)  # shape: [1280]
    #sequence_embedding = residue_embeddings
    return residue_embeddings.cpu().numpy()


def process_batch(pdb_path, output_path):
    filenames = os.listdir(pdb_path)
    model, tokenizer = load_model()
    model.to("cuda")
    for filename in tqdm.tqdm(filenames):
        pdb = os.path.join(pdb_path, filename)
        pdb_file = os.path.join(output_path, filename[:4])
        if not os.path.exists(pdb_file):
            os.makedirs(pdb_file, exist_ok=True)

        embedding_filenames = os.listdir(pdb_file)

        end = False
        for embedding_filename in embedding_filenames:
            if embedding_filename.startswith('embedding'):
                end = True
        if end:
            continue

        chain_sequences = cif_chain_sequences(pdb)

        for chain in chain_sequences.keys():
            residue_embedding_path = os.path.join(output_path, filename[:4], f'residue_embedding_{chain}.npy')
            if not os.path.exists(residue_embedding_path):
                print(f'Embedding to {residue_embedding_path}')
                seq = chain_sequences[chain]
                sequence_embedding = embed_sequence(seq, model, tokenizer)
                np.save(residue_embedding_path, sequence_embedding)

            embedding_path = os.path.join(output_path, filename[:4], f'embedding_{chain}.npy')
            if not os.path.exists(embedding_path):
                print(f'Embedding to {embedding_path}')
                seq = chain_sequences[chain]
                sequence_embedding = embed_sequence(seq, model, tokenizer)
                np.save(embedding_path, sequence_embedding.mean(0))


def process_from_data_table(data_table_path, output_path):
    model, tokenizer = load_model()
    model.to("cuda")

    data_table = pd.read_excel(data_table_path, sheet_name='Sheet3')
    for index, row in data_table.iterrows():
        mol_name = row['MoleculeString']
        output_dir = os.path.join(output_path, mol_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        embedding_filenames = os.listdir(output_dir)

        end = False
        for embedding_filename in embedding_filenames:
            if embedding_filename.startswith('embedding'):
                end = True
        if end:
            continue

        seq = row['Sequence']
        chain = 'A'  # Assuming single chain 'A' for simplicity; modify as needed

        residue_embedding_path = os.path.join(output_dir, f'residue_embedding_{chain}.npy')
        try:
            if not os.path.exists(residue_embedding_path):
                print(f'Embedding to {residue_embedding_path}')
                sequence_embedding = embed_sequence(seq, model, tokenizer)
                np.save(residue_embedding_path, sequence_embedding)

            embedding_path = os.path.join(output_dir, f'embedding_{chain}.npy')
            if not os.path.exists(embedding_path):
                print(f'Embedding to {embedding_path}')
                sequence_embedding = embed_sequence(seq, model, tokenizer)
                np.save(embedding_path, sequence_embedding.mean(0))
        except Exception as e:
            print(f"Error processing {mol_name}: {e}")


