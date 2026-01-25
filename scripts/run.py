from Gen_Unfold.src import (train_pipeline, curve_prediction)


def train():
    seeds = [42, 37, 56]

    for i in range(0, 3):
        train_pipeline(
            fr'C:/PycharmProjects/pythonProject1/Gen_SMFS/config/ablation_study/E04.yaml',
            #checkpoint_path=r"C:/RawData/checkpoints/DiffusionModelTrainer\202510291514\checkpoint_epoch_80.pt",
            seed=seeds[i])


def inference():
    pretrained_model_path = r"C:\RawData\checkpoints\DiffusionModelTrainer\202510251933"
    num_samples = 1024

    # GFP 1EMB
    curve_prediction(pretrained_model_path,
                     pdb_id_or_path=r'D:\Dataset\Mech\mmCif_files\1emb.cif',
                     chain='A',
                     feature_path=r'D:\Dataset\Mech\features',
                     save_path=r'D:\Dataset\Mech\features\1emb\old\predictions.npy',
                     num_samples=num_samples,
                     device='cuda',
                     eta=0.0,
                     )




if __name__ == '__main__':

    #train()
    inference()







