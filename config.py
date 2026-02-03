from utils import train


class Config:
    SEED = 42

    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"

    BATCH_SIZE = 256
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    EPOCHS = 30
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 4

    TRAIN_DF_PATH = "data/imdb_train.csv"
    VAL_DF_PATH = "data/imdb_val.csv"
    SAVE_PATH = "best_model.pth"



device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Config()

train(cfg, device)
