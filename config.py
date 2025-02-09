class Config:
    MODEL_NAME = "zora-transformer"
    TRAIN_DATA_PATH = "data/train.json"
    VAL_DATA_PATH = "data/val.json"
    MODEL_SAVE_PATH = "models/zora_model.pt"
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    EPOCHS = 5