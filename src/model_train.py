import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
from typing import Dict , Any
import logging

#Ensure the log directory exsits
log_dir = "logs"
os.makedirs(log_dir , exist_ok=True)

#loging configuration
logger = logging.getLogger("model_train")
logger.setLevel("DEBUG")

#setting console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

#setting file handler for logging
log_file_path = os.path.join(log_dir ,"model_train.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

#setting formater for logging
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

#putting filer and console handler back to logger object
logger.addHandler(file_handler)
logger.addHandler(console_handler)





def load_embeddings(embeddings_path: str) -> np.ndarray:
    return np.load(embeddings_path)


def load_metadata(metadata_path: str) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def validate_alignment(embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
    try:
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch found: metadata has {len(metadata)} rows but embeddings have {embeddings.shape[0]} rows."
            )

        if "embedding_row_id" not in metadata.columns:
            raise ValueError("Missing 'embedding_row_id' column in metadata.")
    except Exception as e:
        logger.error('Alignment not validated %s',e)
        raise


def train_kmeans(
    embeddings: np.ndarray,
    params:dict[str,Any],
):
   try:
        model = KMeans(
            n_clusters=params['best_k'],
            random_state=params["random_state"],
            n_init=params["n_init"],
            init=params["init"],
        )
        cluster_labels = model.fit_predict(embeddings)
        logger.debug("Model Trained")
        return model, cluster_labels
   except Exception as e:
       logger.error("Error occured during Training %s",e)
       raise


def save_model(model, model_output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, "wb") as f:
            joblib.dump(model, f)
        logger.debug("Model saved Succeesfully")
    except Exception as e:
        logger.error("Error occured while saving model %s",e)
        raise


def save_clustered_output(
    metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_path: str
) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df = metadata.copy()
        output_df["cluster_id"] = cluster_labels
        output_df.to_csv(output_path, index=False)
        logger.debug("Ouput saved")
    except Exception as e:
        logger.error('Error occured while saving final output %s', e)
        raise


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return params
    except Exception as e:
        logger.error('Error occured while loading parameters%s', e)
        raise




def run_training_pipeline(
    embeddings_path: str,
    metadata_path: str,
    model_output_path: str,
    clustered_output_path: str,
    params_path:str,
    
) -> None:
    embeddings = load_embeddings(embeddings_path)
    logger.debug("Embeddings loaded")
    metadata = load_metadata(metadata_path)
    logger.debug("Meta_data loaded")
    params = load_params(params_path)
    logger.debug("Prameters loaded ")

    validate_alignment(embeddings, metadata)
    logger.debug("meta data and embedding allignment validated")

    model, cluster_labels = train_kmeans(
        embeddings=embeddings,
        params=params
    )

    save_model(model, model_output_path)
    save_clustered_output(metadata, cluster_labels, clustered_output_path)



if __name__ == "__main__":
    run_training_pipeline(
        embeddings_path="artifacts/embeddings/embeddings.npy",
        metadata_path="artifacts/embeddings/metadata.csv",
        model_output_path="artifacts/models/kmeans_model.joblib",
        clustered_output_path="artifacts/model_output/clustered_output.csv",
        params_path= "artifacts/evaluation/params.json",
    )