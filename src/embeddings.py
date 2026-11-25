import pandas as pd
import os
import logging
import json
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

#Ensure the log directory exsits
log_dir = "logs"
os.makedirs(log_dir , exist_ok=True)

#loging configuration
logger = logging.getLogger("embeddings_creation")
logger.setLevel("DEBUG")

#setting console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

#setting file handler for logging
log_file_path = os.path.join(log_dir ,"embeddings_creation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

#setting formater for logging
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

#putting filer and console handler back to logger object
logger.addHandler(file_handler)
logger.addHandler(console_handler)



def load_preprocessed_data(input_csv_path: str) -> pd.DataFrame:
    """load the data"""
    try:
        df = pd.read_csv(input_csv_path)
        df = df.reset_index(drop=True)
        logger.debug("Data loaded succesfully from : %s", input_csv_path)
        return df
    except Exception as e:
        logger.error("Error occured in loading data: %s" , e)
        raise


def validate_text_column(df: pd.DataFrame, text_column: str) -> None:
    """validate the teacher profile column exsist and have value"""
    if text_column not in df.columns:
        raise ValueError(f"Missing required text column: {text_column}")

    if df[text_column].isnull().all():
        raise ValueError(f"Column '{text_column}' is completely empty.")
    logger.debug("Teacher profile column validation done Succefully")


def add_embedding_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """generates embedding row id in data frame so that after it can be mapped back"""
    try:
        df = df.copy()
        df["embedding_row_id"] = range(len(df))
        logger.debug("Embedding id generated succefully")
        return df
    except Exception as e:
        logger.error("Error occured in generating embedding row id : %s" , e)
        raise



def generate_embeddings(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logger.debug("Embeddings created")
        normalized_embeddings = normalize(embeddings)
        logger.debug("Embeddings are normalized")
        return normalized_embeddings
    except Exception as e:
        logger.error("Error occured in generating embedding  : %s" , e)
        raise


def save_embeddings(normalized_embeddings: np.ndarray, output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, normalized_embeddings)
        logger.debug('Embedding has be saved')
    except Exception as e:
        logger.error("Error occured in saving embeddings : %s" , e)
        raise


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.debug("Embeddings meta data has been saved")
    except Exception as e:
        logger.error("Error occured in saving embeddings metadata : %s" , e)
        raise


def save_embedding_info(
    info_output_path: str,
    model_name: str,
    text_column: str,
    embeddings: np.ndarray
) -> None:
    info = {
        "embedding_model": model_name,
        "text_column": text_column,
        "num_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
    }
    try:
        os.makedirs(os.path.dirname(info_output_path), exist_ok=True)
        with open(info_output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
        logger.debug("Embedding info saved")
    except Exception as e:
         logger.error("Error occured in saving embeddings info : %s" , e)
         raise
 

def run_embedding_pipeline(
    input_csv_path: str,
    embeddings_output_path: str,
    metadata_output_path: str,
    info_output_path: str,
    text_column: str = "Teacher_profile",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> None:
    
    try:
        df = load_preprocessed_data(input_csv_path)
        validate_text_column(df, text_column)
        df = add_embedding_row_id(df)

        texts = df[text_column].fillna("").astype(str).tolist()
        logger.debug("Teacher profile column converted into list of text ")
        normalized_embeddings = generate_embeddings(texts, model_name=model_name)

        save_embeddings(normalized_embeddings, embeddings_output_path)
        save_metadata(df, metadata_output_path)
        save_embedding_info(info_output_path, model_name, text_column, normalized_embeddings)

        logger.debug("Embedding pipeline completed successfully.")
    except Exception as e:
        logger.error("Embedding pipeline doesnot run succefully")


if __name__ == "__main__":
    run_embedding_pipeline(
        input_csv_path="./data/preprocessed/Teacher_processed_data.csv",
        embeddings_output_path="./artifacts/embeddings/embeddings.npy",
        metadata_output_path="./artifacts/embeddings/metadata.csv",
        info_output_path="./artifacts/embeddings/embedding_info.json"
    )