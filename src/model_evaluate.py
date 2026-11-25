import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging
from typing import Dict , Any

log_dir = "logs"
os.makedirs(log_dir , exist_ok=True)

#loging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

#setting console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

#setting file handler for logging
log_file_path = os.path.join(log_dir ,"model_evaluation.log")
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
    logger.debug("Embeddings loaded")
    return np.load(embeddings_path)




def evaluate_k_range(
    embeddings: np.ndarray,
    k_values: range,
    params : dict[str, Any]

) -> tuple[pd.DataFrame , str]:
    results = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=params["random_state"],
            n_init=params["n_init"],
            init=params["init"]
        )
        labels = model.fit_predict(embeddings)

        result = {
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_score": float(silhouette_score(embeddings, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(embeddings, labels))
        }
        results.append(result)
        
    model_name = model.__class__.__name__   
    logger.debug("Model evaluated on different k values")
    return pd.DataFrame(results), model_name


def select_best_k(results_df: pd.DataFrame) -> tuple[int ,float , float]:
    try:
        # Example rule: maximize silhouette score
        best_row = results_df.sort_values("silhouette_score", ascending=False).iloc[0]
        logger.debug("best k_value finded")
        best_k = best_row["k"]
        silhouette_score =best_row["silhouette_score"] 
        inertia = best_row["inertia"]
        return int(best_k),silhouette_score, inertia
    except Exception as e:
        logger.error("Error occured during selecting k best : %s" , e)
        raise


def save_results(results_df: pd.DataFrame, output_csv_path: str, metrics: str , model_name:str , params:dict[str , Any] , params_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        logger.debug("Results of different k values are stored")

        best_k , silhouette_score , inertia = select_best_k(results_df)
        info = {
            "best_k":best_k,
            "silhouette_score":silhouette_score,
            "inertia":inertia,
            "model": model_name

        }
       #saving metrics
        with open(metrics, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
            logger.debug("metrics stored json stored")
        
        #saving parameters
        #adding best k into parametrs also
        params["best_k"]=best_k
        with open(params_path, "w", encoding="utf-8") as obj:
            json.dump(params, obj, indent=4)
            logger.debug(" Model parameters stored json stored")
        
    except Exception as e:
        logger.error("Erro occured during saving results %s" ,e)
        raise


def run_evaluation_pipeline(
    embeddings_path: str,
    output_csv_path: str,
    params_path:str,
    metrics: str,
    params : dict[str , Any],
    k_values=range(2, 15),
   

) -> None:
    try:
        embeddings = load_embeddings(embeddings_path)
        results_df , model_name = evaluate_k_range(embeddings, k_values, params)
        save_results(results_df, output_csv_path, metrics , model_name ,params , params_path)
    except Exception as e:
        logger.error("Evaluation pipeline not completed : %s " ,e)
        raise

 


if __name__ == "__main__":
    run_evaluation_pipeline(
        embeddings_path="artifacts/embeddings/embeddings.npy",
        output_csv_path="artifacts/evaluation/kmeans_k_comparison.csv",
        metrics="artifacts/evaluation/metrics.json",
        params_path="artifacts/evaluation/params.json",
        params={
            "random_state":42,
            "n_init":"auto",
            'init':'k-means++' 
        }
    )