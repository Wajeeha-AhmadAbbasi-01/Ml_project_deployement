import pandas as pd 
import os
import logging

#Ensure the log directory exsits
log_dir = "logs"
os.makedirs(log_dir , exist_ok=True)

#loging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

#setting console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

#setting file handler for logging
log_file_path = os.path.join(log_dir ,"data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

#setting formater for logging
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

#putting filer and console handler back to logger object
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_data(data_url: str ) -> pd.DataFrame:
    """load data from csv"""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from : %s' , data_url)
        return df 
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s' , e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data : %s ', e)
        raise

def preprocess_data(df : pd.DataFrame)-> pd.DataFrame:
    """preprcess data removing unnecssary columns"""
    try: 
        df.drop(columns=df.columns[df.columns.str.contains('unnamed', case=False)], inplace=True)
        return df
        logger.debug("Data preprocessing completed")
    except Exception as e:
        logger.error("Unexpected error occur while preprocesing : %s", e)
        raise


def save_data(df:pd.DataFrame , data_path:str)->None:
    try:
        raw_data_path = os.path.join(data_path,"raw")
        os.makedirs(raw_data_path , exist_ok=True)
        df.to_csv(os.path.join(raw_data_path,"Dummy_teacher_data.csv"), index=False)
        logger.debug("Data saved to: %s " , raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occur while saving the data:%s ", e)
        raise


def main():
    try:
        url = "https://raw.githubusercontent.com/Wajeeha-AhmadAbbasi-01/Ml_project_deployement/refs/heads/main/Dummy_Teachers_Data_version2.csv"
        df = load_data(data_url=url)
        df = preprocess_data(df)
        save_data(df , data_path='./data')
    except Exception as e:
        logger.error("Failed to complete data ingestion : %s", e)
        print(f"Error : {e}")

    
if __name__=='__main__':
    main()

