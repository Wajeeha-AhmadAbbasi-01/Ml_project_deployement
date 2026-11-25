import pandas as pd
import logging
import os

#Ensure the log directory exsits
log_dir = "logs"
os.makedirs(log_dir , exist_ok=True)

#loging configuration
logger = logging.getLogger("pre_processing")
logger.setLevel("DEBUG")

#setting console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

#setting file handler for logging
log_file_path = os.path.join(log_dir ,"preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

#setting formater for logging
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

#putting filer and console handler back to logger object
logger.addHandler(file_handler)
logger.addHandler(console_handler)





def preprocess_teacher_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean teacher dataset and generate structured teacher profile text
    suitable for embeddings / clustering / recommendation systems.
    """
    try:
        # -----------------------------
        # 1. Remove duplicates
        # -----------------------------
        if "Teacher_ID" in df.columns:
            df = df.drop_duplicates(subset="Teacher_ID")

        df = df.drop_duplicates(keep="first")
        logger.debug("Duplicates removed")

        # -----------------------------
        # 2. Handle missing values
        # -----------------------------
        df = df.fillna({
            "Skills": "general teaching",
            "Regional_Priority_Skills": "none",
            "Region": "unknown",
            "Subject": "general subject",
            "Grade_Level": "unknown",
            "Years_Exp": 0,
            "School_SES_Rating": 0,
            "Student_Teacher_Ratio": 0
        })
        logger.debug("Missing values handled")

        # -----------------------------
        # 3. Standardize text columns
        # -----------------------------
        text_cols = [
            "Subject",
            "Skills",
            "Regional_Priority_Skills",
            "Region",
            "Grade_Level"
        ]

        for col in text_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
            )
        logger.debug("Text columns are standardized")

        # -----------------------------
        # 4. Clean skill formatting
        # -----------------------------
        df["Skills"] = (
            df["Skills"]
            .str.replace("/", ",", regex=False)
            .str.replace("-", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
        )
        logger.debug("Skills column cleaned")

        # -----------------------------
        # 5. Convert numeric columns
        # -----------------------------
        df["Years_Exp"] = pd.to_numeric(df["Years_Exp"], errors="coerce").fillna(0)
        df["School_SES_Rating"] = pd.to_numeric(df["School_SES_Rating"], errors="coerce").fillna(0)
        df["Student_Teacher_Ratio"] = pd.to_numeric(df["Student_Teacher_Ratio"], errors="coerce").fillna(0)
        logger.debug("Numerical columns assured as numeric")

        # -----------------------------
        # 6. Experience bucketing
        # -----------------------------
        def experience_bucket(years):
            if years <= 3:
                return "beginner"
            elif years <= 7:
                return "intermediate"
            else:
                return "experienced"

        df["Experience_level"] = df["Years_Exp"].apply(experience_bucket)
        logger.debug("Experience level column created")

        # -----------------------------
        # 7. Teacher profile generator
        # -----------------------------
        def teacher_to_text(row):
            economic_label = (
                "low income school"
                if row["School_SES_Rating"] < 0.5
                else "moderate income school"
            )

            class_label = (
                "small class size"
                if row["Student_Teacher_Ratio"] < 25
                else "large class size"
            )

            return (
                f"subject: {row['Subject']}. "
                f"grade_level: {row['Grade_Level']}. "
                f"experience_level: {row['Experience_level']}. "
                f"skills: {row['Skills']}. "
                f"school_income: {economic_label}. "
                f"class_size: {class_label}. "
                f"regional_priority_skills: {row['Regional_Priority_Skills']}. "
                f"region: {row['Region']}."
            )

        df["Teacher_profile"] = df.apply(teacher_to_text, axis=1)
        logger.debug("Teacher profile created")

        # -----------------------------
        # 8. Final text cleanup
        # -----------------------------
        df["Teacher_profile"] = df["Teacher_profile"].str.replace(r"\s+", " ", regex=True)
        logger.debug("Teacher profile cleaned")

        return df

    except Exception as e:
        logger.error(f"Error in preprocessing teacher data: {e}")
        raise


def save_data(df:pd.DataFrame , data_path:str)->None:
    try:
        processed_data_path = os.path.join(data_path,"preprocessed")
        os.makedirs(processed_data_path , exist_ok=True)
        df.to_csv(os.path.join(processed_data_path,"Teacher_processed_data.csv"), index=False)
        logger.debug("Processed Data saved to: %s " , processed_data_path)
    except Exception as e:
        logger.error("Unexpected error occur while saving the data:%s ", e)
        raise

def main():
    try:
        df = pd.read_csv("./data/raw/Dummy_teacher_data.csv")
        logger.debug("Data loaded properly")
        df_preprocessed = preprocess_teacher_data(df)
        logger.debug("Data processed succefully")
        save_data(df_preprocessed, data_path="./data")
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data in file: %s", e)
    except Exception as e:
        logger.error("Failed to complete Data transformation process: %s" , e)
        print(f"Error{e}")


if __name__=="__main__":
    main()

     

