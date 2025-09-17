import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
## model 
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.ML_Project.utils import save_object

from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''

        this function is responsible for data transformation
        '''
        try:
            num_features =['writing_score','reading_score']
            cat_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                    ]
            )
            
            cat_pipeline = Pipeline(steps = [
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    ]
            )
            
            logging.info(f"categorical Columns :{cat_features}")
            logging.info(f"numerical columns :{num_features}")
            
            preprocessor = ColumnTransformer(
                  [
                      ("num_pipeline",num_pipeline,num_features),
                      ("cat_pipeline",cat_pipeline,cat_features)

                    ]
            )
            
            return preprocessor
    
        except Exception as e:
            raise CustomException(e,sys) 
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_column = ['writing_score','reading_score']

            ## divide the train data set into independent and dependent features


            input_feature_train_df = train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            ##divide the test data set into independent and dependent features

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]



            logging.info("Applying preprocessing on training and testing dataframe ")

            input_features_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr=np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            ) 






        except Exception as e:
            raise CustomException(e,sys)