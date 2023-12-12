import pandas as pd

class ModelGenerator():
    def __init__(self):
        self.data = pd.read_csv('data_regression.csv').drop(['nationality', 'positions'],axis=1)

    # returns numeric columns
    def get_numeric_columns(self):
        return self.data.select_dtypes(include=['int64','float64']).columns
    
    # returns all columns
    def get_all_cols(self):
        return self.data.columns
    
    # returns a dataframe with the selected columns
    def get_selected_data(self,cols):
        return self.data[cols]