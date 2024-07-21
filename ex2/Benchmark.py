import pandas as pd

class Benchmark:
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = filepath.split('/')[-1]
        self.name = self.name.split('.')[0]
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.filepath)
        df = df.astype(float)
        return df
    
    def get_time(self, np):
        if 'np' in self.df.columns:
            return self.df[self.df['np'] == np]['time'].values[0]
        elif 'nt' in self.df.columns:
            return self.df[self.df['nt'] == np]['time'].values[0]
        