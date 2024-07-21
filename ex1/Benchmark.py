import os
import pandas as pd

class Benchmark:
    OPERATIONS = {
        "bcast": [
            "ignore",
            "basic_linear",
            "chain",
            "pipeline",
            "split_binary_tree",
            "binary_tree",
            "binomial_tree",
            "knomial_tree",
            "scatter_allgather",
            "scatter_allgather_ring"
        ],
        "reduce": [
            "ignore",
            "linear",
            "chain",
            "pipeline",
            "binary",
            "binomial",
            "in-order_binary",
            "rabenseifner"
        ]
    }
    
    def __init__(self, filepath):
        self.filepath = filepath # ex: /reduce/np$np-a$alg.csv
        filename = os.path.basename(filepath)
        self.operation = os.path.basename(os.path.dirname(filepath))
        self.processes, self.algorithm = filename.split("-")
        self.processes = self.processes[2:]
        self._parse_algorithm()
        self._load_data()
    
    def _parse_algorithm(self):
        self.algorithm = self.algorithm.replace(".csv", "")
        self.algorithm = self.algorithm.replace("a", "")
        self.algorithm = self.OPERATIONS[self.operation][int(self.algorithm)]
    
    def _load_data(self):
        with open(self.filepath, 'r') as file:
            file_content = file.read()
            data_lines = [line.strip() for line in file_content.strip().split('\n') if line and not line.startswith('#')]
            
            df = pd.DataFrame([line.split() for line in data_lines], columns=['Size', 'Avg_Latency', 'Min_Latency', 'Max_Latency', 'Iterations'])
            df = df.astype({'Size': int, 'Avg_Latency': float, 'Min_Latency': float, 'Max_Latency': float, 'Iterations': int})
            
            self.data = df