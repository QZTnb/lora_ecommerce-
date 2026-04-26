import os
import pandas as pd
from datasets import Dataset, DatasetDict
from langdetect import detect
from src.config.config import Config

class DataProcessor:
    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
    
    def load_data(self, data_path):
        """加载数据"""
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, quotechar='"', quoting=0)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format")
        return df
    
    def detect_language(self, text):
        """检测文本语言"""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def preprocess_data(self, df):
        """预处理数据"""
        # 检测语言
        df['language'] = df['review'].apply(self.detect_language)
        
        # 过滤支持的语言
        df = df[df['language'].isin(self.config.SUPPORTED_LANGUAGES)]
        
        # 移除空值
        df = df.dropna(subset=['review', 'summary'])
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
    
    def create_dataset(self, df):
        """创建数据集"""
        dataset = Dataset.from_pandas(df)
        
        # 分割数据集
        train_test_split = dataset.train_test_split(test_size=0.2, seed=self.config.SEED)
        train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=self.config.SEED)
        
        dataset_dict = DatasetDict({
            'train': train_val_split['train'],
            'validation': train_val_split['test'],
            'test': train_test_split['test']
        })
        
        return dataset_dict
    
    def process(self, data_path):
        """处理数据的主方法"""
        df = self.load_data(data_path)
        df = self.preprocess_data(df)
        dataset = self.create_dataset(df)
        return dataset
