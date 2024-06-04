import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from omegaconf import OmegaConf

def train(config):
    print('Training the model...')
    train_inputs = joblib.load(config.features.train_features_save_path)
    train_outputs = pd.read_csv(config.data.train_csv_save_path)['label'].values
    
    penalty = config.train.penalty
    C = config.train.C
    solver = config.train.solver
    max_iter = config.train.max_iter
    
    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
    model.fit(train_inputs, train_outputs)
    
    joblib.dump(model, config.train.model_save_path)

if __name__ == '__main__':
    config = OmegaConf.load('./params.yaml')
    train(config)