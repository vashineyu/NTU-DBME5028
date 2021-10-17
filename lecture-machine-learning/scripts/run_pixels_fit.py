import sys
import argparse
import re
import numpy as np
from collections import defaultdict
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

sys.path.append(".")
from src.utils import load_image_targets, get_result_metrics


def make_inputs(inputs):
    inputs = inputs.reshape((len(inputs), -1))
    inputs = np.float32(inputs) / 255.
    return inputs


class Trainer:
    def __init__(self, method):
        self._inputs_train = None
        self._targets_train = None
        self._inputs_valid = None
        self._targets_valid = None
        self.model = None

        if method == "logistic":
            self.train = self._train_logistic
            self.predict = self._predict_logistic
        elif method == "xgb":
            self.train = self._train_xgboost
            self.predict = self._predict_xgboost
        else:
            assert False, "Method {} not found".format(method)

    def set_train_data(self, inputs: np.ndarray, targets: np.ndarray):
        self._inputs_train = inputs
        self._targets_train = targets

    def set_valid_data(self, inputs: np.ndarray, targets: np.ndarray):
        self._inputs_valid = inputs
        self._targets_valid = targets

    def _train_logistic(self, params=None):
        self.model = LogisticRegression().fit(
            self._inputs_train,
            self._targets_train
        )

    def _predict_logistic(self, inputs):
        return self.model.predict_log_proba(inputs)[:, 1]

    def _train_xgboost(self, params=None):
        train_data = xgb.DMatrix(self._inputs_train, label=self._targets_train)
        valid_data = xgb.DMatrix(self._inputs_valid, label=self._targets_valid)

        config = {
            "objective": "binary:logistic",
            "nthread": 8,
            "eval_metric": "auc",
            "tree_method": "gpu_hist",
            "gpu_id": 0
        }
        if params:
            config.update(params)
        eval_list = [(valid_data, "eval"), (train_data, "train")]
        self.model = xgb.train(
            config,
            train_data,
            num_boost_round=100,
            evals=eval_list
        )

    def _predict_xgboost(self, inputs):
        inputs = xgb.DMatrix(inputs)
        return self.model.predict(inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="logistic",
        help="Model Type, support [logistic|xgb]"
    )
    parser.add_argument(
        "--data_root",
        default="./data/",
        help="Where is the data"
    )
    args = parser.parse_args()
    print(args)

    files = glob("{}/*[x|y].h5".format(args.data_root))
    filepath = defaultdict(dict)
    for i in files:
        file_ = Path(i).stem
        condition = re.findall(pattern="train|valid|test", string=file_)[0]
        target_type = re.findall(pattern="_[x|y]", string=file_)[0]
        filepath[condition][target_type] = i

    ## ONLY FOR COURSE DEMO (NORMALLY, YOU SHOULD USE TRAIN FILES)
    train_x, train_y = load_image_targets(
        filepath["valid"]["_x"],
        filepath["valid"]["_y"]
    )
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=0.1
    )

    test_x, test_y = load_image_targets(
        filepath["test"]["_x"],
        filepath["test"]["_y"]
    )

    train_x_vector = make_inputs(train_x)
    valid_x_vector = make_inputs(valid_x)
    test_x_vector = make_inputs(test_x)

    trainer = Trainer(method=args.model)
    trainer.set_train_data(train_x_vector, train_y)
    trainer.set_valid_data(valid_x_vector, valid_y)

    print("Train Start")
    trainer.train()
    y_pred = trainer.predict(test_x_vector)

    results = get_result_metrics(
        y_true=test_y,
        y_score=y_pred
    )
    print(results)