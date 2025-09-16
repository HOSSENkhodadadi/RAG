import os
import pickle
import numpy as np
from typing import Any, Iterable
from src.config import Config

class Dataset:
    def __init__(self, file_name: str):
        self._file_name: str = file_name
        self.dataset_name: Config.DatasetNames = None
        self.passage_list = list[str]()
        self.query_list = list[str]()
        self.passage_augmentation_list = list[dict[str, dict[str, int]]]()
        self.query_augmentation_list = list[dict[str, dict[str, int]]]()
        self.augmentation_dict = dict[str, set[int]]()
        self.relation_list = list[set[int]]()
        self.train_set = set[int]()
        self.validation_set = set[int]()
        self.test_set = set[int]()
        self._stat_dict = {
            'passages': dict[str, int](),
            'queries': dict[str, int](),
            'augmentations': dict[str, int](),
            'relations': dict[str, int](),
            'learning': dict[str, int]()
        }
        path = os.path.join(Config.Paths.DATASET_ROOT, f'{file_name}-no-augmentation.pickle')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, 'rb') as f:
            public_dataset = pickle.load(f)
            for attr in public_dataset:
                setattr(self, attr, public_dataset[attr])

        if self.dataset_name not in {item.value for item in Config.DatasetNames}:
            raise ValueError('Invalid dataset name')
        self.dataset_name = Config.DatasetNames(self.dataset_name)

        self._update_stat()

    def __str__(self):
        output_list = [f'names -> file: {self._file_name}, dataset: {self.dataset_name}']
        for stat in self._stat_dict:
            if not self._stat_dict[stat]:
                continue
            items = ', '.join(f'{k}: {v}' for k, v in self._stat_dict[stat].items())
            output_list.append(f'{stat} -> {items}')
        return '\n'.join(output_list)

    def _update_stat(self):
        def count(key: str, suffix: str, lst: list[Any]):
            self._stat_dict[key][f'total_{suffix}'] = len(lst)

        def compute_lengths(key: str, suffix: str, lst: list[Iterable]):
            if not lst:
                return
            self._stat_dict[key][f'min_{suffix}'] = min(len(x) for x in lst)
            self._stat_dict[key][f'avg_{suffix}'] = round(sum(len(x) for x in lst) / len(lst))
            self._stat_dict[key][f'max_{suffix}'] = max(len(x) for x in lst)

        count('passages', '', self.passage_list)
        compute_lengths('passages', 'length', self.passage_list)
        count('queries', '', self.query_list)
        compute_lengths('queries', 'length', self.query_list)

        for aug in self.augmentation_dict:
            count('augmentations', f'queries_augmented_with_{aug}', self.augmentation_dict[aug])

        compute_lengths('relations', 'related_passages', self.relation_list)
        count('learning', 'queries_in_train_set', self.train_set)
        count('learning', 'queries_in_validation_set', self.validation_set)
        count('learning', 'queries_in_test_set', self.test_set)

    def _get_recall(self, query_index: int, retrieved_indices: np.ndarray) -> list[float]:
        true_set = self.relation_list[query_index]
        total = len(true_set)
        return [
            len(true_set.intersection(retrieved_indices[:k])) / total
            for k in range(1, retrieved_indices.size + 1)
        ]

    def _get_mrr(self, query_index: int, retrieved_indices: np.ndarray, optimistic=True) -> float:
        true_set = self.relation_list[query_index]
        if optimistic:
            for rank, idx in enumerate(retrieved_indices, start=1):
                if idx in true_set:
                    return 1.0 / rank
            return 0.0
        else:
            total = len(true_set)
            for rank in range(total, retrieved_indices.size + 1):
                if len(true_set.intersection(retrieved_indices[:rank])) == total:
                    return total / rank
            return 0.0

    def get_metrics(self, query_indices: list[int], retrieved_matrix: np.ndarray):
        recalls = {}
        recall_star = {}
        o_mrrs = []
        p_mrrs = []

        for i, query_idx in enumerate(query_indices):
            true_set = self.relation_list[query_idx]
            if not true_set:
                continue
            r_list = self._get_recall(query_idx, retrieved_matrix[i])
            o_mrr = self._get_mrr(query_idx, retrieved_matrix[i], optimistic=True)
            p_mrr = self._get_mrr(query_idx, retrieved_matrix[i], optimistic=False)

            for k, r in enumerate(r_list, start=1):
                recalls.setdefault(k, []).append(r)
                if k == len(true_set):
                    recall_star.setdefault(len(true_set), []).append(r)

            o_mrrs.append(o_mrr)
            p_mrrs.append(p_mrr)

        avg_recall = {k: sum(v)/len(v) for k, v in recalls.items()}
        avg_recall_star = {k: sum(v)/len(v) for k, v in recall_star.items()}
        return avg_recall, avg_recall_star, sum(o_mrrs)/len(o_mrrs), sum(p_mrrs)/len(p_mrrs)

    def print_metrics(self, query_indices: list[int], retrieved_matrix: np.ndarray):
        recall, recall_star, o_mrr, p_mrr = self.get_metrics(query_indices, retrieved_matrix)
        print(f'Dataset Name -> {self.dataset_name}')
        print('Recall ->', ' | '.join(f'{k}: {100 * v:.2f}%' for k, v in recall.items()))
        print('Cluster Recall ->', ' | '.join(f'{k}: {100 * v:.2f}%' for k, v in recall_star.items()))
        print(f'MRR -> optimistic: {100 * o_mrr:.2f}% | pessimistic: {100 * p_mrr:.2f}%')
