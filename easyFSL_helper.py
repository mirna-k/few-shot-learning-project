import random
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Iterator
from abc import abstractmethod
from numpy import ndarray
from torch import Tensor
from collections import defaultdict

from tqdm.notebook import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)


"""
    EasyFSL helper
    Source: https://github.com/sicara/easy-few-shot-learning.git
"""

def predict_embeddings(dataloader: DataLoader, model: nn.Module, device = None,) -> pd.DataFrame:
    """
    Predict embeddings for a dataloader.
    Args:
        dataloader: dataloader to predict embeddings for. Must deliver tuples (images, class_names)
        model: model to use for prediction
        device: device to cast the images to. If none, no casting is performed. Must be the same as
            the device the model is on.
    Returns:
        dataframe with columns embedding and class_name
    """
    all_embeddings = []
    all_class_names = []
    with torch.no_grad():
        for images, class_names in tqdm(dataloader, unit="batch", desc="Predicting embeddings"):
            if device is not None:
                images = images.to(device)

            all_embeddings.append(model(images).detach().cpu())

            if isinstance(class_names, torch.Tensor):
                all_class_names += class_names.tolist()
            else:
                all_class_names += class_names

    concatenated_embeddings = torch.cat(all_embeddings)

    return pd.DataFrame({"embedding": list(concatenated_embeddings), "class_name": all_class_names})


""" Few Shot Dataset """

class FewShotDataset(Dataset):
    """
    Abstract class for all datasets used in a context of Few-Shot Learning.
    The tools we use in few-shot learning, especially TaskSampler, expect an
    implementation of FewShotDataset.
    Compared to PyTorch's Dataset, FewShotDataset forces a method get_labels.
    This exposes the list of all items labels and therefore allows to sample
    items depending on their label.
    """

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __getitem__ method."
        )

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __len__ method."
        )

    @abstractmethod
    def get_labels(self) -> List[int]:
        raise NotImplementedError(
            "Implementations of FewShotDataset need a get_labels method."
        )


class FeaturesDataset(FewShotDataset):
    def __init__(self, labels: List[int], embeddings: Tensor, class_names: List[str]):
        """
        Initialize a FeaturesDataset from explicit labels, class_names and embeddings.
        You can also initialize a FeaturesDataset from:
            - a dataframe with from_dataframe();
            - a dictionary with from_dict();
        Args:
            labels: list of labels, one for each embedding
            embeddings: tensor of embeddings with shape (n_images_for_this_class, **embedding_dimension)
            class_names: the name of the class associated to each integer label
                (length is the number of unique integers in labels)
        """
        self.labels = labels
        self.embeddings = embeddings
        self.class_names = class_names

    @classmethod
    def from_dataframe(cls, source_dataframe: pd.DataFrame):
        """
        Instantiate a FeaturesDataset from a dataframe.
        embeddings and class_names are directly inferred from the dataframe's content,
        while labels are inferred from the class_names.
        Args:
            source_dataframe: must have the columns embedding and class_name.
                Embeddings must be tensors or numpy arrays.
        """
        if not {"embedding", "class_name"}.issubset(source_dataframe.columns):
            raise ValueError(
                f"Source dataframe must have the columns embedding and class_name, "
                f"but has columns {source_dataframe.columns}"
            )

        class_names = list(source_dataframe.class_name.unique())
        labels = list(
            source_dataframe.class_name.map(
                {
                    class_name: class_id
                    for class_id, class_name in enumerate(class_names)
                }
            )
        )
        if len(source_dataframe) == 0:
            warnings.warn(
                UserWarning(
                    "Empty source dataframe. Initializing an empty FeaturesDataset."
                )
            )
            embeddings = torch.empty(0)
        else:
            embeddings = torch.from_numpy(np.stack(list(source_dataframe.embedding)))

        return cls(labels, embeddings, class_names)

    @classmethod
    def from_dict(cls, source_dict: Dict[str, Union[ndarray, Tensor]]):
        """
        Instantiate a FeaturesDataset from a dictionary.
        Args:
            source_dict: each key is a class's name and each value is a numpy array or torch tensor
                with shape (n_images_for_this_class, **embedding_dimension)
        """
        class_names = []
        labels = []
        embeddings_list = []
        for class_id, (class_name, class_embeddings) in enumerate(source_dict.items()):
            class_names.append(class_name)
            if isinstance(class_embeddings, ndarray):
                embeddings_list.append(torch.from_numpy(class_embeddings))
            elif isinstance(class_embeddings, Tensor):
                embeddings_list.append(class_embeddings)
            else:
                raise ValueError(
                    f"Each value of the source_dict must be a ndarray or torch tensor, "
                    f"but the value for class {class_name} is {class_embeddings}"
                )
            labels += len(class_embeddings) * [class_id]
        return cls(labels, torch.cat(embeddings_list), class_names)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.embeddings[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> List[int]:
        return self.labels

    def number_of_classes(self):
        return len(self.class_names)


""" Task Sampler """

GENERIC_TYPING_ERROR_MESSAGE = (
    "Check out the output's type of your dataset's __getitem__() method."
    "It must be a Tuple[Tensor, int] or Tuple[Tensor, 0-dim Tensor]."
)

class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(self, dataset: FewShotDataset, n_way: int, n_shot: int, n_query: int, n_tasks: int,):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have implement get_labels() from
                FewShotDataset.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label: Dict[int, List[int]] = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

        self._check_dataset_size_fits_sampler_parameters()

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        """
        Sample n_way labels uniformly at random,
        and then sample n_shot + n_query items for each label, also uniformly at random.
        Yields:
            a list of indices of length (n_way * (n_shot + n_query))
        """
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    for label in random.sample(
                        sorted(self.items_per_label.keys()), self.n_way
                    )
                ]
            ).tolist()

    def episodic_collate_fn(self, input_data: List[Tuple[Tensor, Union[Tensor, int]]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images of shape (n_way * n_shot, n_channels, height, width),
                - their labels of shape (n_way * n_shot),
                - query images of shape (n_way * n_query, n_channels, height, width)
                - their labels of shape (n_way * n_query),
                - the dataset class ids of the class sampled in the episode
        """
        input_data_with_int_labels = self._cast_input_data_to_tensor_int_tuple(input_data)

        true_class_ids = list({x[1] for x in input_data_with_int_labels})
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data_with_int_labels])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data_with_int_labels]
        ).reshape((self.n_way, self.n_shot + self.n_query))

        support_images = all_images[:, : self.n_shot].reshape((-1, *all_images.shape[2:]))
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()
        return (support_images, support_labels, query_images, query_labels, true_class_ids)

    @staticmethod
    def _cast_input_data_to_tensor_int_tuple(input_data: List[Tuple[Tensor, Union[Tensor, int]]]) -> List[Tuple[Tensor, int]]:
        """
        Check the type of the input for the episodic_collate_fn method, and cast it to the right type if possible.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            the input data with the labels cast to int
        Raises:
            TypeError : Wrong type of input images or labels
            ValueError: Input label is not a 0-dim tensor
        """
        for image, label in input_data:
            if not isinstance(image, Tensor):
                raise TypeError(
                    f"Illegal type of input instance: {type(image)}. "
                    + GENERIC_TYPING_ERROR_MESSAGE
                )
            if not isinstance(label, int):
                if not isinstance(label, Tensor):
                    raise TypeError(
                        f"Illegal type of input label: {type(label)}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )
                if label.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
                    raise TypeError(
                        f"Illegal dtype of input label tensor: {label.dtype}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )
                if label.ndim != 0:
                    raise ValueError(
                        f"Illegal shape for input label tensor: {label.shape}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )

        return [(image, int(label)) for (image, label) in input_data]

    def _check_dataset_size_fits_sampler_parameters(self):
        """
        Check that the dataset size is compatible with the sampler parameters
        """
        self._check_dataset_has_enough_labels()
        self._check_dataset_has_enough_items_per_label()

    def _check_dataset_has_enough_labels(self):
        if self.n_way > len(self.items_per_label):
            raise ValueError(
                f"The number of labels in the dataset ({len(self.items_per_label)} "
                f"must be greater or equal to n_way ({self.n_way})."
            )

    def _check_dataset_has_enough_items_per_label(self):
        number_of_samples_per_label = [
            len(items_for_label) for items_for_label in self.items_per_label.values()
        ]
        minimum_number_of_samples_per_label = min(number_of_samples_per_label)
        label_with_minimum_number_of_samples = number_of_samples_per_label.index(
            minimum_number_of_samples_per_label
        )
        if self.n_shot + self.n_query > minimum_number_of_samples_per_label:
            raise ValueError(
                f"Label {label_with_minimum_number_of_samples} has only {minimum_number_of_samples_per_label} samples"
                f" but all classes must have at least n_shot + n_query ({self.n_shot + self.n_query}) samples."
            )
        
def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """

    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )


class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(self, backbone = None, use_softmax: bool = False, feature_centering = None, feature_normalization = None,):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method).
                If None is passed, the backbone will be initialized as nn.Identity().
            use_softmax: whether to return predictions as soft probabilities
            feature_centering: a features vector on which to center all computed features.
                If None is passed, no centering is performed.
            feature_normalization: a value by which to normalize all computed features after centering.
                It is used as the p argument in torch.nn.functional.normalize().
                If None is passed, no normalization is performed.
        """
        super().__init__()

        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax

        self.prototypes = torch.tensor(())
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())

        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        self.feature_normalization = feature_normalization

    @abstractmethod
    def forward(self, query_images: Tensor,) -> Tensor:
        """
        Predict classification labels.
        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    def process_support_set(self, support_images: Tensor, support_labels: Tensor,):
        """
        Harness information from the support set, so that query labels can later be predicted using a forward call.
        The default behaviour shared by most few-shot classifiers is to compute prototypes and store the support set.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    @staticmethod
    def is_transductive() -> bool:
        raise NotImplementedError(
            "All few-shot algorithms must implement a is_transductive method."
        )

    def compute_features(self, images: Tensor) -> Tensor:
        """
        Compute features from images and perform centering and normalization.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension)
        """
        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(
                centered_features, p=self.feature_normalization, dim=1
            )
        return centered_features

    def softmax_if_specified(self, output: Tensor, temperature: float = 1.0) -> Tensor:
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method of shape (n_query, n_classes)
            temperature: temperature of the softmax
        Returns:
            output as it was, or output as soft probabilities, of shape (n_query, n_classes)
        """
        return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return (
            nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(self, support_images: Tensor, support_labels: Tensor):
        """
        Extract support features, compute prototypes, and store support labels, features, and prototypes.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.support_labels = support_labels
        self.support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(self.support_features)
        self.prototypes = compute_prototypes(self.support_features, support_labels)

    @staticmethod
    def _raise_error_if_features_are_multi_dimensional(features: Tensor):
        if len(features.shape) != 2:
            raise ValueError(
                "Illegal backbone or feature shape. "
                "Expected output for an image is a 1-dim tensor."
            )


class PrototypicalNetworks(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.
        """
        # Extract the features of query images
        query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)

        # Compute the euclidean distance from queries to prototypes
        scores = self.l2_distance_to_prototypes(query_features)

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
    

# expanded to calculate precision and recall
def evaluate_on_one_task(model, support_images: Tensor, support_labels: Tensor, query_images: Tensor, query_labels: Tensor) -> Tuple[int, int, dict, dict, dict]:
    """
    Returns the number of correct predictions of query labels, the total number of
    predictions, and per-class true positives, false positives, and false negatives.
    """
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    pred_labels = torch.max(predictions, 1)[1]

    correct_predictions = int((pred_labels == query_labels).sum().item())
    total_predictions = len(query_labels)

    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for true_label, pred_label in zip(query_labels, pred_labels):
        if true_label == pred_label:
            true_positives[true_label.item()] += 1
        else:
            false_positives[pred_label.item()] += 1
            false_negatives[true_label.item()] += 1

    return correct_predictions, total_predictions, true_positives, false_positives, false_negatives


def evaluate(model, data_loader: DataLoader, device: str = "cuda", use_tqdm: bool = True, tqdm_prefix=None) -> Tuple[float, float, float]:
    """
    Evaluate the model on few-shot classification tasks.
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks
        device: where to cast data tensors. Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy, macro-averaged precision, macro-averaged recall
    """
    total_predictions = 0
    correct_predictions = 0
    all_true_positives = defaultdict(int)
    all_false_positives = defaultdict(int)
    all_false_negatives = defaultdict(int)

    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(data_loader), total=len(data_loader), disable=not use_tqdm, desc=tqdm_prefix) as tqdm_eval:
            for _, (support_images, support_labels, query_images, query_labels, _) in tqdm_eval:
                correct, total, true_positives, false_positives, false_negatives = evaluate_on_one_task(
                    model, 
                    support_images.to(device), 
                    support_labels.to(device), 
                    query_images.to(device), 
                    query_labels.to(device)
                )

                total_predictions += total
                correct_predictions += correct

                for label in true_positives.keys():
                    all_true_positives[label] += true_positives[label]
                for label in false_positives.keys():
                    all_false_positives[label] += false_positives[label]
                for label in false_negatives.keys():
                    all_false_negatives[label] += false_negatives[label]

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    # Calculate precision and recall for each class
    precision_per_class = {}
    recall_per_class = {}
    for label in all_true_positives.keys():
        tp = all_true_positives[label]
        fp = all_false_positives[label]
        fn = all_false_negatives[label]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_per_class[label] = precision
        recall_per_class[label] = recall

    # Calculate macro-averaged precision and recall
    macro_precision = sum(precision_per_class.values()) / len(precision_per_class)
    macro_recall = sum(recall_per_class.values()) / len(recall_per_class)
    accuracy = correct_predictions / total_predictions

    return accuracy, macro_precision, macro_recall