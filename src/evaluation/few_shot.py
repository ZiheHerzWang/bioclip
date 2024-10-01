import datetime
import logging
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import json
import pickle
import numpy as np
import random
import shutil
from numpy import linalg as LA
from scipy.stats import mode

from .data import DatasetFromFile
from .params import parse_args
from .utils import init_device, random_seed
from sklearn.model_selection import train_test_split

from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
    trace_model,
)
import open_clip
from ..training.logger import setup_logging
from ..training.precision import get_autocast


def load_bioclip_model():
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    return model, preprocess_train, preprocess_val


def save_pickle(base_path, data):
    os.makedirs(base_path, exist_ok=True)
    file = os.path.join(base_path, 'pickle.p')
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    return file


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_dataloader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
    )


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run(model, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if cast_dtype is not None:
        model = model.half()
    else:
        model = model.float()

    model = model.to(args.device)

    with torch.no_grad():
        feature_list = []
        target_list = []
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            target_list.append(target.numpy())
            images = images.to(args.device)

            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                feature_list.append(image_features.detach().cpu().numpy())

        file = save_pickle(args.log_path, [np.vstack(feature_list), np.hstack(target_list), dataloader.dataset.samples, dataloader.dataset.class_to_idx])

    return file


def few_shot_eval(model, data, args):
    results = {}

    logging.info("Starting few-shot.")

    for split in data:
        logging.info("Building few-shot %s classifier.", split)

        file = run(model, data[split], args)

        logging.info("Finished few-shot %s with total %d classes.", split, len(data[split].dataset.classes))

    logging.info("Finished few-shot.")

    return results, file


def linear_probe_5_shot(features, labels, c2i, k_shot=5):
    selected_features = []
    selected_labels = []

    for class_idx in np.unique(labels):
        class_indices = np.where(labels == class_idx)[0]

        if len(class_indices) >= k_shot:
            selected_class_indices = np.random.choice(class_indices, k_shot, replace=False)
        else:
            logging.warning(f"Class {class_idx} has fewer than {k_shot} samples, using all available samples.")
            selected_class_indices = class_indices

        selected_features.append(features[selected_class_indices])
        selected_labels.append(labels[selected_class_indices])

    selected_features = torch.cat([torch.tensor(x) for x in selected_features])
    selected_labels = torch.cat([torch.tensor(x) for x in selected_labels])

    return selected_features, selected_labels


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    random_seed(args.seed, 0)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = init_device(args)

    if args.save_logs and args.name is None:
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"p_{args.precision}",
                "few_shot",
            ]
        )

    if args.save_logs is None:
        args.log_path = None
    else:
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        args.log_path = os.path.join(log_base_path, "out.log")

    setup_logging(args.log_path, logging.INFO)

    model, preprocess_train, preprocess_val = load_bioclip_model()

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    data = {
        "val-unseen": get_dataloader(
            DatasetFromFile(args.data_root, args.label_filename, transform=preprocess_val),
            batch_size=args.batch_size, num_workers=args.workers
        ),
    }

    _, feature_file = few_shot_eval(model, data, args)

    logging.info("Starting Linear Probing.")

    feature, target, samples, c2i = load_pickle(feature_file)

    features = torch.tensor(feature, dtype=torch.float32)
    labels = torch.tensor(target, dtype=torch.long)

    train_features, train_labels = linear_probe_5_shot(features, labels, c2i, k_shot=5)

    linear_classifier = torch.nn.Linear(train_features.shape[1], len(c2i))
    linear_classifier = linear_classifier.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=0.01)

    epochs = 300
    for epoch in range(epochs):
        linear_classifier.train()
        optimizer.zero_grad()

        outputs = linear_classifier(train_features.to(args.device))
        loss = criterion(outputs, train_labels.to(args.device))

        loss.backward()
        optimizer.step()

        logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    logging.info("Finished Linear Probing.")
