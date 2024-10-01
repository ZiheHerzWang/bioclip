import os
import pandas as pd
import json

def create_class_mapping(metadata):
    class_names = metadata['class'].unique()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    return class_to_idx

def split_data(metadata, dataset_dir, class_to_idx):
    items = []
    for _, row in metadata.iterrows():
        img_path = os.path.join(dataset_dir, "output", row['filepath'])
        label = class_to_idx[row['class']]
        items.append({"img_path": img_path, "label": label})
    class_image_dict = {label: [] for label in class_to_idx.values()}
    for item in items:
        class_image_dict[item['label']].append(item)

    train, val, test = [], [], []
    for label, images in class_image_dict.items():
        if len(images) > 1:
            train_images = images[:1]
            remaining_images = images[1:]
        else:
            train_images = images
            remaining_images = []

        half = len(remaining_images) 
        val_images = remaining_images[:half]
        test_images = remaining_images[half:]

        train.extend(train_images)
        val.extend(val_images)
        test.extend(test_images)

    return train, val, test

def save_split_to_json(train, val, test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train, f, indent=4)
        
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val, f, indent=4)
        
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test, f, indent=4)
    
    print(f"Data splits saved to {output_dir}")

def main(dataset_dir, metadata_file, output_dir):
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    class_to_idx = create_class_mapping(metadata)
    train, val, test = split_data(metadata, dataset_dir, class_to_idx)
    save_split_to_json(train, val, test, output_dir)

if __name__ == "__main__":
    dataset_dir = "/home/wang.14629/CoCoOp_BioCLIP/test_image/own_cocoop/DATA/Insect"  
    metadata_file = os.path.join(dataset_dir, "metadata.csv")
    output_dir = "/home/wang.14629/CoCoOp_BioCLIP/test_image/own_cocoop/data_load"  

    main(dataset_dir, metadata_file, output_dir)
