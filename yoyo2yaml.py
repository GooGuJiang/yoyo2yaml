import os
import json
from sklearn.model_selection import train_test_split
import shutil
import yaml

def create_dataset(notes_path, images_path, labels_path, classes_path, output_path, val_size=0.2, test_size=0.1):
    # Load notes.json
    with open(notes_path, 'r') as f:
        notes = json.load(f)
    
    # Load classes.txt
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Create a mapping from class name to class id
    class_to_id = {c['name']: c['id'] for c in notes['categories']}
    
    # Get a list of all image files
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    
    # Split the data into train/val/test sets
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size/(1-test_size), random_state=42)
    
    # Create the output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'test'), exist_ok=True)
    
    # Copy the images and labels to the output directories
    for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for file in files:
            # Copy the image file
            src_image_file = os.path.join(images_path, file)
            dst_image_file = os.path.join(output_path, 'images', split, file)
            shutil.copy(src_image_file, dst_image_file)
            
            # Copy the label file
            src_label_file = os.path.join(labels_path, file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
            dst_label_file = os.path.join(output_path, 'labels', split, file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
            shutil.copy(src_label_file, dst_label_file)
    
    # Create the data.yaml file
    data = {
        'train': '../images/train',
        'val': '../images/val',
        'test': '../images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        yaml.dump(data, f)

# Example usage:
create_dataset(
    notes_path='./data/notes.json',
    images_path='./data/images',
    labels_path='./data/labels',
    classes_path='./data/classes.txt',
    output_path='./output',
    val_size=0.2,
    test_size=0.1
)
