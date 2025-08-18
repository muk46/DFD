# Utility functions used in preprocessing steps

import glob
import os
from torchvision.transforms import Resize, ToPILImage, ToTensor
import networkx as nx
from matplotlib import pyplot as plt

# Returns all the files paths with a specific extension inside a requested root directory
def get_paths(rootdir, list_file=None, ext="png"):
    video_ids = set()
    if list_file:
        with open(list_file, 'r') as f:
            for line in f:
                _, video_id = line.strip().split()
                video_id = video_id.replace(".mp4", "")
                video_ids.add(video_id)

    paths = []
    for video_folder in os.listdir(rootdir):
        if list_file and video_folder not in video_ids:
            continue
        folder_path = os.path.join(rootdir, video_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in glob.glob(os.path.join(folder_path, f'*.{ext}')):
            paths.append(file)

    return paths

# Cluster the images generating a graph of connected components
def _generate_connected_components(similarities, similarity_threshold=0.80):
    graph = nx.Graph()
    for i in range(len(similarities)):
        for j in range(len(similarities)):
            if i != j and similarities[i, j] > similarity_threshold:
                graph.add_edge(i, j)

    components_list = []
    for component in nx.connected_components(graph):
        components_list.append(list(component))
    graph.clear()
    graph = None

    return components_list

# Method used to preprocess the image before features extraction in clustering step
def preprocess_images(img, shape=[128, 128]):
    img = Resize(shape)(img)
    return img

def check_correct(y_pred, labels):
    # This is a placeholder function. You need to implement this based on your project's logic.
    # It seems to check the correctness of predictions.
    corrects = (y_pred.round() == labels).sum().item()
    positive_class = (y_pred.round() == 1).sum().item()
    negative_class = (y_pred.round() == 0).sum().item()
    return corrects, positive_class, negative_class

def unix_time_millis(dt):
    # This is a placeholder function. You need to implement this based on your project's logic.
    return int(dt.total_seconds() * 1000)

def slowfast_input_transform(videos):
    # This is a placeholder function. You need to implement this based on your project's logic.
    return videos