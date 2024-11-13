import uuid
import json
import sys, getopt
from pymilvus import MilvusClient, DataType, Collection
import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

CLUSTER_ENDPOINT="http://localhost:27017"
client = MilvusClient(uri=CLUSTER_ENDPOINT)
query_image = "tiger.jpeg"
client.load_collection("ceph_bkt_0cfd8f98_74a7_4227_8d07_cdd604a908e2")
extractor = FeatureExtractor("resnet34")
res = client.search(
    collection_name="ceph_bkt_0cfd8f98_74a7_4227_8d07_cdd604a908e2",  # target collection
    data=[extractor(query_image)],  # query vectors
    limit=2,  # number of returned entities
    output_fields=["url"],  # specifies fields to be returned
    consistency_level="Strong" ## NOTE: without defining that, the search might return empty result.
)
print(res)
