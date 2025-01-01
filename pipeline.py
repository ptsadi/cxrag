import os
import logging
import warnings
import numpy as np
import pandas as pd
from huggingface_hub import login
from huggingface_hub.utils import HfFolder
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import snapshot_download
from datasets import load_dataset
from PIL import Image
import tensorflow as tf
import io
import tensorflow_text
from tqdm import tqdm
import threading

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging


class CXRImageRetrieval:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, pinecone_api_key=None, hf_token=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CXRImageRetrieval, cls).__new__(cls)
                cls._instance._is_initialized = False
                cls._instance._pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
                cls._instance._hf_token = hf_token or os.getenv("HF_TOKEN")
        return cls._instance

    def __init__(self, pinecone_api_key=None, hf_token=None):
        with self._lock:
            if self._is_initialized:
                return
            
            if not self._pinecone_api_key:
                raise ValueError("Pinecone API key is required")
            if not self._hf_token:
                raise ValueError("HuggingFace token is required")
            
            # Login to HuggingFace
            login(token=self._hf_token)
            if HfFolder.get_token() is None:
                raise ValueError("Failed to set HuggingFace token")

            self._initialize()
            self._is_initialized = True

    def _initialize(self):
        """Initialize all the components only once"""
        self.index_name = "cxr-embeddings"
        self.hf_repo_id = "google/cxr-foundation"

        # Initialize Pinecone with the provided API key
        pc = Pinecone(api_key=self._pinecone_api_key)

        try:
            self.index = pc.Index(self.index_name)
        except Exception as e:
            pc.create_index(
                name=self.index_name,
                dimension=4096,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
                metric="cosine"
            )
            self.index = pc.Index(self.index_name)

        # Load models only once
        model_dir = os.path.join(os.getcwd(), 'model_files')
        os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(os.path.join(model_dir, 'elixr-c-v2-pooled')):
            snapshot_download(
                repo_id="google/cxr-foundation",
                local_dir=model_dir,
                allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

        tf.compat.v1.enable_resource_variables()

        try:
            _ = tensorflow_text.SentencepieceTokenizer
            self.elixrc_model = tf.saved_model.load(os.path.join(model_dir, 'elixr-c-v2-pooled'))
            self.elixrc_infer = self.elixrc_model.signatures['serving_default']
            self.qformer_model = tf.saved_model.load(os.path.join(model_dir, 'pax-elixr-b-text'))
        except Exception as e:
            raise Exception(f"Failed to load models: {str(e)}. Make sure tensorflow-text is installed.")

    def generate_embedding(self, image):
        """Generate embedding for a single image using ELIXR models."""
        # Convert image to TF example
        serialized_img_tf_example = self.png_to_tfexample(np.array(image)).SerializeToString()

        # Step 1: Generate ELIXR-C embedding
        elixrc_output = self.elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
        elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

        # Step 2: Generate ELIXR-B embedding using QFormer
        qformer_input = {
            'image_feature': elixrc_embedding.tolist(),
            'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
            'paddings': np.zeros((1, 1, 128), dtype=np.float32).tolist(),
        }

        qformer_output = self.qformer_model.signatures['serving_default'](**qformer_input)
        embedding = qformer_output['all_contrastive_img_emb']
        embedding_list = embedding.numpy().flatten().tolist()
        return embedding_list

    def load_and_process_dataset(self):
        # Load MIMIC-CXR dataset
        dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset")

        vectors_to_upsert = []
        batch_size = 1

        # Get existing vector IDs from Pinecone in batches
        try:
            stats = self.index.describe_index_stats()
            existing_count = stats.total_vector_count
            if existing_count > 0:
                print(f"Found {existing_count} existing vectors in the index")
                # Fetch IDs in batches of 1000
                existing_ids = set()
                batch_size = 1000
                for i in range(0, existing_count, batch_size):
                    batch_ids = [str(j) for j in range(i, min(i + batch_size, existing_count))]
                    batch_vectors = self.index.fetch(ids=batch_ids)
                    existing_ids.update(batch_vectors['vectors'].keys())
            else:
                existing_ids = set()
        except Exception as e:
            print(f"Error checking existing vectors: {str(e)}")
            existing_ids = set()

        # Add tqdm progress bar
        total_items = len(dataset["train"])

        for i, item in tqdm(enumerate(dataset["train"]), total=total_items, desc="Processing images"):
            # Skip if vector ID already exists
            if str(i) in existing_ids:
                continue

            # Process image
            image = item["image"]
            findings = item["findings"]
            impression = item["impression"]

            # Generate image embedding
            embedding_list = self.generate_embedding(image)

            # Prepare vector for Pinecone
            metadata = {
                "findings": findings,
                "impression": impression
            }

            vector = {
                "id": str(i),
                "values": embedding_list,
                "metadata": metadata
            }
            vectors_to_upsert.append(vector)

            # Batch upsert
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

        # Upsert any remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)

    def store_in_pinecone(self, embeddings_df: pd.DataFrame):
        """Store embeddings in Pinecone for faster retrieval."""
        try:
            # Try to delete existing vectors
            self.index.delete(deleteAll=True)
        except Exception:
            print(f"No existing vectors to delete")

        # Store embeddings in batches
        batch_size = 100
        try:
            for i in range(0, len(embeddings_df), batch_size):
                batch = embeddings_df.iloc[i:i + batch_size]
                vectors = [(
                    str(id),  # Pinecone requires string IDs
                    emb.tolist(),
                    {"type": "cxr_image"}
                ) for id, emb in zip(batch['image_id'], batch['embeddings'])]

                self.index.upsert(vectors=vectors)
            print(f"Successfully stored {len(embeddings_df)} embeddings in Pinecone")
        except Exception as e:
            raise Exception(f"Failed to store embeddings in Pinecone: {str(e)}")

    def png_to_tfexample(self, image_array: np.ndarray) -> tf.train.Example:
        """Creates a tf.train.Example from a NumPy array."""
        # Convert RGB to grayscale if needed
        if image_array.ndim == 3:
            # Using standard RGB to grayscale conversion weights
            image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])

        # Convert the image to float32 and shift the minimum value to zero
        image = image_array.astype(np.float32)
        image -= image.min()

        if image_array.dtype == np.uint8:
            # For uint8 images, no rescaling is needed
            pixel_array = image.astype(np.uint8)
            mode = 'L'  # 8-bit pixels, grayscale
        else:
            # For other data types, scale image to use the full 16-bit range
            max_val = image.max()
            if max_val > 0:
                image *= 65535 / max_val  # Scale to 16-bit range
            pixel_array = image.astype(np.uint16)
            mode = 'I;16'  # 16-bit unsigned integer pixels

        # Ensure the array is 2-D (grayscale image)
        if pixel_array.ndim != 2:
            raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

        # Convert numpy array to PIL Image and encode as PNG
        pil_image = Image.fromarray(pixel_array, mode=mode)
        output = io.BytesIO()
        pil_image.save(output, format='PNG')
        png_bytes = output.getvalue()

        # Create a tf.train.Example and assign the features
        example = tf.train.Example()
        features = example.features.feature
        features['image/encoded'].bytes_list.value.append(png_bytes)
        features['image/format'].bytes_list.value.append(b'png')

        return example


def main():
    # Example of using the class with explicit API keys
    retrieval_system = CXRImageRetrieval()
    retrieval_system.load_and_process_dataset()


if __name__ == '__main__':
    main()
