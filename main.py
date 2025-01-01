import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import logging
logging.getLogger('absl').setLevel(logging.ERROR)  # Suppress absl logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress tensorflow logging
import tempfile
from typing import Dict, List, Any

from PIL import Image
import streamlit as st
from pinecone import Pinecone
from google.generativeai import configure, GenerativeModel

from pipeline import CXRImageRetrieval


class VectorDBManager:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
    
    def retrieve_similar_images(self, image_embedding: List[float], threshold: float = 0.95) -> Dict[str, List]:
        query_results = self.index.query(
            vector=image_embedding,
            top_k=10,
            include_metadata=True
        )
        
        filtered_matches = [match for match in query_results['matches'] if match.score > threshold]
        
        return {
            'documents': [match.metadata['impression'] for match in filtered_matches],
            'distances': [match.score for match in filtered_matches],
            'metadatas': [match.metadata for match in filtered_matches],
            'ids': [match.id for match in filtered_matches]
        }
    
    def store_manual_impression(self, image_embedding: List[float], impression: str, original_metadata: Dict = None):
        import uuid
        manual_id = str(uuid.uuid4())
        
        metadata = {
            'impression': impression,
            'type': 'manual'
        }
        if original_metadata:
            metadata.update(original_metadata)
            metadata.update({
                'impression': impression,
                'type': 'manual'
            })
        print(metadata)

        self.index.upsert(vectors=[{
            'id': manual_id,
            'values': image_embedding,
            'metadata': metadata
        }])


class ReportGenerator:
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model = GenerativeModel('gemini-1.5-pro')
        
    def _prepare_image(self, image_path: str, max_size: tuple = (800, 800)) -> Image:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
    
    def generate_report(self, context: List[str], image_path: str) -> str:
        img = self._prepare_image(image_path)
        
        prompt = """You are an assistant designed to write impression summaries for the radiology report.
You will be provided with a chest X-ray image and similar reports for context.

Instructions:
• Write an impression based on both the provided image and context information.
• The impression should not mention anything about follow-up actions.
• Impression should not contain any mentions of prior or previous studies.
• Use bullet points and never repeat findings.
• If you have multiple context reports, only mention the overlap findings that are also visible in the provided image.
• The impressions are already sorted by relevance to the image.

Context reports: {context}

Impression summary:"""

        response = self.model.generate_content([prompt.format(context=context), img])
        return response.text

class CXRImpressionApp:
    def __init__(self):
        if 'image_retrieval' not in st.session_state:
            st.session_state.image_retrieval = CXRImageRetrieval(
                pinecone_api_key=st.secrets["pinecone_api_key"],
                hf_token=st.secrets["hf_token"]
            )
        self.vector_db = VectorDBManager(st.secrets["pinecone_api_key"], "cxr-embeddings")
        self.report_generator = ReportGenerator(st.secrets["google_api_key"])
        self.image_retrieval = st.session_state.image_retrieval
        # Add new state variables
        if 'current_file_index' not in st.session_state:
            st.session_state.current_file_index = 0
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'accuracy_stats' not in st.session_state:
            st.session_state.accuracy_stats = {'correct': 0, 'total': 0}
        if 'processing_started' not in st.session_state:
            st.session_state.processing_started = False
    
    def run(self):
        st.title("CXR Impression Generator")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload chest X-ray images",
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Store uploaded files in session state when new files are uploaded
        if uploaded_files and uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.session_state.current_file_index = 0
            st.session_state.accuracy_stats = {'correct': 0, 'total': 0}
            st.session_state.processing_started = False  # Reset processing state
        
        # Add start processing button
        if st.session_state.uploaded_files:
            st.write(f"Number of files uploaded: {len(st.session_state.uploaded_files)}")
            if not st.session_state.processing_started:
                if st.button("Start Processing" if st.session_state.current_file_index == 0 else "Start New Batch"):
                    st.session_state.processing_started = True
                    st.session_state.current_file_index = 0  # Reset index when starting new batch
                    st.rerun()
            
            # Only show processing if started
            if st.session_state.processing_started:
                self._process_current_image()
    
    def _process_current_image(self):
        # Add state variable for editing mode
        if 'editing_mode' not in st.session_state:
            st.session_state.editing_mode = False
        
        # Show progress
        st.write(f"Processing image {st.session_state.current_file_index + 1} of {len(st.session_state.uploaded_files)}")
        
        current_file = st.session_state.uploaded_files[st.session_state.current_file_index]
        uploaded_img = Image.open(current_file)
        st.image(uploaded_img, width=500)

        # Store results in session state to avoid recomputation
        if 'current_embedding' not in st.session_state:
            st.session_state.current_embedding = self.image_retrieval.generate_embedding(uploaded_img)
        if 'current_query_results' not in st.session_state:
            st.session_state.current_query_results = self.vector_db.retrieve_similar_images(st.session_state.current_embedding)
        
        # Use stored results
        query_results = st.session_state.current_query_results
        
        # Add logging for query results
        if query_results['documents']:  # Only show if there are matches
            st.write("Similar cases found:")
            for i, (doc, score) in enumerate(zip(query_results['documents'], query_results['distances'])):
                st.write(f"Match {i+1} (similarity: {score:.3f}):")
                st.write(doc)
                st.write("---")

        # Only generate report if not already generated
        if 'current_report' not in st.session_state:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, current_file.name)
            with open(path, "wb") as f:
                f.write(current_file.getvalue())
            st.session_state.current_report = self.report_generator.generate_report(query_results['documents'], path)
        
        st.write(st.session_state.current_report)
        
        # Only show buttons if not in editing mode
        if not st.session_state.editing_mode:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct", key=f"thumbs_up_{st.session_state.current_file_index}"):
                    st.session_state.accuracy_stats['correct'] += 1
                    st.session_state.accuracy_stats['total'] += 1
                    self._clear_current_state()
                    self._next_image()
            
            with col2:
                if st.button("👎 Incorrect", key=f"thumbs_down_{st.session_state.current_file_index}"):
                    st.session_state.accuracy_stats['total'] += 1
                    st.session_state.editing_mode = True
                    st.rerun()
        else:
            # Show editing interface
            self._handle_manual_impression()

        # Show accuracy statistics
        if st.session_state.accuracy_stats['total'] > 0:
            accuracy = (st.session_state.accuracy_stats['correct'] / st.session_state.accuracy_stats['total']) * 100
            st.write(f"Current Accuracy: {accuracy:.2f}% ({st.session_state.accuracy_stats['correct']}/{st.session_state.accuracy_stats['total']})")

    def _clear_current_state(self):
        # Clear temporary state variables
        if 'current_embedding' in st.session_state:
            del st.session_state.current_embedding
        if 'current_query_results' in st.session_state:
            del st.session_state.current_query_results
        if 'current_report' in st.session_state:
            del st.session_state.current_report
        if 'editing_mode' in st.session_state:
            del st.session_state.editing_mode

    def _handle_manual_impression(self):
        edited_impression = st.text_area("Edit Impression:", value=st.session_state.current_report)
        if st.button("Save Manual Impression"):
            self.vector_db.store_manual_impression(
                st.session_state.current_embedding,
                edited_impression,
                {'previous_impression': st.session_state.current_report}
            )
            st.success("Manual impression saved successfully!")
            self._clear_current_state()
            self._next_image()

    def _next_image(self):
        if st.session_state.current_file_index < len(st.session_state.uploaded_files) - 1:
            st.session_state.current_file_index += 1
            st.rerun()
        else:
            st.session_state.processing_started = False  # Reset processing state
            st.success("All images processed!")

if __name__ == '__main__':
    app = CXRImpressionApp()
    app.run()
