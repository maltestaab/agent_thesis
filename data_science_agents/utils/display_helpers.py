"""
Utilities for displaying analysis results
"""
import os
import re
import streamlit as st

def display_output_with_inline_images(text_output: str, created_images: list):
    """
    Display text output with images injected inline where ![IMAGE: filename] is mentioned.
    Sole purpose is to display images inline in Streamlit to make it more readable and user-friendly.
    """
    if not created_images:
        st.markdown(text_output)
        return
    
    # Create filename lookup
    filename_to_path = {}
    for img_path in created_images:
        filename = os.path.basename(img_path)
        filename_to_path[filename] = img_path
    
    # Simple pattern: ![IMAGE: filename.png]
    pattern = r'\!\[IMAGE:\s*([^\]]+)\]'
    
    # Split text and process
    parts = re.split(pattern, text_output)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular text
            if part.strip():
                st.markdown(part)
        else:
            # Image reference
            filename = part.strip()
            if filename in filename_to_path:
                image_path = filename_to_path[filename]
                if os.path.exists(image_path):
                    st.image(image_path, caption=filename, use_column_width=True)