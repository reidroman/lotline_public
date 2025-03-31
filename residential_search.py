import os
import streamlit as st
from mistralai import Mistral
from supabase import create_client, Client

# Set page title and description
st.title("Residential Category Search")
st.write("Search for residential categories using semantic embeddings")

# Set up Mistral client
@st.cache_resource
def get_mistral_client():
    api_key = os.environ["MISTRAL_API_KEY"]
    return Mistral(api_key=api_key)

# Set up Supabase client
@st.cache_resource
def get_supabase_client():
    url: str = os.environ.get("SUPABASE_CLIENT_URL")
    key: str = os.environ.get("SUPABASE_SECRET_SERVICE_ROLE_KEY")
    return create_client(url, key)

# Get clients
mistral_client = get_mistral_client()
supabase_client = get_supabase_client()

# User input for search query
search_query = st.text_input("Enter your search query:", "concrete or external groundwork")

# Slider for match threshold
match_threshold = st.slider("Match threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# Number input for match count
match_count = st.number_input("Number of results to return", min_value=1, max_value=50, value=10)

# Search button
if st.button("Search") or search_query:
    with st.spinner("Generating embeddings and searching..."):
        # Generate embeddings
        model = "mistral-embed"
        embeddings_response = [
            mistral_client.embeddings.create(model=model, inputs=search_query)
        ]
        embedding = embeddings_response[0].data[0].embedding
        
        # Fetch data from the residential_category_reference table
        response = supabase_client.rpc(
            "match_categories",
            {
                "query_embedding": embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
            },
        ).execute()
        
        # Display results
        st.subheader("Search Results")
        if response.data:
            for idx, item in enumerate(response.data):
                with st.expander(f"{idx+1}. {item.get('asset', 'Unknown')} - Similarity: {item.get('similarity', 0):.4f}"):
                    st.json(item)
        else:
            st.info("No matching results found. Try adjusting the match threshold.")

# Add some explanatory text at the bottom
st.markdown("""
### How it works
This app uses Mistral AI to generate embeddings for your search query and then 
uses Supabase vector search to find semantically similar categories.

- Higher threshold = more strict matching
- Lower threshold = more results, but possibly less relevant
""")
