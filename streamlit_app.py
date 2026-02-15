"""
FinanceRAG - Streamlit UI
Interactive interface for the Financial Document RAG System
"""

import streamlit as st
import requests
import os
from typing import Optional
import pandas as pd
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="FinanceRAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    .source-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f9f9f9;
        border-left: 3px solid #2ca02c;
        margin: 0.5rem 0;
    }
    .error-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffebee;
        border-left: 3px solid #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.session_state.api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_url,
        help="URL of the FinanceRAG FastAPI backend"
    )
    
    # Test connection
    if st.button("🔗 Test Connection", use_container_width=True):
        try:
            response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
            if response.status_code == 200:
                status = response.json()
                st.success(f"✅ Connected! Status: {status['status']}")
                st.info(f"Vector Store: {status.get('vector_store', 'unknown')}")
            else:
                st.error(f"❌ Connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
    
    st.divider()
    st.markdown("### 📝 Info")
    st.markdown("""
    **FinanceRAG** is a Retrieval-Augmented Generation system 
    for financial document analysis.
    
    - Upload financial documents (PDF, DOCX, TXT)
    - Query with natural language
    - Get cited answers powered by Claude
    """)

# Main content
st.title("📊 FinanceRAG - Financial Document Analysis")
st.markdown("Extract insights from financial documents using AI-powered retrieval")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "📤 Upload Documents", "📊 Metrics", "📚 History"])

# TAB 1: QUERY
with tab1:
    st.header("Query Financial Documents")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Your Financial Question",
            placeholder="e.g., What was the total revenue in Q3 2023? What are the main risk factors?",
            height=100
        )
    
    with col2:
        top_k = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of document chunks to use for the answer"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        submit_button = st.button("🚀 Submit Query", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Process query
    if submit_button and query.strip():
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/api/query",
                    json={"query": query, "top_k": top_k},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "top_k": top_k
                    })
                    
                    # Display answer
                    st.success("✅ Answer Generated")
                    
                    st.markdown("### 💡 Answer")
                    st.markdown(f"> {result['answer']}")
                    
                    # Display sources
                    if result.get('sources'):
                        st.markdown("### 📖 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i} - Relevance: {source.get('relevance_score', 'N/A'):.2f}" if 'relevance_score' in source else f"Source {i}"):
                                st.markdown(source.get('text', 'No text available'))
                                
                                # Show metadata
                                metadata = source.get('metadata', {})
                                if metadata:
                                    st.markdown("**Metadata:**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Filename:** {metadata.get('filename', 'N/A')}")
                                    with col2:
                                        st.write(f"**Chunk ID:** {metadata.get('chunk_id', 'N/A')}")
                    
                    # Display metrics
                    if result.get('metrics'):
                        st.markdown("### ⏱️ Performance Metrics")
                        metrics_cols = st.columns(len(result['metrics']))
                        
                        for col, (key, value) in zip(metrics_cols, result['metrics'].items()):
                            with col:
                                st.metric(
                                    key.replace('_', ' ').title(),
                                    f"{value:.2f}" if isinstance(value, float) else value
                                )
                
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.error(response.json().get('detail', 'Unknown error'))
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. The API took too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection error. Please check if the FastAPI server is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif submit_button:
        st.warning("Please enter a query")

# TAB 2: UPLOAD DOCUMENTS
with tab2:
    st.header("Upload Financial Documents")
    st.markdown("Supported formats: **PDF, DOCX, TXT**")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or multiple financial documents"
    )
    
    if uploaded_files:
        st.subheader(f"📋 Selected Files ({len(uploaded_files)})")
        
        # Display file info
        files_data = []
        for file in uploaded_files:
            files_data.append({
                "Filename": file.name,
                "Size (KB)": f"{file.size / 1024:.2f}",
                "Type": file.name.split('.')[-1].upper()
            })
        
        st.dataframe(pd.DataFrame(files_data), use_container_width=True)
        
        if st.button("⬆️ Upload & Process", use_container_width=True):
            progress_bar = st.progress(0)
            status_container = st.container()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                with status_container:
                    st.info(f"Processing: {uploaded_file.name}...")
                
                try:
                    # Upload file
                    files = {'file': (uploaded_file.name, uploaded_file.getbuffer())}
                    
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        response = requests.post(
                            f"{st.session_state.api_url}/api/upload",
                            files=files,
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        with status_container:
                            st.success(f"✅ {uploaded_file.name}")
                            st.success(f"Created {result.get('chunks_created', 0)} chunks")
                        
                        st.session_state.uploaded_files.append({
                            "filename": uploaded_file.name,
                            "timestamp": datetime.now().isoformat(),
                            "chunks": result.get('chunks_created', 0),
                            "status": "success"
                        })
                    else:
                        with status_container:
                            st.error(f"❌ Failed to upload {uploaded_file.name}")
                        
                        st.session_state.uploaded_files.append({
                            "filename": uploaded_file.name,
                            "timestamp": datetime.now().isoformat(),
                            "chunks": 0,
                            "status": "failed"
                        })
                
                except Exception as e:
                    with status_container:
                        st.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
                
                # Update progress bar
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.balloons()

# TAB 3: METRICS
with tab3:
    st.header("System Metrics & Analytics")
    
    col1, col2 = st.columns(2)
    
    # Fetch metrics
    try:
        response = requests.get(
            f"{st.session_state.api_url}/api/metrics",
            timeout=10
        )
        
        if response.status_code == 200:
            metrics = response.json()
            
            with col1:
                st.metric(
                    "📄 Total Documents",
                    metrics.get('total_documents', 0),
                    help="Number of documents in the system"
                )
                st.metric(
                    "⏱️ Avg Latency (ms)",
                    f"{metrics.get('avg_latency_ms', 0):.0f}",
                    help="Average query response time"
                )
            
            with col2:
                st.metric(
                    "🔍 Total Queries",
                    metrics.get('total_queries', 0),
                    help="Number of queries processed"
                )
                st.metric(
                    "⭐ Avg Relevance",
                    f"{metrics.get('avg_relevance', 0):.2f}",
                    help="Average relevance score of retrieved documents"
                )
            
            # System Health
            st.markdown("### 🏥 System Health")
            health_response = requests.get(
                f"{st.session_state.api_url}/health",
                timeout=5
            )
            
            if health_response.status_code == 200:
                health = health_response.json()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.success(f"Status: {health.get('status', 'unknown').upper()}")
                with col2:
                    st.info(f"Version: {health.get('version', 'unknown')}")
                with col3:
                    vs_status = health.get('vector_store', 'unknown').upper()
                    if vs_status == 'CONNECTED':
                        st.success(f"Vector Store: {vs_status}")
                    else:
                        st.error(f"Vector Store: {vs_status}")
        else:
            st.error("Unable to fetch metrics")
    
    except Exception as e:
        st.error(f"Error fetching metrics: {str(e)}")

# TAB 4: HISTORY
with tab4:
    st.header("Query & Upload History")
    
    tab4_1, tab4_2 = st.tabs(["Query History", "Upload History"])
    
    with tab4_1:
        if st.session_state.query_history:
            history_data = []
            for item in st.session_state.query_history:
                history_data.append({
                    "Query": item['query'][:50] + "..." if len(item['query']) > 50 else item['query'],
                    "Time": item['timestamp'],
                    "Sources": item['top_k']
                })
            
            st.dataframe(
                pd.DataFrame(history_data),
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()
        else:
            st.info("No queries yet. Start by submitting a query!")
    
    with tab4_2:
        if st.session_state.uploaded_files:
            upload_data = []
            for item in st.session_state.uploaded_files:
                upload_data.append({
                    "Filename": item['filename'],
                    "Time": item['timestamp'],
                    "Chunks": item['chunks'],
                    "Status": item['status']
                })
            
            st.dataframe(
                pd.DataFrame(upload_data),
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("🗑️ Clear Upload History"):
                st.session_state.uploaded_files = []
                st.rerun()
        else:
            st.info("No uploads yet. Start uploading documents!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>FinanceRAG © 2024 | Powered by FastAPI, ChromaDB, and Claude AI</p>
</div>
""", unsafe_allow_html=True)
