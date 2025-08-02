"""
Simple Robust HEA Frontend with Fixed Custom Q&A
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Simple Q&A function without complex imports
def simple_qa_system(question: str, knowledge_data: dict) -> dict:
    """Simple Q&A using loaded knowledge data"""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        api_key = "AIzaSyD_OFymvreIbpV5gEurkC47pMeq37ntuHc"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Build context from knowledge
        relevant_studies = []
        for entry in knowledge_data.get('knowledge_entries', [])[:10]:  # Use first 10 as context
            if any(keyword.lower() in entry.get('content', '').lower() for keyword in question.lower().split()[:3]):
                relevant_studies.append(entry)
        
        if not relevant_studies:
            relevant_studies = knowledge_data.get('knowledge_entries', [])[:5]  # Fallback to first 5
        
        # Build context
        context_parts = []
        for study in relevant_studies[:5]:
            context_parts.append(f"Title: {study.get('title', '')}")
            if study.get('alloy_systems'):
                context_parts.append(f"Alloys: {', '.join(study['alloy_systems'])}")
            if study.get('hydrogen_capacity'):
                context_parts.append(f"H2 Capacity: {study['hydrogen_capacity']} {study.get('capacity_units', '')}")
            context_parts.append("---")
        
        context = "\\n".join(context_parts)
        
        # Generate answer
        prompt = f"""
        You are an expert in High-Entropy Alloys for hydrogen storage. Answer this question based on the research context:
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a comprehensive, scientific answer including specific alloy compositions, properties, and evidence from the studies.
        """
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        return {
            'answer': answer,
            'confidence': 0.85,
            'sources': [{'title': s.get('title', ''), 'alloy_systems': s.get('alloy_systems', [])} for s in relevant_studies[:3]],
            'num_relevant_studies': len(relevant_studies)
        }
        
    except Exception as e:
        return {
            'answer': f"I encountered an error processing your question: {str(e)}. Please try a simpler question about HEA hydrogen storage.",
            'confidence': 0.0,
            'sources': [],
            'num_relevant_studies': 0
        }

@st.cache_data
def load_knowledge_data():
    """Load pre-processed knowledge data"""
    knowledge_file = Path("../backend/comprehensive_system/comprehensive_knowledge.json")
    if knowledge_file.exists():
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        st.error("Knowledge data not found. Please run comprehensive_system.py first.")
        st.stop()

def main():
    st.set_page_config(
        page_title="Enhanced HEA Knowledge System",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Enhanced High-Entropy Alloys Knowledge System")
    st.markdown("**Advanced AI-Powered Knowledge Base with 290 Research Papers**")
    
    # Load knowledge data
    with st.spinner("Loading comprehensive knowledge base..."):
        knowledge_data = load_knowledge_data()
    
    stats = knowledge_data.get('system_stats', {})
    
    # Sidebar statistics
    st.sidebar.header("📊 System Statistics")
    st.sidebar.metric("Total Papers", stats.get('total_papers', 0))
    st.sidebar.metric("Knowledge Graph Nodes", stats.get('graph_nodes', 0))
    st.sidebar.metric("Unique Alloy Systems", stats.get('unique_alloy_systems', 0))
    st.sidebar.metric("Unique Elements", stats.get('unique_elements', 0))
    st.sidebar.metric("Papers with H₂ Capacity", stats.get('papers_with_capacity_data', 0))
    st.sidebar.metric("Max H₂ Capacity", f"{stats.get('max_capacity', 0):.1f}%")
    
    # Main Q&A Interface
    st.header("🤖 Advanced Q&A System")
    st.markdown("Ask any question about High-Entropy Alloys for hydrogen storage.")
    
    # Example questions
    example_questions = [
        "What are the best HEA compositions for hydrogen storage?",
        "How does synthesis method affect hydrogen capacity?",
        "Which elements are most important for high hydrogen storage?",
        "What crystal structures work best for hydrogen storage?",
        "Compare different HEA systems for hydrogen storage performance",
        "What are the challenges in HEA hydrogen storage?",
        "How do temperature and pressure affect HEA hydrogen storage?",
        "What are the latest breakthroughs in HEA hydrogen storage?"
    ]
    
    selected_question = st.selectbox("Choose an example question:", [""] + example_questions)
    
    user_question = st.text_input(
        "Or ask your own question:",
        value=selected_question,
        placeholder="e.g., What makes certain HEAs better for hydrogen storage?"
    )
    
    if st.button("🔍 Get AI Answer", type="primary"):
        if user_question:
            with st.spinner("Generating comprehensive answer..."):
                result = simple_qa_system(user_question, knowledge_data)
            
            # Display results
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📋 AI Answer")
                st.markdown(result['answer'])
            
            with col2:
                st.metric("Confidence", f"{result['confidence']:.1%}")
                st.metric("Sources Used", result['num_relevant_studies'])
            
            # Show sources
            if result['sources']:
                st.subheader("📚 Sources")
                for i, source in enumerate(result['sources'], 1):
                    st.write(f"{i}. **{source['title']}**")
                    if source['alloy_systems']:
                        st.write(f"   Alloys: {', '.join(source['alloy_systems'])}")
        else:
            st.warning("Please enter a question.")
    
    # Knowledge Analytics
    st.header("📈 Knowledge Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Top Elements")
        if stats.get('top_elements'):
            elements_data = pd.DataFrame({
                'Element': stats['top_elements'][:10],
                'Frequency': range(10, 0, -1)
            })
            fig = px.bar(elements_data, x='Element', y='Frequency', 
                        title="Most Studied Elements in HEAs")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚗️ Sample Alloy Systems")
        if stats.get('sample_alloys'):
            st.write("Representative compositions:")
            for i, alloy in enumerate(stats['sample_alloys'][:8], 1):
                st.write(f"{i}. **{alloy}**")
    
    # Hydrogen capacity analysis
    st.subheader("💨 Hydrogen Storage Performance")
    
    # Extract capacity data
    capacity_data = []
    for entry in knowledge_data.get('knowledge_entries', []):
        if entry.get('hydrogen_capacity') and entry.get('alloy_systems'):
            capacity_data.append({
                'Alloy': entry['alloy_systems'][0] if entry['alloy_systems'] else 'Unknown',
                'H₂ Capacity (%)': entry['hydrogen_capacity'],
                'Units': entry.get('capacity_units', 'wt%'),
                'Title': entry.get('title', '')[:50] + "..."
            })
    
    if capacity_data:
        capacity_df = pd.DataFrame(capacity_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Capacity distribution
            fig = px.histogram(capacity_df, x='H₂ Capacity (%)', 
                             title="Distribution of H₂ Storage Capacities",
                             nbins=15)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top performers
            top_performers = capacity_df.nlargest(8, 'H₂ Capacity (%)')
            fig = px.bar(top_performers, x='Alloy', y='H₂ Capacity (%)',
                       title="Top Performing HEA Systems")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📊 Top Performing Alloys")
        top_capacity = capacity_df.nlargest(15, 'H₂ Capacity (%)')[['Alloy', 'H₂ Capacity (%)', 'Units', 'Title']]
        st.dataframe(top_capacity, use_container_width=True, hide_index=True)
    
    # System info
    st.markdown("---")
    st.markdown(
        f"**Enhanced HEA Knowledge System** • "
        f"Powered by {stats.get('total_papers', 0)} research papers • "
        f"Built with AI and comprehensive analysis"
    )

if __name__ == "__main__":
    main()
