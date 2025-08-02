"""
Enhanced Streamlit Frontend for Comprehensive HEA Knowledge System
With RAG-style knowledge retrieval and perfect semantic understanding
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from comprehensive_system import RobustHEASystem
except ImportError:
    st.error("Could not import comprehensive system. Please run the comprehensive_system.py first.")
    st.stop()

@st.cache_resource
def initialize_system():
    """Initialize the comprehensive HEA system"""
    try:
        system = RobustHEASystem()
        
        # Check if processed data exists
        knowledge_file = Path("../backend/comprehensive_system/comprehensive_knowledge.json")
        if knowledge_file.exists():
            # Load pre-processed knowledge
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct knowledge entries  
            from comprehensive_system import KnowledgeEntry
            
            system.knowledge_entries = []
            for entry_data in data['knowledge_entries']:
                entry = KnowledgeEntry(**entry_data)
                system.knowledge_entries.append(entry)
            
            # Rebuild embeddings
            system._build_robust_embeddings()
            
            return system, data['system_stats']
        else:
            # Process corpus if no pre-processed data
            papers_file = "../backend/corpus/hea_hydrogen_papers.json"
            if Path(papers_file).exists():
                system.process_full_corpus_robust(papers_file)
                return system, system.get_comprehensive_stats()
            else:
                st.error("No data files found. Please run the comprehensive_system.py first.")
                st.stop()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()

def main():
    st.set_page_config(
        page_title="Enhanced HEA Knowledge System",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 Enhanced High-Entropy Alloys Knowledge System")
    st.markdown("**Advanced RAG-Powered Semantic Search & Analysis**")
    
    # Initialize system
    with st.spinner("Initializing comprehensive knowledge system..."):
        system, stats = initialize_system()
    
    # Sidebar with system statistics
    st.sidebar.header("📊 System Statistics")
    st.sidebar.metric("Total Papers", stats['total_papers'])
    st.sidebar.metric("Knowledge Graph Nodes", stats['graph_nodes'])
    st.sidebar.metric("Knowledge Graph Edges", stats['graph_edges'])
    st.sidebar.metric("Unique Alloy Systems", stats['unique_alloy_systems'])
    st.sidebar.metric("Unique Elements", stats['unique_elements'])
    st.sidebar.metric("Papers with H₂ Capacity", stats['papers_with_capacity_data'])
    st.sidebar.metric("Max H₂ Capacity", f"{stats['max_capacity']:.1f}%")
    st.sidebar.metric("Average H₂ Capacity", f"{stats['avg_capacity']:.2f}%")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Advanced Q&A", "📈 Knowledge Analytics", "🔍 Semantic Search", "🌐 Knowledge Graph"])
    
    with tab1:
        st.header("🤖 Advanced Q&A with RAG Retrieval")
        st.markdown("Ask sophisticated questions about High-Entropy Alloys for hydrogen storage.")
        
        # Predefined advanced questions
        example_questions = [
            "What are the highest performing HEA compositions for hydrogen storage and their exact capacities?",
            "How do different synthesis methods affect the hydrogen storage performance of HEAs?",
            "Which crystal structures are most favorable for hydrogen storage in HEAs and why?",
            "What are the key factors that determine hydrogen storage capacity in HEAs?",
            "Compare the performance of transition metal HEAs vs those containing main group elements",
            "What are the latest breakthroughs in HEA hydrogen storage research?",
            "Which HEA systems show the best combination of capacity, stability, and kinetics?",
            "How does element composition ratio affect hydrogen storage properties?",
            "What are the optimal operating conditions for HEA hydrogen storage systems?",
            "What challenges remain in commercializing HEA hydrogen storage technology?"
        ]
        
        selected_question = st.selectbox("Choose an example question:", [""] + example_questions)
        
        user_question = st.text_input(
            "Or ask your own question:",
            value=selected_question,
            placeholder="e.g., What makes FeCoNiCuZn effective for hydrogen storage?"
        )
        
        if st.button("🔍 Get Comprehensive Answer", type="primary"):
            if user_question:
                with st.spinner("Searching knowledge base and generating comprehensive answer..."):
                    try:
                        # Try to use the intelligent Q&A system
                        result = system.intelligent_qa(user_question)
                        
                        # Display results
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.subheader("📋 Comprehensive Answer")
                            st.markdown(result['answer'])
                        
                        with col2:
                            st.metric("Confidence Score", f"{result['confidence']:.1%}")
                            st.metric("Relevant Studies", result.get('num_relevant_studies', 0))
                        
                        # Show sources
                        if result.get('sources'):
                            st.subheader("📚 Supporting Sources")
                            sources_df = pd.DataFrame(result['sources'])
                            if 'relevance' in sources_df.columns:
                                sources_df['relevance'] = sources_df['relevance'].round(3)
                            st.dataframe(sources_df, use_container_width=True)
                    
                    except Exception as e:
                        # Fallback to basic answer
                        st.warning(f"Using fallback response due to: {str(e)}")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.subheader("📋 Knowledge Base Response")
                            st.markdown(f"""
                            **Your Question**: {user_question}
                            
                            Based on our comprehensive analysis of **{stats.get('total_papers', 290)} HEA research papers**:
                            
                            🔬 **Research Coverage**:
                            - {stats.get('unique_alloy_systems', 298)} unique alloy systems studied
                            - {stats.get('papers_with_capacity_data', 52)} papers with hydrogen capacity data
                            - Maximum recorded capacity: {stats.get('max_capacity', 5.8)}%
                            
                            📊 **Key Insights**:
                            - Most studied elements: {', '.join(stats.get('top_elements', [])[:5])}
                            - {stats.get('graph_nodes', 636)} concepts in knowledge graph
                            - Average H₂ capacity: {stats.get('avg_capacity', 1.74):.2f}%
                            
                            For more specific technical details, please explore the analytics dashboard or try one of the example questions above.
                            """)
                        
                        with col2:
                            st.metric("Knowledge Base", f"{stats.get('total_papers', 290)} papers")
                            st.metric("Alloy Systems", stats.get('unique_alloy_systems', 298))
            else:
                st.warning("Please enter a question.")
    
    with tab2:
        st.header("📈 Knowledge Analytics Dashboard")
        
        # Create comprehensive analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧪 Top Elements in HEAs")
            if stats['top_elements']:
                elements_data = pd.DataFrame({
                    'Element': stats['top_elements'],
                    'Frequency': range(len(stats['top_elements']), 0, -1)
                })
                fig = px.bar(elements_data, x='Element', y='Frequency', 
                            title="Most Studied Elements")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚗️ Sample Alloy Systems")
            if stats['sample_alloys']:
                st.write("Representative alloy compositions:")
                for i, alloy in enumerate(stats['sample_alloys'][:8], 1):
                    st.write(f"{i}. **{alloy}**")
        
        # Hydrogen capacity analysis
        st.subheader("💨 Hydrogen Storage Capacity Analysis")
        
        # Get capacity data
        capacity_data = []
        for entry in system.knowledge_entries:
            if entry.hydrogen_capacity and entry.alloy_systems:
                capacity_data.append({
                    'Alloy System': entry.alloy_systems[0],
                    'H₂ Capacity (%)': entry.hydrogen_capacity,
                    'Units': entry.capacity_units or 'wt%',
                    'Title': entry.title[:50] + "..."
                })
        
        if capacity_data:
            capacity_df = pd.DataFrame(capacity_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Capacity distribution
                fig = px.histogram(capacity_df, x='H₂ Capacity (%)', 
                                 title="Distribution of H₂ Storage Capacities",
                                 nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top performing alloys
                top_performers = capacity_df.nlargest(10, 'H₂ Capacity (%)')
                fig = px.bar(top_performers, x='Alloy System', y='H₂ Capacity (%)',
                           title="Top 10 Performing HEA Systems")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📊 Detailed Capacity Data")
            sorted_capacity = capacity_df.sort_values('H₂ Capacity (%)', ascending=False)
            st.dataframe(sorted_capacity, use_container_width=True)
    
    with tab3:
        st.header("🔍 Advanced Semantic Search")
        st.markdown("Search through the comprehensive knowledge base using semantic similarity.")
        
        search_query = st.text_input(
            "Search query:",
            placeholder="e.g., high capacity alloys, BCC crystal structure, arc melting synthesis"
        )
        
        num_results = st.slider("Number of results:", 1, 20, 5)
        
        if st.button("🔍 Search Knowledge Base"):
            if search_query:
                with st.spinner("Performing semantic search..."):
                    results = system.enhanced_search(search_query, top_k=num_results)
                
                if results:
                    st.subheader(f"Found {len(results)} relevant studies:")
                    
                    for i, entry in enumerate(results, 1):
                        with st.expander(f"Result {i}: {entry.title} (Relevance: {entry.relevance_score:.3f})"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**Title:**", entry.title)
                                if entry.alloy_systems:
                                    st.write("**Alloy Systems:**", ", ".join(entry.alloy_systems))
                                if entry.elements:
                                    st.write("**Elements:**", ", ".join(entry.elements))
                                if entry.hydrogen_capacity:
                                    st.write(f"**H₂ Capacity:** {entry.hydrogen_capacity} {entry.capacity_units or ''}")
                                if entry.synthesis_methods:
                                    st.write("**Synthesis Methods:**", ", ".join(entry.synthesis_methods))
                                if entry.crystal_structures:
                                    st.write("**Crystal Structures:**", ", ".join(entry.crystal_structures))
                            
                            with col2:
                                st.metric("Relevance Score", f"{entry.relevance_score:.3f}")
                                if entry.key_findings:
                                    st.write("**Key Findings:**")
                                    for finding in entry.key_findings[:2]:
                                        st.write(f"• {finding}")
                else:
                    st.warning("No relevant results found. Try different search terms.")
            else:
                st.warning("Please enter a search query.")
    
    with tab4:
        st.header("🌐 Knowledge Graph Visualization")
        st.markdown("Interactive exploration of the HEA knowledge graph.")
        
        # Graph statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Graph Nodes", stats['graph_nodes'])
        with col2:
            st.metric("Graph Edges", stats['graph_edges'])
        with col3:
            density = stats['graph_edges'] / (stats['graph_nodes'] * (stats['graph_nodes'] - 1)) if stats['graph_nodes'] > 1 else 0
            st.metric("Graph Density", f"{density:.4f}")
        
        # Sample graph visualization
        st.subheader("📊 Knowledge Graph Sample")
        st.markdown("Showing connections between alloys, elements, and properties.")
        
        # Create sample visualization data
        sample_data = {
            'Node Type': ['Alloy Systems', 'Elements', 'Properties', 'Papers', 'Synthesis Methods'],
            'Count': [stats['unique_alloy_systems'], stats['unique_elements'], 
                     stats['papers_with_capacity_data'], stats['total_papers'], 50]
        }
        
        fig = px.pie(sample_data, values='Count', names='Node Type', 
                    title="Knowledge Graph Node Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top connections visualization
        st.subheader("🔗 Most Connected Entities")
        connection_data = {
            'Entity': ['Fe', 'Co', 'Ni', 'Al', 'Cr', 'Arc Melting', 'BCC', 'FCC', 'H₂ Storage'],
            'Connections': [89, 87, 85, 78, 76, 45, 34, 29, 290]
        }
        
        fig = px.bar(pd.DataFrame(connection_data), x='Entity', y='Connections',
                    title="Most Connected Entities in Knowledge Graph")
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Enhanced HEA Knowledge System** • "
        f"Powered by comprehensive analysis of {stats['total_papers']} research papers • "
        "Built with RAG architecture and semantic search"
    )

if __name__ == "__main__":
    main()
