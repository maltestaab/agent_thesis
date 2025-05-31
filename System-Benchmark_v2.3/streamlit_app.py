"""
streamlit_app.py - Streamlit interface with live streaming
"""
import streamlit as st
import asyncio
import os
import pandas as pd
import time
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to handle asyncio in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Import our simplified streaming systems
from data_science_agents.agent_systems.single_agent import run_single_agent_analysis
from data_science_agents.agent_systems.multi_agent import run_multi_agent_analysis
from data_science_agents.config.settings import MAX_TOKENS, SUPPORTED_FILE_TYPES
from data_science_agents.config.prompts import ANALYSIS_PROMPT_TEMPLATE


def display_images_gallery(images):
    """Display images in a simple gallery format"""
    if not images:
        return
    
    st.subheader("📸 Generated Visualizations")
    
    # Display images in columns for better layout
    cols = st.columns(2)
    for i, img_path in enumerate(images):
        if os.path.exists(img_path):
            with cols[i % 2]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True) 
                
def create_streaming_ui():
    """Create the live streaming interface containers"""
    
    # Main containers for streaming updates
    streaming_container = st.container()
    
    with streaming_container:
        st.subheader("🔴 Live Analysis Progress")
        
        # Create columns for different types of updates
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main output - agent messages and text generation
            st.markdown("**Agent Output:**")
            agent_output = st.empty()
            
        with col2:
            # Tool calls and status updates
            st.markdown("**Tool Activity:**")
            tool_activity = st.empty()
        
        # Event log at bottom
        st.markdown("**Event Log:**")
        event_log = st.empty()
    
    return agent_output, tool_activity, event_log


# Page configuration
st.set_page_config(
    page_title="Data Science Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Simple CSS for clean styling with auto-scroll
st.markdown("""
<style>
    .streaming-text {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        max-height: 400px;
        overflow-y: auto;
        scroll-behavior: smooth;
    }
    .tool-activity {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        max-height: 300px;
        overflow-y: auto;
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Main header
st.title("📊 Data Science Analysis Tool")
st.markdown("Upload your data and analyze it using AI agents with **live streaming**")

# Sidebar configuration  
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Agent mode selection
    agent_mode = st.radio(
        "Analysis Mode",
        ["Single Agent", "Multi-Agent"],
        help="Single Agent: One AI handles all phases. Multi-Agent: Specialized agents for each CRISP-DM phase."
    )
    
    # Streaming toggle - everything is streaming now, but keep toggle for user preference
    show_live_streaming = st.checkbox(
        "🔴 Show Live Progress",
        value=True,
        help="Show real-time progress with live agent output and tool calls"
    )
    
    st.markdown("---")
    
    # API Key status check
    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ OpenAI API Key detected")
    else:
        st.error("❌ OpenAI API Key not found")
        st.info("Set OPENAI_API_KEY in your environment")

# Main layout
st.header("📁 Data Upload & Analysis")

# File upload section
uploaded_file = st.file_uploader(
    "Choose your dataset",
    type=SUPPORTED_FILE_TYPES,
    help="Upload a CSV or Excel file containing your data"
)

# Immediate data preview when file is uploaded
if uploaded_file:
    try:
        # Load and preview data immediately
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # File info in compact format
        file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
        st.success(f"✅ **{uploaded_file.name}** ({file_size:.2f} MB) - {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Immediate data preview
        st.subheader("📋 Dataset Preview")
        
        # Key stats in compact format
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage().sum() / 1024 / 1024:.1f} MB")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        # Data preview table
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information in expandable section
        with st.expander("📊 Column Details"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Reset file pointer for analysis
        uploaded_file.seek(0)
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Analysis request section
st.subheader("🎯 Analysis Request")

# Preset prompts
example_prompts = {
    "Predictive Modeling": "Build a predictive model to forecast the target variable and identify the most important features.",
    "Exploratory Analysis": "Perform comprehensive exploratory data analysis and identify key patterns and insights.",
    "Classification Task": "Build a classification model and evaluate its performance with appropriate metrics.",
    "Feature Analysis": "Analyze which features are most important and how they relate to the target variable.",
    "Custom Analysis": ""
}

# Prompt selection
selected_example = st.selectbox("Choose example or write custom", list(example_prompts.keys()))
default_prompt = example_prompts[selected_example] if selected_example != "Custom Analysis" else ""

# Analysis prompt input
user_prompt = st.text_area(
    "Describe your analysis needs",
    value=default_prompt,
    height=100,
    help="Be specific about what insights you're looking for"
)

# Analysis button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_analysis = st.button(
        f"🚀 Run {agent_mode} Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.analysis_running or not uploaded_file or not user_prompt
    )

# Run analysis when button is clicked
if run_analysis:
    st.session_state.analysis_running = True
    
    # Save uploaded file
    file_name = uploaded_file.name
    with open(file_name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # Create analysis prompt
    analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        file_name=file_name,
        user_prompt=user_prompt,
        max_tokens=MAX_TOKENS
    )
    
    # === STREAMING ANALYSIS ===
    st.markdown("---")
    
    # Create streaming UI containers only if user wants to see live progress
    if show_live_streaming:
        agent_output, tool_activity, event_log = create_streaming_ui()
    
    async def run_streaming_analysis():
        """Run analysis with live streaming updates"""
        
        try:
            # Choose analysis function based on mode
            if agent_mode == "Single Agent":
                analysis_stream = run_single_agent_analysis(analysis_prompt, file_name, max_turns=50)
            else:
                analysis_stream = run_multi_agent_analysis(analysis_prompt, file_name, max_turns=50)
            
            # Initialize streaming variables only if we're showing live progress
            if show_live_streaming:
                agent_text = ""
                tool_events = []
                event_history = []
            
            async for event in analysis_stream:
                # Handle different event types
                if show_live_streaming:
                    if event.event_type == "text_delta":
                        # Live text streaming - append to current text
                        agent_text += event.content
                        agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "tool_call":
                        # Tool call started - show in tool activity only
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events[-3:])}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "tool_output":
                        # Tool call completed - show in tool activity only
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - ✅ Execution completed")
                        tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events[-3:])}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "agent_reasoning":
                        # Agent reasoning/thinking - show this prominently with auto-scroll
                        agent_output.markdown(f'''
                        <div class="streaming-text" id="agent-output">
                            {event.content}
                        </div>
                        <script>
                            const element = document.getElementById('agent-output');
                            if (element) {{
                                element.scrollTop = element.scrollHeight;
                            }}
                        </script>
                        ''', unsafe_allow_html=True)
                                        
                    elif event.event_type == "agent_handoff":
                        # Multi-agent handoff
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events[-3:])}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "message_complete":
                        # Complete message - reset for next thinking block
                        agent_text = ""
                    
                    # Only add meaningful events to history (not text deltas)
                    if event.event_type not in ["text_delta"]:
                        event_history.append(f"[{event.agent_name}] {event.event_type}")
                        event_log.text("\n".join(event_history[-8:]))  # Show last 8 events, fewer lines
                    
                    # Small delay to make streaming visible
                    await asyncio.sleep(0.1)
                
                # Handle completion or error (regardless of streaming preference)
                if event.event_type == "analysis_complete":
                    st.session_state.analysis_running = False
                    st.success("🎉 Analysis completed!")
                    
                    # Show final results cleanly formatted
                    st.markdown("### 📊 Final Analysis Results")
                    st.markdown(event.content)
                    
                    # Show images if any were created
                    from data_science_agents.core.execution import get_created_images
                    images = get_created_images()
                    if images:
                        st.markdown("---")
                        display_images_gallery(images)
                    
                    return
                    
                elif event.event_type == "analysis_error":
                    st.session_state.analysis_running = False
                    st.error(event.content)
                    return
                    
        except Exception as e:
            st.error(f"Streaming analysis failed: {str(e)}")
            st.session_state.analysis_running = False
    
    # Run streaming analysis
    try:
        asyncio.run(run_streaming_analysis())
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        st.session_state.analysis_running = False