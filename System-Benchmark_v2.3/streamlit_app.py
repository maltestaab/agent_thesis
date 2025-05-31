"""
streamlit_app.py - Streamlit interface

# ==== WHAT THIS FILE DOES ====
# Creates a web interface where users can:
# 1. Upload data files (CSV, Excel)
# 2. See a preview of their data
# 3. Describe what analysis they want
# 4. Watch AI agents analyze their data in real-time
# 5. See results, charts, and insights

"""
import streamlit as st
import asyncio
import os
import pandas as pd
import time
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to handle asyncio in Streamlit (allows nested asyncio loops)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Import our simplified streaming systems
from data_science_agents.agent_systems.single_agent import run_single_agent_analysis
from data_science_agents.agent_systems.multi_agent import run_multi_agent_analysis
from data_science_agents.config.settings import MAX_TOKENS, SUPPORTED_FILE_TYPES
from data_science_agents.config.prompts import ANALYSIS_PROMPT_TEMPLATE


def display_images_gallery(images):
    """Display created images at the end of the analysis"""
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
        st.subheader("Live Analysis Progress")
        
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


## MAIN APP ##

# Page configuration
st.set_page_config(
    page_title="Data Science Analysis Tool",
    layout="wide", #full width of browser
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# CSS styling for the streaming UI
st.markdown("""
<style>
    .streaming-text {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        max-height: 400px;
        overflow-y: auto;
    }
    .tool-activity {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .event-log {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        max-height: 200px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.8rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state. State remembers between page reloads.
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

if 'cancel_analysis' not in st.session_state:
    st.session_state.cancel_analysis = False

if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = None

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
    
    # Streaming toggle (Streaming yes or no)
    show_live_streaming = st.checkbox(
        "🔴 Show Live Progress",
        value=True,
        help="Show real-time progress with live agent output and tool calls"
    )
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🤖 Model Settings")
    
    # Available models from token costs
    available_models = [
        "gpt-4o-mini",
        "gpt-4o", 
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    
    # Model descriptions
    model_descriptions = {
        "gpt-4o-mini": "Fast and cost-effective, ideal for most data analysis tasks",
        "gpt-4o": "Latest GPT-4 model with best performance and reasoning capabilities", 
        "gpt-4": "Previous GPT-4 version, very capable for complex analysis",
        "gpt-3.5-turbo": "Fastest and most economical option for basic analysis"
    }
    
    selected_model = st.selectbox(
        "Choose AI Model",
        available_models,
        index=0,  # Default to gpt-4o-mini
        help="Select the AI model for analysis"
    )
    
    # Show model description and pricing
    if selected_model:
        from data_science_agents.config.settings import TOKEN_COSTS
        costs = TOKEN_COSTS[selected_model]
        
        # Description
        st.markdown(f"**{model_descriptions[selected_model]}**")
        
        # Pricing in a clean, smaller format
        st.markdown("**Pricing per 1 million tokens:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📥 **Input:** ${costs['input']:.2f}")
        with col2:
            st.write(f"📤 **Output:** ${costs['output']:.2f}")

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

# Analysis prompt input
user_prompt = st.text_area(
    "Describe your analysis needs",
    value="Please find the effect the explanatory variables have on the target variable winpercent and visualize your results.",
    height=100,
    help="Be specific about what insights you're looking for"
)

# Analysis buttons with stop functionality
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    stop_analysis = st.button(
        "🛑 Stop Analysis",
        disabled=not st.session_state.analysis_running,
        use_container_width=True
    )
    if stop_analysis:
        st.session_state.cancel_analysis = True
        st.warning("⏹️ Cancellation requested...")

with col2:
    run_analysis = st.button(
        f"🚀 Run {agent_mode} Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.analysis_running or not uploaded_file or not user_prompt
    )

with col3:
    st.empty()  # Spacing

# Run analysis when button is clicked
if run_analysis:
    st.session_state.analysis_running = True
    st.session_state.cancel_analysis = False  # Reset cancel flag
    st.session_state.analytics_data = None  # Reset analytics
    
    # Save uploaded file and analysis details
    st.session_state.file_name = uploaded_file.name
    st.session_state.user_prompt = user_prompt
    st.session_state.agent_mode = agent_mode
    st.session_state.show_live_streaming = show_live_streaming
    st.session_state.selected_model = selected_model  # Store selected model
    
    # Save the file
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # Force rerun to show the stop button as enabled
    st.rerun()

# Run analysis if we're in running state
if st.session_state.analysis_running and 'file_name' in st.session_state:
    # Create analysis prompt
    analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        file_name=st.session_state.file_name,
        user_prompt=st.session_state.user_prompt,
        max_tokens=MAX_TOKENS
    )
    
    # === STREAMING ANALYSIS ===
    st.markdown("---")
    
    # Create live analytics at the top
    st.subheader("📊 Live Analytics")
    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
    with analytics_col1:
        duration_metric = st.empty()
    with analytics_col2:
        tools_metric = st.empty()
    with analytics_col3:
        cost_metric = st.empty()
    with analytics_col4:
        images_metric = st.empty()
    
    st.markdown("---")
    
    # Create streaming UI containers only if user wants to see live progress
    if st.session_state.show_live_streaming:
        agent_output, tool_activity, event_log = create_streaming_ui()
    
    async def run_streaming_analysis():
        """Run analysis with live streaming updates"""
        
        try:
            # Choose analysis function based on mode
            if st.session_state.agent_mode == "Single Agent":
                analysis_stream = run_single_agent_analysis(
                    analysis_prompt, 
                    st.session_state.file_name, 
                    max_turns=50,
                    model=st.session_state.get('selected_model', 'gpt-4o-mini')
                )
            else:
                analysis_stream = run_multi_agent_analysis(
                    analysis_prompt, 
                    st.session_state.file_name, 
                    max_turns=50,
                    model=st.session_state.get('selected_model', 'gpt-4o-mini')
                )
            
            # Initialize analytics tracking (always)
            live_analytics = {
                'start_time': time.time(),
                'tool_calls': 0,
                'total_tokens': 0,
                'images_created': 0
            }
            
            # Update live analytics display function
            def update_live_analytics():
                duration = time.time() - live_analytics['start_time']
                # Updated cost calculation for per-million-token rates
                estimated_cost = live_analytics['total_tokens'] * 0.15 / 1_000_000  # Using gpt-4o-mini rate as default
                
                duration_metric.metric("Duration", f"{duration:.1f}s")
                tools_metric.metric("Tool Calls", live_analytics['tool_calls'])
                cost_metric.metric("Est. Cost", f"${estimated_cost:.4f}")
                images_metric.metric("Images", live_analytics['images_created'])
            
            # Initialize streaming variables only if we're showing live progress
            if st.session_state.show_live_streaming:
                agent_text = ""
                full_conversation = []  # Keep full conversation history
                tool_events = []
                event_history = []
            
            async for event in analysis_stream:
                current_time = time.time()
                
                # Always track analytics regardless of streaming mode
                if event.event_type == "text_delta":
                    # Estimate tokens (rough: 4 chars = 1 token)
                    live_analytics['total_tokens'] += len(event.content) / 4
                elif event.event_type == "tool_call":
                    live_analytics['tool_calls'] += 1
                
                # Handle different event types for streaming
                if st.session_state.show_live_streaming:
                    if event.event_type == "text_delta":
                        # Live text streaming - append to current text
                        agent_text += event.content
                        agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "analytics_start":
                        # Initialize analytics display - just update local tracking
                        pass
                    
                    elif event.event_type == "analytics_complete":
                        # Parse final analytics from content and store in session state
                        try:
                            import re
                            duration_match = re.search(r'(\d+\.?\d*)s', event.content)
                            tools_match = re.search(r'(\d+) tools', event.content)
                            phases_match = re.search(r'(\d+) phases', event.content)
                            cost_match = re.search(r'\$(\d+\.?\d+)', event.content)
                            
                            from data_science_agents.core.execution import get_created_images
                            
                            st.session_state.analytics_data = {
                                'total_duration': float(duration_match.group(1)) if duration_match else 0,
                                'tool_calls': int(tools_match.group(1)) if tools_match else live_analytics['tool_calls'],
                                'phases_completed': int(phases_match.group(1)) if phases_match else 0,
                                'estimated_cost': float(cost_match.group(1)) if cost_match else live_analytics['total_tokens'] * 0.00015 / 1000,
                                'images_created': len(get_created_images()),
                                'agent_durations': {}
                            }
                            
                            # Update live display with final values
                            update_live_analytics()
                            
                        except Exception as e:
                            # Use live analytics as fallback
                            st.session_state.analytics_data = {
                                'total_duration': time.time() - live_analytics['start_time'],
                                'tool_calls': live_analytics['tool_calls'],
                                'estimated_cost': live_analytics['total_tokens'] * 0.00015 / 1000,
                                'images_created': live_analytics['images_created'],
                                'agent_durations': {}
                            }
                    
                    elif event.event_type == "tool_call":
                        # Tool call started
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        # Show ALL tool events, not just last 3
                        tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events)}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "tool_output":
                        # Tool call completed
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - ✅ Execution completed")
                        tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events)}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "agent_reasoning":
                        # Add reasoning to conversation history
                        full_conversation.append(f"🤔 {event.agent_name}: {event.content}")
                        # Show complete conversation, not just current text
                        conversation_html = "<br><br>".join(full_conversation)
                        agent_output.markdown(f'<div class="streaming-text">{conversation_html}</div>', unsafe_allow_html=True)
                                        
                    elif event.event_type in ["agent_handoff", "agent_result", "sub_agent_start", "sub_agent_complete"]:
                        # Multi-agent events
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events)}</div>', unsafe_allow_html=True)
                    
                    elif event.event_type == "message_complete":
                        # Message complete - add to conversation history but don't reset current text
                        if agent_text.strip():
                            full_conversation.append(agent_text.strip())
                        agent_text = ""  # Reset for next message
                    
                    # Add ALL events to history (except text deltas for readability)
                    if event.event_type not in ["text_delta"]:
                        event_history.append(f"[{event.agent_name}] {event.event_type}")
                        # Show only last 20 events with scrolling
                        recent_events = event_history[-20:] if len(event_history) > 20 else event_history
                        event_log.markdown(f'<div class="event-log">{"<br>".join(recent_events)}</div>', unsafe_allow_html=True)
                    
                    # Update live analytics in main area every few events
                    if len(event_history) % 5 == 0 or event.event_type == "tool_call":
                        update_live_analytics()
                    
                    # Store analytics in session state for sidebar (updated less frequently)
                    if len(event_history) % 10 == 0:
                        duration = current_time - live_analytics['start_time']
                        estimated_cost = live_analytics['total_tokens'] * 0.00015 / 1000
                        
                        st.session_state.analytics_data = {
                            'total_duration': duration,
                            'tool_calls': live_analytics['tool_calls'],
                            'estimated_cost': estimated_cost,
                            'images_created': live_analytics['images_created'],
                            'agent_durations': {}
                        }
                    
                    # Small delay to make streaming visible
                    await asyncio.sleep(0.05)
                
                # Always update analytics every 10 events, even without streaming
                if hasattr(event, 'event_type') and len(str(event.event_type)) > 0:
                    if live_analytics['tool_calls'] % 5 == 0:
                        update_live_analytics()
                
                # Handle completion or error (regardless of streaming preference)
                if event.event_type == "analytics_complete":
                    # Always capture final analytics, even if streaming is off
                    try:
                        import re
                        duration_match = re.search(r'(\d+\.?\d*)s', event.content)
                        tools_match = re.search(r'(\d+) tools', event.content)
                        phases_match = re.search(r'(\d+) phases', event.content)
                        cost_match = re.search(r'\$(\d+\.?\d+)', event.content)
                        
                        from data_science_agents.core.execution import get_created_images
                        
                        final_analytics = {
                            'total_duration': float(duration_match.group(1)) if duration_match else (time.time() - live_analytics['start_time']),
                            'tool_calls': int(tools_match.group(1)) if tools_match else live_analytics['tool_calls'],
                            'phases_completed': int(phases_match.group(1)) if phases_match else 0,
                            'estimated_cost': float(cost_match.group(1)) if cost_match else (live_analytics['total_tokens'] * 0.00015 / 1000),
                            'images_created': len(get_created_images()),
                            'agent_durations': {}
                        }
                        
                        st.session_state.analytics_data = final_analytics
                        
                        # Update live display with final values
                        live_analytics.update({
                            'tool_calls': final_analytics['tool_calls'],
                            'total_tokens': final_analytics['estimated_cost'] * 1000 / 0.00015,  # Reverse calculate
                            'images_created': final_analytics['images_created']
                        })
                        update_live_analytics()
                        
                    except Exception as e:
                        # Use live analytics as fallback
                        st.session_state.analytics_data = {
                            'total_duration': time.time() - live_analytics['start_time'],
                            'tool_calls': live_analytics['tool_calls'],
                            'estimated_cost': live_analytics['total_tokens'] * 0.00015 / 1000,
                            'images_created': live_analytics['images_created'],
                            'agent_durations': {}
                        }
                        update_live_analytics()
                
                elif event.event_type == "analysis_complete":
                    st.session_state.analysis_running = False
                    
                    # Ensure final analytics are saved
                    if not st.session_state.get('analytics_data'):
                        st.session_state.analytics_data = {
                            'total_duration': time.time() - live_analytics['start_time'],
                            'tool_calls': live_analytics['tool_calls'],
                            'estimated_cost': live_analytics['total_tokens'] * 0.15 / 1_000_000,
                            'images_created': live_analytics['images_created'],
                            'agent_durations': {}
                        }
                    
                    # Final analytics update
                    update_live_analytics()
                    
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
                    
                elif event.event_type == "analysis_cancelled":
                    st.session_state.analysis_running = False
                    st.warning(event.content)
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
        st.session_state.cancel_analysis = False