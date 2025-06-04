"""
streamlit_app.py - Main Web Interface for Data Science Agent System

====== WHAT THIS FILE DOES ======
This is the main entry point for the entire application. It creates a web interface where users can:
1. Upload data files (CSV, Excel) through a simple drag-and-drop interface
2. See an immediate preview of their data with key statistics
3. Describe what kind of analysis they want in plain English
4. Choose between Single Agent or Multi-Agent analysis modes
5. Watch AI agents analyze their data in real-time with live progress updates
6. See results, charts, and comprehensive insights generated automatically

====== HOW TO USE ======
1. Run in Termainal: streamlit run streamlit_app.py
2. Open browser to displayed URL (usually http://localhost:8501)
3. Upload your data file and enter analysis request
4. Watch the AI agents work and see results
"""

import streamlit as st
import asyncio
import os
import pandas as pd
import time
import re
from dotenv import load_dotenv
import nest_asyncio
from collections import deque

                            

# Apply nest_asyncio to handle asyncio in Streamlit (allows nested asyncio loops for streaming)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

from data_science_agents.agent_systems.single_agent import run_single_agent_analysis
from data_science_agents.agent_systems.multi_agent import run_multi_agent_analysis
from data_science_agents.core.execution import get_created_images
from data_science_agents.config.settings import SUPPORTED_FILE_TYPES
from data_science_agents.config.prompts import ANALYSIS_PROMPT_TEMPLATE


def display_images_gallery(images):
    """
    Display created visualizations in a gallery format.
    
    Args:
        images: List of image file paths created during analysis
        
    What this does:
    - Filters for actual image files (png, jpg, etc.) since agents sometimes create files that are not images.
    - Creates a two-column layout to display images
    - Shows each image with its filename as caption
    - Skips over files that can't be displayed as images
    """
    if not images:
        return
    
    # Valid image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff', '.webp')
    
    # Filter for image files only
    image_files = [img for img in images if img.lower().endswith(image_extensions) and os.path.exists(img)]
    
    if not image_files:
        return
    
    st.subheader("📸 Generated Visualizations")
    
    # Display images in two columns for better layout
    cols = st.columns(2)
    for i, img_path in enumerate(image_files):
        try:
            with cols[i % 2]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        except Exception as e:
            # Skip files that can't be displayed as images
            continue
                
def create_streaming_ui():
    """
    Create the real-time interface containers for live analysis updates.
    
    Returns:
        tuple: Three Streamlit containers for different types of updates
            - agent_output: Shows AI thinking and responses
            - tool_activity: Shows when AI is running code or using tools
            - event_log: Shows overall progress timeline
    """
    
    # Main containers for all streaming updates
    streaming_container = st.container()
    
    with streaming_container:
        st.subheader("Analysis Progress")
        
        # Create columns for different types of updates
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Agent messages streaming output
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

# Page configuration - sets up the browser tab and layout
st.set_page_config(
    page_title="Data Science Analysis Tool",
    layout="wide", #full width of browser
    initial_sidebar_state="expanded", # show sidebar by default
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
# Important when we switch between analysis mode but want to keep prompt and file.
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False # Tracks if analysis runs

if 'cancel_analysis' not in st.session_state:
    st.session_state.cancel_analysis = False # Tracks if use wants to cancel analysis

if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = None # Tracks analytics data

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
        help="Single Agent: One AI handles all phases. Multi-Agent: Specialized agents for different phases."
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
        index=0,  # Default to gpt-4o-mini since it's the cheapest
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
    
    # Save uploaded file and analysis details accessible for AI agents
    st.session_state.file_name = uploaded_file.name
    st.session_state.user_prompt = user_prompt
    st.session_state.agent_mode = agent_mode
    st.session_state.show_live_streaming = show_live_streaming
    st.session_state.selected_model = selected_model  
    
    # Save the file to working directory
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # Force page refresh to show the stop button as enabled
    st.rerun()

# Run analysis if we're in running state
if st.session_state.analysis_running and 'file_name' in st.session_state:
    # Create analysis prompt (combines user prompt, file name and max tokens)
    analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        file_name=st.session_state.file_name,
        user_prompt=st.session_state.user_prompt
    )
    
    # === STREAMING ANALYSIS ===
    st.markdown("---")
    
    # Create live analytics at the top
    st.subheader("📊 Analytics")
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
        """
        Main analysis orchestration function.
        
        This function:
        1. Chooses between single-agent or multi-agent analysis
        2. Sets up real-time progress tracking
        3. Processes streaming events from AI agents
        4. Updates the user interface in real-time
        5. Handles completion, errors, and cancellation
        """
        
        try:
            # Choose analysis function based on mode (single or Multi)
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
            
            # Initialize streaming variables only if showing live progress is toggled
            if st.session_state.show_live_streaming:
                agent_text = ""  # Live window text (limited)
                full_agent_text = ""  # Complete text for final display
                text_buffer = ""  # Buffer for text deltas
                last_text_update = time.time()
                
                full_conversation = []  # Complete conversation history for final display
                tool_events = deque()  # Tool activity history
                event_history = deque()  # Event timeline
                
                # Time-based update tracking
                last_analytics_update = time.time()
                last_tool_update = time.time()
                last_event_update = time.time()
                event_counter = 0
            
            # Main event loop for streaming analysis
            async for event in analysis_stream:
                current_time = time.time()
                
                # Always track analytics regardless of streaming mode (text_delta, tool_call, etc.)
                if event.event_type == "text_delta":
                    # Estimate tokens (rough: 4 chars = 1 token)
                    live_analytics['total_tokens'] += len(event.content) / 4 # Count tokens based on output length in characters
                elif event.event_type == "tool_call":
                    live_analytics['tool_calls'] += 1 # Count tool calls
                
                # Process streaming events if live streaming is toggled
                if st.session_state.show_live_streaming:
                    if event.event_type == "text_delta":
                        # Buffer text updates - update every 2.5 seconds to improve streamlit performance
                        text_buffer += event.content
                        full_agent_text += event.content  # Store complete text
                        
                        if current_time - last_text_update >= 2.5:
                            agent_text += text_buffer
                            text_buffer = ""
                            
                            # Limits live visible text to 400 characters to improve streamlit performance
                            if len(agent_text) > 400:
                                agent_text = agent_text[-400:]
                            
                            agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                            last_text_update = current_time
                    
                    elif event.event_type == "analytics_start":
                        # Initialize analytics display - just update local tracking
                        pass
                    
                    elif event.event_type == "analytics_complete":
                        # Extracting final metrics from the analysis completion event
                        try:
                            
                            duration_match = re.search(r'(\d+\.?\d*)s', event.content)
                            tools_match = re.search(r'(\d+) tools', event.content)
                            phases_match = re.search(r'(\d+) phases', event.content)
                            cost_match = re.search(r'\$(\d+\.?\d+)', event.content)

                            
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
                    
                    elif event.event_type == "tool_call": # Show when agents are running code or using tools
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        # Update display every 2.5 seconds to improve streamlit performance
                        if current_time - last_tool_update >= 2.5:
                            # Show only last 2 items during streaming to improve streamlit performance
                            recent_tools = list(tool_events)[-2:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "tool_output":
                        # Tool call completed
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - ✅ Execution completed")
                        
                        if current_time - last_tool_update >= 2.5:
                            # Show only last 2 items during streaming to improve streamlit performance
                            recent_tools = list(tool_events)[-2:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "agent_reasoning": 
                        # Capture sub-agent reasoning for final display
                        if st.session_state.agent_mode == "Multi-Agent":
                            full_agent_text += f"\n🤔 {event.agent_name}: {event.content}\n"

                    elif event.event_type in ["agent_handoff", "agent_result", "sub_agent_start", "sub_agent_complete"]:
                        # Multi-agent events
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 2.5:
                            # Show only last 2 items during streaming
                            recent_tools = list(tool_events)[-2:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "message_complete":
                        # Message complete - flush any remaining text buffer when agent finishes a message
                        if text_buffer:
                            agent_text += text_buffer
                            full_agent_text += text_buffer  # Add to complete text too
                            text_buffer = ""
                            
                            # Show only last 400 characters in live window to improve streamlit performance
                            if len(agent_text) > 400:
                                agent_text = agent_text[-400:]
                            
                            agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                        
                        # Reset for next message
                        agent_text = ""
                    
                    # Add events to history - show only last 3 during streaming, update every 8 events to improve streamlit performance
                    if event.event_type not in ["text_delta"]:
                        event_history.append(f"[{event.agent_name}] {event.event_type}")
                        event_counter += 1
                        
                        if event_counter % 8 == 0:
                            # Show only last 3 events during streaming to improve streamlit performance
                            recent_events = list(event_history)[-3:]
                            event_log.markdown(f'<div class="event-log">{"<br>".join(recent_events)}</div>', unsafe_allow_html=True)
                    
                    # Update analytics every 4 seconds or 10 events to improve streamlit performance
                    if (current_time - last_analytics_update >= 4.0) or (event_counter % 10 == 0):
                        update_live_analytics()
                        last_analytics_update = current_time
                    
                    # Store analytics in session state less frequently to improve streamlit performance
                    if event_counter % 20 == 0:
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
                
                # Always update ANALYTICS every 10 events, even without streaming
                if hasattr(event, 'event_type') and len(str(event.event_type)) > 0:
                    if live_analytics['tool_calls'] % 10 == 0:
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
                    
                    # Flush any remaining text buffer
                    if st.session_state.show_live_streaming and 'text_buffer' in locals() and text_buffer:
                        agent_text += text_buffer
                        full_agent_text += text_buffer
                        if len(agent_text) > 600:
                            agent_text = agent_text[-600:]
                        agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                    
                    # Convert live windows to full scrollable versions to look at the complete process after the analysis is complete
                    if st.session_state.show_live_streaming:
                        # Replace agent output with full content
                        if full_agent_text or full_conversation:
                            combined_content = ""
                            if full_agent_text:
                                combined_content += full_agent_text
                            if full_conversation:
                                if combined_content:
                                    combined_content += "<br><br>" + "<br><br>".join(full_conversation)
                                else:
                                    combined_content = "<br><br>".join(full_conversation)
                            agent_output.markdown(f'<div class="streaming-text">{combined_content}</div>', unsafe_allow_html=True)
                        
                        # Replace tool activity with full history to look at the complete process after the analysis is complete
                        if tool_events:
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events)}</div>', unsafe_allow_html=True)
                        
                        # Replace event log with full history to look at the complete process after the analysis is complete
                        if event_history:
                            event_log.markdown(f'<div class="event-log">{"<br>".join(event_history)}</div>', unsafe_allow_html=True)
                    
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
                    
                    # Show final results of the analysis
                    st.markdown("### 📊 Final Analysis Results")
                    st.markdown(event.content)
                    
                    # Show images if any were created
                    from data_science_agents.core.execution import get_created_images
                    images = get_created_images()
                    if images:
                        time.sleep(0.3)  # Gives the system some time to finish saving the images before displaying them
                        st.markdown("---")
                        display_images_gallery(images)
                    
                    return
                    
                elif event.event_type == "analysis_error": # Show when analysis fails
                    st.session_state.analysis_running = False
                    st.error(event.content)
                    return
                    
                elif event.event_type == "analysis_cancelled": # Show when analysis is cancelled
                    st.session_state.analysis_running = False
                    st.warning(event.content)
                    return
                    
        except Exception as e:
            st.error(f"Streaming analysis failed: {str(e)}")
            st.session_state.analysis_running = False
    
    # Run streaming analysis (this is the main function that runs the analysis)
    try:
        asyncio.run(run_streaming_analysis())
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        st.session_state.analysis_running = False
        st.session_state.cancel_analysis = False