Data Science Analysis Tool
An automated data science analysis system using AI agents. The tool supports both single-agent and multi-agent approaches for comprehensive data analysis following the CRISP-DM methodology.

Features
Single Agent Mode: One comprehensive agent handles all analysis phases
Multi-Agent Mode: Specialized agents for each CRISP-DM phase (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment)
Live Streaming: Real-time progress updates during analysis
Multiple AI Models: Support for GPT-4o, GPT-4o-mini, GPT-4, and GPT-3.5-turbo
Data Upload: CSV and Excel file support
Visualization: Automatic chart generation and display
Analytics Tracking: Comprehensive performance and cost tracking
Setup
Requirements
Python 3.7 or higher
OpenAI API key
Installation
Clone the repository
Install dependencies:
bash
pip install -r requirements.txt
Set up environment variables:
bash
export OPENAI_API_KEY="your_api_key_here"
Or create a .env file:
OPENAI_API_KEY=your_api_key_here
Running the Application
Start the Streamlit web interface:

bash
streamlit run streamlit_app.py
The application will open in your web browser at http://localhost:8501

Usage
Upload Data: Choose a CSV or Excel file containing your dataset
Preview Data: Review the automatic data preview and statistics
Analysis Request: Describe what analysis you want to perform
Choose Mode: Select between Single Agent or Multi-Agent analysis
Select Model: Choose the AI model based on your needs and budget
Run Analysis: Start the analysis and watch real-time progress
View Results: See the complete analysis results and generated visualizations
Analysis Modes
Single Agent
One expert agent handles all phases of analysis
Faster execution with fewer overhead
Good for straightforward analysis tasks
Multi-Agent
Specialized agents for each CRISP-DM phase
More thorough analysis with expert knowledge per phase
Better for complex, multi-faceted analysis
Configuration
Key settings can be modified in data_science_agents/config/settings.py:

DEFAULT_MODEL: Default AI model to use
MAX_TURNS_SINGLE: Maximum turns for single agent analysis
MAX_TURNS_SPECIALIST: Maximum turns for specialist agents
MAX_TOKENS: Maximum tokens per analysis
Architecture
The system is built using the OpenAI Agents SDK and follows a modular architecture:

Agent Systems: Single and multi-agent analysis implementations
Core Modules: Execution engine, analytics tracking, context management
Utilities: Shared streaming and agent factory utilities
Configuration: Prompts, settings, and model configurations
Cost Tracking
The tool provides real-time cost estimation based on token usage and current OpenAI pricing. Costs are displayed during analysis and in the final analytics summary.

Supported File Types
CSV files (.csv)
Excel files (.xlsx, .xls)
Output
Analysis results include:

Comprehensive written analysis and insights
Automatically generated visualizations
Performance metrics and cost tracking
Downloadable charts and plots
