# Data Science Analysis Tool

An AI-powered data science analysis tool that automatically analyzes your datasets using advanced AI agents. Upload your data, describe what insights you need, and watch as AI agents perform comprehensive analysis with real-time progress tracking. Perfect to compare the single- and multi-agent architecture on a variety of Data Science tasks.

## 🚀 Quick Start

```bash
# 1. Clone and navigate
git clone <repository-url>
cd System-Benchmark_v2.5

# 2. Install the package (recommended)
pip install -e .

# 3. Create environment file
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 4. Run the application
streamlit run streamlit_app.py
```

## 🚀 Features

- **Automated Analysis**: Upload CSV/Excel files and get comprehensive data science insights
- **Two Analysis Modes**:
  - **Single Agent**: One AI agent handles the complete analysis
  - **Multi-Agent**: Specialized agents for different analysis phases (Data Understanding, Preparation, Modeling, Evaluation)
- **Real-time Progress**: Watch AI agents work with live streaming of their reasoning and code execution
- **Multiple AI Models**: Choose from various OpenAI models (GPT-4o-mini, GPT-4.1, o3-mini, etc.)
- **Automatic Visualizations**: Agents create and save relevant charts and plots
- **Live Analytics**: Monitor analysis duration, costs, and tool usage in real-time

## 📦 Setup

### 1. Prerequisites
- Python 3.7 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 2. Clone the Repository
```bash
git clone <repository-url>
cd System-Benchmark_v2.5
```

### 3. Install the Package
Choose one of the following installation methods:

#### Option A: Development Installation (Recommended)
```bash
# Install in development mode - allows you to modify the code
pip install -e .
```

#### Option B: Direct Installation
```bash
# Install the package and all dependencies
pip install .
```

#### Option C: Manual Dependencies (Alternative)
```bash
# Install dependencies manually if setup.py doesn't work
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root directory:
```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Langfuse Observability (for monitoring)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5. Run the Application
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### 7. Verify Installation
- Upload a sample CSV file
- Try a simple analysis request
- Check that visualizations appear in the `Images/` folder

## 🎯 How to Use

1. **Upload Data**: Click "Choose your dataset" and upload a CSV or Excel file
2. **Preview Data**: Review the automatic data preview and column information
3. **Describe Analysis**: In the text area, describe what insights you're looking for
4. **Configure Settings**:
   - Choose **Single Agent** or **Multi-Agent**
   - Select an AI model (GPT o3 might run into erros due to Token Per Minute Limits depending on the usage Tier of your OpenAI Developer Account)
   - Enable "Show Live Progress" to watch the analysis in real-time
5. **Run Analysis**: Click "🚀 Run Analysis" and watch the AI agents work
6. **View Results**: See the complete analysis results and any generated visualizations


## 📁 Supported File Types

- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

## ⚙️ Requirements

### System Requirements
- Python 3.7+
- Internet connection (required for AI model access)
- OpenAI API key with sufficient credits


## 📂 Project Structure

```
System-Benchmark_v2.5/
├── 📄 .env                           # Environment variables (create this)
├── 📄 .gitignore                     # Git ignore rules
├── 📄 README.md                      # Project documentation
├── 📄 requirements.txt               # Python dependencies
├── 📄 setup.py                       # Package setup configuration
├── 📄 streamlit_app.py              # 🚀 Main web application interface
│
├── 📁 Images/                        # 🖼️ Generated visualizations (auto-created)
│
└── 📁 data_science_agents/          # 🤖 Core AI agent system
    ├── 📄 __init__.py               # Package initialization
    │
    ├── 📁 agent_systems/            # 🔄 Analysis execution systems
    │   ├── 📄 __init__.py
    │   ├── 📄 multi_agent.py        # Multi-agent orchestration system
    │   └── 📄 single_agent.py       # Single comprehensive agent system
    │
    ├── 📁 config/                   # ⚙️ Configuration and settings
    │   ├── 📄 __init__.py
    │   ├── 📄 prompts.py            # AI agent instructions and prompts
    │   └── 📄 settings.py           # System configuration and model pricing
    │
    └── 📁 core/                     # 🛠️ Core system components
        ├── 📄 __init__.py
        ├── 📄 analytics.py          # Performance tracking and cost calculation
        ├── 📄 context.py            # Analysis context and state management
        ├── 📄 events.py             # Real-time streaming event system
        └── 📄 execution.py          # Python code execution engine
```

### 📋 Key Files Explained

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main web interface - start here |
| `setup.py` | **Package installer** - handles dependencies and makes the project properly installable |
| `multi_agent.py` | Orchestrates specialized AI agents for complex analysis |
| `single_agent.py` | Single AI agent handling complete analysis workflow |
| `execution.py` | Enables AI agents to execute Python code for real analysis |
| `prompts.py` | Contains all AI agent instructions and methodologies |
| `settings.py` | Model configurations and pricing information |
| `analytics.py` | Tracks performance, costs, and usage metrics |
| `events.py` | Powers real-time streaming progress updates |
| `context.py` | Manages analysis state and agent coordination |


## 💰 Cost Information

The tool uses OpenAI's API with transparent cost tracking. Typical analysis costs range from $0.01-$0.10 depending on dataset size and model choice. Cost estimates are shown in real-time during analysis.

## 🤖 Available AI Models

| Model | Best For | Cost (per 1M tokens) |
|-------|----------|---------------------|
| gpt-4o-mini | General analysis, testing | $0.15 input, $0.60 output |
| gpt-4.1-nano | Fast, economical tasks | $0.10 input, $0.40 output |
| gpt-4.1-mini | Balanced performance | $0.40 input, $1.60 output |
| o4-mini | Complex reasoning | $1.10 input, $4.40 output |
| o3-mini | Advanced STEM tasks | $1.10 input, $4.40 output |
| o3 | Most complex problems | $10.00 input, $40.00 output |


Author: Malte Staab