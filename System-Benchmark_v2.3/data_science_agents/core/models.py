"""
data_science_agents/core/models.py - Pydantic models for structured agent communication
"""
from pydantic import BaseModel
from typing import Optional


class AgentResult(BaseModel):
    """Structured result returned by each CRISP-DM phase agent"""
    phase: str
    summary: str
    data_variables: dict[str, str]
    key_findings: dict[str, str]
    images_created: list[str]
    next_phase_recommendation: str
    input_file_used: str
    output_file_created: str = ""


class AnalysisMetrics(BaseModel):
    """Simple metrics for tracking analysis performance"""
    total_duration: float
    agent_durations: dict[str, float]
    tool_calls: int
    phases_completed: int
    estimated_cost: float


class AnalysisResults(BaseModel):
    """Complete results from a data science analysis"""
    final_output: str
    metrics: AnalysisMetrics
    created_images: list[str]
    agent_results: list[AgentResult]
    analysis_type: str
    success: bool
    error_message: Optional[str] = None