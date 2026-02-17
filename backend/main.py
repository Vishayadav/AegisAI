"""
AegisAI Main API
Unified AI Observability & Governance Platform for Banks

Integrates:
- ML Monitoring (model drift, accuracy)
- LLM Monitoring (latency, tokens, hallucination)
- Risk Engine (AI Health Score, Governance)
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import monitoring modules
from llm_monitor import get_llm_metrics
from ml_monitor import get_ml_metrics
from risk_engine import evaluate_governance, explain_score_breakdown

# ============================
# FastAPI Application Setup
# ============================

app = FastAPI(
    title="AegisAI - AI Governance Platform",
    description="Unified monitoring and governance for ML + LLM systems in banking",
    version="0.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# Response Models
# ============================

class MLMetricsResponse(BaseModel):
    drift: float
    accuracy: float
    status: str


class LLMMetricsResponse(BaseModel):
    latency: float
    tokens: int
    hallucination: int


class GovernanceResponse(BaseModel):
    ai_health_score: float
    risk_level: str
    alerts: list


class DetailedGovernanceResponse(BaseModel):
    ai_health_score: float
    risk_level: str
    alerts: list
    breakdown: dict


# ============================
# Health Check Endpoint
# ============================

@app.get("/", tags=["System"])
def root():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "AegisAI - AI Governance Platform",
        "version": "0.1.0"
    }


# ============================
# ML Monitoring Endpoints
# ============================

@app.get("/api/ml-metrics", response_model=MLMetricsResponse, tags=["ML Monitoring"])
def get_ml_metrics_endpoint(trigger_drift: bool = Query(False, description="Simulate model drift for testing")):
    """
    Get ML model monitoring metrics.
    
    Returns:
    - drift: Data distribution drift score (0-1)
    - accuracy: Model prediction accuracy (0-1)
    - status: Model status (stable/drift_detected/error)
    """
    metrics = get_ml_metrics(trigger_drift=trigger_drift)
    return metrics


# ============================
# LLM Monitoring Endpoints
# ============================

@app.get("/api/llm-metrics", response_model=LLMMetricsResponse, tags=["LLM Monitoring"])
def get_llm_metrics_endpoint(trigger_attack: bool = Query(False, description="Simulate hallucination/unsafe response for testing")):
    """
    Get LLM system monitoring metrics.
    
    Returns:
    - latency: Response time in seconds
    - tokens: Tokens consumed in the response
    - hallucination: Binary flag (0=safe, 1=unsafe/hallucinated)
    """
    metrics = get_llm_metrics(trigger_attack=trigger_attack)
    return metrics


# ============================
# Integrated Governance Endpoints
# ============================

@app.get("/api/governance", response_model=GovernanceResponse, tags=["Governance"])
def get_governance(
    trigger_drift: bool = Query(False, description="Simulate ML drift"),
    trigger_attack: bool = Query(False, description="Simulate LLM hallucination")
):
    """
    Get unified AI Governance assessment combining ML + LLM risks.
    
    Returns:
    - ai_health_score: Overall health score (0-100)
    - risk_level: Risk classification (STABLE/MONITORING/ELEVATED RISK/CRITICAL)
    - alerts: List of active governance alerts
    """
    ml_metrics = get_ml_metrics(trigger_drift=trigger_drift)
    llm_metrics = get_llm_metrics(trigger_attack=trigger_attack)
    
    governance = evaluate_governance(ml_metrics, llm_metrics)
    return governance


@app.get("/api/governance/detailed", response_model=DetailedGovernanceResponse, tags=["Governance"])
def get_governance_detailed(
    trigger_drift: bool = Query(False, description="Simulate ML drift"),
    trigger_attack: bool = Query(False, description="Simulate LLM hallucination")
):
    """
    Get detailed AI Governance assessment with score breakdown.
    
    Returns complete governance data plus:
    - breakdown: Individual contribution of each risk factor to the overall score
    """
    ml_metrics = get_ml_metrics(trigger_drift=trigger_drift)
    llm_metrics = get_llm_metrics(trigger_attack=trigger_attack)
    
    governance = evaluate_governance(ml_metrics, llm_metrics)
    breakdown = explain_score_breakdown(ml_metrics, llm_metrics)
    
    return {
        **governance,
        "breakdown": breakdown
    }


# ============================
# System Status Dashboard
# ============================

@app.get("/api/dashboard", tags=["Dashboard"])
def get_dashboard_snapshot(
    trigger_drift: bool = Query(False),
    trigger_attack: bool = Query(False)
):
    """
    Get complete system snapshot for dashboard visualization.
    Combines all metrics from ML, LLM, and governance systems.
    """
    ml_metrics = get_ml_metrics(trigger_drift=trigger_drift)
    llm_metrics = get_llm_metrics(trigger_attack=trigger_attack)
    governance = evaluate_governance(ml_metrics, llm_metrics)
    breakdown = explain_score_breakdown(ml_metrics, llm_metrics)
    
    return {
        "ml_monitoring": ml_metrics,
        "llm_monitoring": llm_metrics,
        "governance": governance,
        "score_breakdown": breakdown,
        "timestamp": "2026-02-18"  # Can be enhanced with real timestamps
    }


# ============================
# Testing/Demo Endpoints
# ============================

@app.post("/api/test/scenario/normal", tags=["Testing"])
def test_normal_scenario():
    """
    Test scenario: Normal operation - all systems healthy.
    """
    ml_metrics = get_ml_metrics(trigger_drift=False)
    llm_metrics = get_llm_metrics(trigger_attack=False)
    governance = evaluate_governance(ml_metrics, llm_metrics)
    
    return {
        "scenario": "Normal Operation",
        "ml_metrics": ml_metrics,
        "llm_metrics": llm_metrics,
        "governance": governance
    }


@app.post("/api/test/scenario/ml-drift", tags=["Testing"])
def test_ml_drift_scenario():
    """
    Test scenario: ML model drift detected.
    """
    ml_metrics = get_ml_metrics(trigger_drift=True)
    llm_metrics = get_llm_metrics(trigger_attack=False)
    governance = evaluate_governance(ml_metrics, llm_metrics)
    
    return {
        "scenario": "ML Drift Detected",
        "ml_metrics": ml_metrics,
        "llm_metrics": llm_metrics,
        "governance": governance
    }


@app.post("/api/test/scenario/llm-attack", tags=["Testing"])
def test_llm_attack_scenario():
    """
    Test scenario: LLM hallucination/unsafe response detected.
    """
    ml_metrics = get_ml_metrics(trigger_drift=False)
    llm_metrics = get_llm_metrics(trigger_attack=True)
    governance = evaluate_governance(ml_metrics, llm_metrics)
    
    return {
        "scenario": "LLM Hallucination Detected",
        "ml_metrics": ml_metrics,
        "llm_metrics": llm_metrics,
        "governance": governance
    }


@app.post("/api/test/scenario/critical", tags=["Testing"])
def test_critical_scenario():
    """
    Test scenario: Critical - Both ML drift AND LLM attack.
    """
    ml_metrics = get_ml_metrics(trigger_drift=True)
    llm_metrics = get_llm_metrics(trigger_attack=True)
    governance = evaluate_governance(ml_metrics, llm_metrics)
    
    return {
        "scenario": "CRITICAL - ML Drift + LLM Attack",
        "ml_metrics": ml_metrics,
        "llm_metrics": llm_metrics,
        "governance": governance
    }


# ============================
# Run Application
# ============================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
