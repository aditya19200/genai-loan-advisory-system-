#  GenAI Loan Advisory System

An Explainable Generative AI-based system for loan decision support that combines Machine Learning, Explainable AI, and LLMs to provide transparent, user-friendly financial insights.

---

##  Overview

This project predicts loan approval using a machine learning model and generates human-readable explanations, financial advice, and counter-offers using Generative AI.

The system follows a hybrid architecture:
- ML for decision-making
- GenAI for explanation and advisory (on-demand)

---

##  Features

- ✅ Loan Approval Prediction (XGBoost)
- 📊 Risk Score Calculation
- 🔍 Explainability using SHAP
- 📚 RAG-based Regulatory Justification (RBI Guidelines)
- 💬 Sentiment-aware responses
- 🤖 LLM-generated:
  - Explanation
  - Financial Advisory
  - Counter-offers (if applicable)
- ⚡ Lazy GenAI (runs only when needed)

---

##  Architecture

### 🔹 Decision Layer (Always Runs)
- XGBoost model predicts:
  - Approved / Rejected
  - Risk score

### 🔹 Generative Layer (On-Demand)
Triggered when:
- User asks for explanation
- OR risk is borderline

Includes:
- SHAP (feature importance)
- RAG (E5 + Qdrant)
- Sentiment Analysis
- LLM (Gemini / Qwen)

---

##  System Flow

1. User inputs loan details  
2. XGBoost predicts decision  
3. Show result immediately  

If needed:
4. SHAP explains decision  
5. RAG retrieves RBI guidelines  
6. Sentiment detected  
7. LLM generates:
   - Explanation
   - Advisory
   - Counter-offer  

---

##  Tech Stack

### Backend
- FastAPI / Flask

### Machine Learning
- XGBoost
- SHAP

### RAG
- E5 Embedding Model
- Qdrant Vector Database

### LLM
- Gemini API (Google AI Studio) / Qwen

---

## Dataset

- German Credit Dataset (Kaggle)
- Optional: Synthetic data for augmentation

---

## Key Design Principles

- Separation of decision (ML) and explanation (LLM)
- On-demand GenAI (cost efficient)
- Grounded responses using RAG
- Modular architecture

---

## 📌Example Output

**Initial Output:**
