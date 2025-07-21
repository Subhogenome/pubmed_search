# 🧬 PubMed Knowledge Graph with LLM & Neo4j

A Streamlit application that converts biomedical questions into PubMed queries, fetches abstracts, and constructs a knowledge graph using LLM-generated Cypher queries in Neo4j.

## ✨ Features
- 🧠 Converts natural questions into PubMed MeSH queries (via LangChain)
- 📄 Fetches abstracts using BioPython & Entrez
- 🧵 Converts text to graph Cypher queries using LLM (Llama 4 via Groq)
- 🧠 Visualizes biomedical knowledge as graphs with NetworkX
- 🔬 Ideal for drug discovery, probiotic research, or academic exploration

## 🛠️ Tech Stack
- LangChain + Groq (Llama 4)
- Neo4j + NetworkX
- Streamlit UI
- BioPython (Entrez)
- Matplotlib

## 🚀 Getting Started

```bash
git clone https://github.com/yourname/pubmed-search.git
cd pubmed-kg
pip install -r requirements.txt
streamlit run app/app.py
