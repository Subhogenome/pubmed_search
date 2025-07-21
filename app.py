### app.py (modularized PubMed-KG project)
import streamlit as st
import time
import matplotlib.pyplot as plt
import networkx as nx
import re

from Bio import Entrez
from neo4j import GraphDatabase
from langchain.chains import LLMChain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq

# === Secrets and Config ===
Entrez.email = "chatterjeesubhodeep08@gmial.com"
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
api = st.secrets["api"]
prompt_template_str = st.secrets["prompt"]

# === LLM Setup ===
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api)
prompt_template = PromptTemplate(input_variables=["abstract", "question"], template=prompt_template_str)
chain = LLMChain(llm=llm, prompt=prompt_template)

# === Few-shot Examples ===
examples = [
    {"input": "harmful effect of probiotics", "output": '"probiotics"[All Fields] AND "adverse effects"[All Fields] AND (side effect* OR complication*)'},
    {"input": "how is Lactobacillus related to human immunity", "output": '"Lactobacillus"[All Fields] AND ("immunity"[All Fields] OR TLR[All Fields] and IgA[All Fields] OR  cytokine[All Fields]) OR  "humans"[All Fields]'},
    {"input": "Human", "output": '"Human"[All Fields] OR "Homo Sapeins"[All Fields] OR "Humans"'}
]

example_prompt = PromptTemplate(input_variables=["input", "output"], template="Q: {input}\nA: {output}")
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are an expert in converting English questions to PubMed-compatible queries.\nReturn only the query string as the answer.\nHere are some examples:",
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

# === Neo4j Driver ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def clear_db():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def run_cypher_in_neo4j(query):
    with driver.session() as session:
        try:
            session.run(query)
        except Exception as e:
            st.error(f"❌ Failed to execute query: {e}")

def fetch_graph():
    query = """
    MATCH (n)-[r]->(m)
    RETURN coalesce(n.name, n.title) AS from, type(r) AS rel, coalesce(m.name, m.title) AS to
    """
    with driver.session() as session:
        return list(session.run(query))

def build_networkx_graph(records):
    G = nx.DiGraph()
    for record in records:
        G.add_edge(record["from"], record["to"], label=record["rel"])
    return G

# === PubMed Functions ===
def search_pubmed(term, retmax=500):
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

def fetch_details(id_list):
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = handle.read()
    handle.close()
    return records

def batch_fetch_details(id_list, batch_size=10):
    all_records = []
    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        batch_ids = id_list[start:end]
        records = fetch_details(batch_ids)
        all_records.append(records)
        time.sleep(1)
    return all_records

def extract_titles_and_articles(records):
    pattern = re.compile(r'(TI  - .*?)(?=\nTI  -|\Z)', re.DOTALL)
    matches = pattern.findall(records)
    return [(re.search(r'TI  - (.*?)\n', match).group(1), match) for match in matches]

def generate_cypher_query(abstract, question):
    return chain.run(abstract=abstract, question=question).strip()

# === Streamlit App ===
st.title("PubMed.AI")
st.subheader("Ask a question and get relevant articles visualized as a knowledge graph")

question = st.text_input("Ask a question")

if st.button("Search") and question:
    clear_db()
    formatted_prompt = few_shot_prompt.format(input=question)
    response = llm.invoke(formatted_prompt)
    search_query = response.content.strip()
    st.write(search_query)

    id_list = search_pubmed(search_query, retmax=10)
    all_articles = batch_fetch_details(id_list)

    cypher_query = generate_cypher_query(question, str(all_articles))
    if cypher_query.startswith('"""') and cypher_query.endswith('"""'):
        run_cypher_in_neo4j(cypher_query[3:-3])
    elif cypher_query.startswith('```') and cypher_query.endswith('```'):
        run_cypher_in_neo4j(cypher_query[3:-3])
    else:
        run_cypher_in_neo4j(cypher_query)

    records = fetch_graph()
    G = build_networkx_graph(records)
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis('off')
    st.pyplot(plt)

    with st.expander("🔍 See Evidence"):
        for articles in all_articles:
            titles_and_articles = extract_titles_and_articles(articles)
            for abstract, title in titles_and_articles:
                st.subheader(title)
                st.markdown("---")
                st.write(abstract)
else:
    st.info("Enter a biomedical question and press Search")
