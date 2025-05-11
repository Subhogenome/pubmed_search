
import streamlit as st
import os
from Bio import Entrez
from Bio import SeqIO
import google.generativeai as genai

from groq import Groq
import time
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
NEO4J_URI = "neo4j+s://a0bc7ea0.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "1kySNKajMIyVuTOq_gkDVlu4RH0kXvIOJVzqU6PFsQM" 
# Provide your email to use Entrez
Entrez.email = "chatterjeesubhodeep08@gmial.com"
api ="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"
model =ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=api)
def clear_db(tx):
    tx.run("MATCH (n) DETACH DELETE n")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
def run_query(cypher_query):
    with driver.session() as session:
        session.run(cypher_query)
    print("Graph created.")
examples = [
    {"input": "harmful effect of probiotics", "output": '"probiotics"[All Fields] AND "adverse effects"[All Fields] AND (side effect* OR complication*)'},
    {"input": "how is Lactobacillus related to human immunity", "output": '"Lactobacillus"[All Fields] AND ("immunity"[All Fields] OR TLR[All Fields] and IgA[All Fields] OR  cytokine[All Fields]) OR  "humans"[All Fields]'},
    {"input": "Human", "output": '"Human"[All Fields] OR "Homo Sapeins"[All Fields] OR "Humans"'}
]




# How to format each example
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

# Corrected: Single string as prefix
prefix = (
    "You are an expert in converting English questions to Python PubMed query.\n"
    "You must create PubMed-compatible queries using MeSH terms, abstract, and title fields.\n"
    "Only return the query string as the answer, no explanation or extra text.\n"
    "Here are some examples:"
)


# Create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)


prompt_template = PromptTemplate(
    input_variables=["abstract"],
    template="""
You are a biomedical knowledge graph engineer. Given a PubMed article abstract, output ONLY the Neo4j Cypher query that:
- Extracts key biomedical entities (e.g. physicians, patients, scores, institutions, interventions).
- Establishes relationships (edges) based on the context.
- Represents it as a hypergraph using Neo4j Cypher syntax.
- Output only the Cypher query no extra text like " Here is the Neo4j Cypher query:".
-  Only return the query string as the answer, no explanation or extra text

Example 1:
Abstract:
Hospital family physicians manage complex cases. This study assesses patient complexity before and after care by a health team using PCAM. 38 patients were studied; 24 admitted, 14 treated as outpatients. Complexity dropped from 36.9 to 23.7. Scores improved across domains: health, social, communication, coordination.

Cypher Output:
CREATE (:CommunityHealthCenter {{name: 'Community Health Centers'}})
CREATE (:HospitalFamilyPhysician {{role: 'Hospital Family Physicians'}})
CREATE (:CareTeam {{type: 'Multidisciplinary Care Team'}})
CREATE (:Patient {{status: 'Referred'}})
...
Abstract:
{abstract}
"""

)

chain = LLMChain(llm=model, prompt=prompt_template)


def fetch_graph(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS from, type(r) AS rel, m.name AS to
    """
    return list(tx.run(query))

def build_networkx_graph(records):
    G = nx.DiGraph()
    for record in records:
        G.add_edge(record["from"], record["to"], label=record["rel"])
    return G

# Fetch data


# Build graph


def generate_cypher_query(abstract: str) -> str:
    return chain.run(abstract=abstract).strip()

def run_cypher_in_neo4j(cypher_query: str):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        try:
            session.run(cypher_query)
            print("✅ Query successfully executed in Neo4j.")
        except Exception as e:
            print("❌ Failed to execute query:", e)
        finally:
            driver.close()



import re
def extract_titles_and_articles(records):
    # Use regular expression to find all titles and their corresponding articles
    pattern = re.compile(r'(TI  - .*?)(?=\nTI  -|\Z)', re.DOTALL)
    matches = pattern.findall(records)
    titles_and_articles = [(re.search(r'TI  - (.*?)\n', match).group(1), match) for match in matches]
    return titles_and_articles

prompt=st.secrets["prompt"]
def search_pubmed(term, retmax=500):
    # Search PubMed with the specified term and return up to retmax results
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
        time.sleep(1)  # Respect rate limits by pausing between requests
    return all_records
# Streamlit UI
st.title("PubMed.AI")

st.subheader("Ask a question and I will give you relevant articles  from Pubmed")
# Input fields for the two words
question = st.text_input("Ask a question")


if st.button("Search"):
   with driver.session() as session:
    session.execute_write(clear_db)
   formatted_prompt = few_shot_prompt.format(input=question)
   response = model.invoke(formatted_prompt)
   output_text = response.content.strip() 
    # Format the string with response.text
   st.write(output_text)
   id_list = search_pubmed(output_text, retmax=10)
   all_articles = batch_fetch_details(id_list)
   
   cypher_query = generate_cypher_query(str(all_articles))
   print("🧾 Generated Cypher Query:\n", cypher_query)
   if cypher_query.startswith('"""') and cypher_query.endswith('"""'):
     converted = '"' + cypher_query[3:-3] + '"'
     #print(converted)
     run_cypher_in_neo4j(converted)
   elif cypher_query.startswith('```') and cypher_query.endswith('```'):
        run_cypher_in_neo4j(cypher_query[3:-3])
   else:
        run_cypher_in_neo4j(cypher_query)    
   with driver.session() as session:
    records = session.read_transaction(fetch_graph)
    G = build_networkx_graph(records)
  
   with st.expander("🔍 See Evidence"):

    if id_list:
   
        # Fetch detailed article data from PubMed
        

        # Extract and display article titles and abstracts
        for articles in all_articles:
            titles_and_articles = extract_titles_and_articles(articles)
            for abstract, title in titles_and_articles:
                st.subheader(title)  # Display title
                st.markdown("---")   # Horizontal separator
                st.write(abstract)   # Display abstract
    else:
     st.info("No articles found matching your query.")
