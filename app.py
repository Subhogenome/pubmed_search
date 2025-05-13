
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
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
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


from langchain.prompts import PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["abstract", "question"],
    template=st.secrets["prompt"]
)



chain = LLMChain(llm=model, prompt=prompt_template)


def fetch_graph(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN 
        coalesce(n.name, n.title) AS from, 
        type(r) AS rel, 
        coalesce(m.name, m.title) AS to
    """
    return list(tx.run(query))

def build_networkx_graph(records):
    G = nx.DiGraph()
    for record in records:
        G.add_edge(record["from"], record["to"], label=record["rel"])
    return G

# Fetch data


# Build graph


def generate_cypher_query(abstract: str, question: str) -> str:
    return chain.run(abstract=abstract, question=question).strip()

def run_cypher_in_neo4j(cypher_query: str):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        try:
            session.run(cypher_query)
           # st.write("✅ Query successfully executed in Neo4j.")
        except Exception as e:
            st.write("❌ Failed to execute query:", e)
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
   
   cypher_query = generate_cypher_query(question,str(all_articles))
   #print("🧾 Generated Cypher Query:\n", cypher_query)
   if cypher_query.startswith('"""') and cypher_query.endswith('"""'):
     converted = '"' + cypher_query[3:-3] + '"'
     #st.write(converted)
     run_cypher_in_neo4j(converted)
   elif cypher_query.startswith('```') and cypher_query.endswith('```'):
       # st.write(cypher_query[3:-3])
        run_cypher_in_neo4j(cypher_query[3:-3])
   else:
       # st.write(cypher_query)
        run_cypher_in_neo4j(cypher_query)    
   with driver.session() as session:
    records = session.read_transaction(fetch_graph)
    G = build_networkx_graph(records)
    # Draw the graph
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_labels = nx.get_edge_attributes(G, 'label')

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Neo4j Graph Visualization")
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
  
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
