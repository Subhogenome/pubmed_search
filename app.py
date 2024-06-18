
import streamlit as st
import os
from Bio import Entrez
from Bio import SeqIO
import google.generativeai as genai


import time

# Provide your email to use Entrez
Entrez.email = "your-email@example.com"
GOOGLE_API_KEY="AIzaSyAimoYBCAPRKN773YUqBwokefkbt0x7Mps"
genai.configure(api_key=GOOGLE_API_KEY)
model=genai.GenerativeModel("gemini-pro")

def get_gemini_response(question,prompt):
   
    response = model.generate_content([prompt,question])
    return response

def search_pubmed(term, retmax=1000):  # Adjust retmax to 10
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

def batch_fetch_details(id_list, batch_size=300):
    all_records = []
    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        batch_ids = id_list[start:end]
        records = fetch_details(batch_ids)
        all_records.append(records)
        time.sleep(1)  # Respect rate limits by pausing between requests
    return all_records
import re
def extract_titles_and_articles(records):
    # Use regular expression to find all titles and their corresponding articles
    pattern = re.compile(r'(TI  - .*?)(?=\nTI  -|\Z)', re.DOTALL)
    matches = pattern.findall(records)
    titles_and_articles = [(re.search(r'TI  - (.*?)\n', match).group(1), match) for match in matches]
    return titles_and_articles

prompt = """
You are an expert in converting English questions to python pubmed querry!



you need to querry pubmed meshterms , abstracts , title feild , for example harmful effect of probiotics



the python querry will be like   '"probiotics"[All Fields] AND "adverse effects"[All Fields] AND (side effect* OR complication*)

or for example how is Lactobacillus related to human immunity ,
 query will be "Lactobacillus"[All Fields] AND "immunity"[All Fields] AND TLR[All Fields] and IgA[All Fields] AND  cytokine[All Fields] AND  "humans"[All Fields]'


we only need the code as an output

     
"""
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

def batch_fetch_details(id_list, batch_size=200):
    all_records = []
    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        batch_ids = id_list[start:end]
        records = fetch_details(batch_ids)
        all_records.append(records)
        time.sleep(1)  # Respect rate limits by pausing between requests
    return all_records
# Streamlit UI
st.title("PubMed Article Search")

st.subheader("Ask a question and I will give you relevant articles  from Pubmed")
# Input fields for the two words
question = st.text_input("Ask a question")


if st.button("Search"):
   response=get_gemini_response(question,prompt)

   new='({})'  # You want to format this string
   search_term = new.format(response.text)  # Format the string with response.text
  
   id_list = search_pubmed(search_term, retmax=1000)
   

   if id_list:
        st.write(" I found {} articles matching your question".format(len(id_list)))
        all_articles = batch_fetch_details(id_list)
        for articles in all_articles:
            titles_and_articles = extract_titles_and_articles(articles)
            for article in titles_and_articles:
    
                st.subheader(article[1])  # Title
                st.write("------")
                st.write(article[0])
   else:
    print("No articles found.")
