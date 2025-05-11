
import streamlit as st
import os
from Bio import Entrez
from Bio import SeqIO
import google.generativeai as genai

from groq import Groq
import time
from langchain_groq import ChatGroq
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

examples_grpah = [
    {"input": "LID - cmaf026 [pii] LID - 10.1093/fampra/cmaf026 [doi] AB - BACKGROUND: Hospital family physicians are recognized for their excellence in managing complex issues. This study aimed to reveal the level of complexities of patients referred to hospital family physicians by community centers, and the degree of change in these complexities following care provided by a health care team that includes hospital family physicians. METHODS: A retrospective cohort analysis. Patients introduced by community centers between 2020 and 2023 were identified. The patients received team-based comprehensive care. Complexity was calculated before and after the care, using the patient-centered assessment method (PCAM), which evaluates 12 items across four domains. Each item is rated from 1 to 4, yielding a total score range of 12 to 48. Pre- and post-intervention scores were compared using paired-sample t-tests, with standardized mean difference calculated using Hedges' g. RESULTS: Of 41 referred patients, three died shortly after the initial consultation. Among the 38 remaining patients, 24 were admitted, and 14 were treated as outpatients. The mean PCAM score significantly decreased from 36.9 to 23.7 after interventions (P < .001, Hedges' g = 2.54). Scores improved significantly across all domains: health and well-being (2.96 vs 1.95; P < .001, g = 2.00), social environment (3.09 vs 1.96; P < .001, g = 2.38), health literacy and communication (2.78 vs 2.46; P < .001, g = 0.67), and service coordination (3.61 vs 1.57; P < .001, g = 4.68). CONCLUSION: Hospital family physicians in Japan often manage patients with exceptionally complex problems and improve patient outcomes across multiple domains. CI - (c) The Author(s) 2025. Published by Oxford University Press. All rights reserved. For commercial re-use, please contact reprints@oup.com for reprints and translation rights for reprints. All other permissions can be obtained through our RightsLink service via the Permissions link on the article page on our site-for further information please contact journals.permissions@oup.com. FAU - Mizumoto, Junki AU - Mizumoto J AUID- ORCID: 0000-0002-0783-7351 AD - Center for General Medicine Education, School of Medicine, Keio University, Tokyo, Japan. AD - Department of Family Practice, Ehime Seikyo Hospital, Matsuyama, Japan. FAU - Hironaka, Yumiko AU - Hironaka Y AD - Regional medical coordination office, Ehime Seikyo Hospital, Matsuyama, Japan. FAU - Fujikawa, Hirohisa AU - Fujikawa H AUID- ORCID: 0000-0002-8195-1267 AD - Center for General Medicine Education, School of Medicine, Keio University, Tokyo, Japan. LA - eng GR - 24K23757/KAKENHI/ PT - Journal Article PL - England TA - Fam Pract JT - Family practice JID - 8500875 SB - IM MH - Humans MH - Retrospective Studies MH - *Referral and Consultation/statistics & numerical data MH - Japan MH - Male MH - Female MH - Middle Aged MH - Aged MH - *Physicians, Family MH - Patient-Centered Care MH - *Community Health Centers MH - Adult OTO - NOTNLM OT - delivery of health care OT - family medicine OT - geriatrics OT - patient-centered care EDAT- 2025/05/11 00:46 MHDA- 2025/05/11 00:47 CRDT- 2025/05/10 11:43 PHST- 2025/05/11 00:47 [medline] PHST- 2025/05/11 00:46 [pubmed] PHST- 2025/05/10 11:43 [entrez] AID - 8128316 [pii] AID - 10.1093/fampra/cmaf026 [doi] PST - ppublish SO - Fam Pract. 2025 Apr 12;42(3):cmaf026. doi: 10.1093/fampra/cmaf026.", "output": """
CREATE (human:Species {name: "Human"})
CREATE (ape:Species {name: "Ape"})
CREATE (ancestor:Species {name: "Common Ancestor", type: "Primate"})
CREATE (evolution:Process {type: "Evolution"})

MERGE (human)-[:EVOLVED_FROM]->(ancestor)
MERGE (ape)-[:EVOLVED_FROM]->(ancestor)
MERGE (evolution)-[:LINKS]->(human)
MERGE (evolution)-[:LINKS]->(ape)
//added with clause
WITH 1 as dummy //Adding a WITH clause to transition
UNWIND ["Genetic Similarity", "Tool Use", "Social Behavior", "Communication"] AS t
CREATE (trait:Trait {name: t})
MERGE (human)-[:SHARES_TRAIT]->(trait)
MERGE (ape)-[:SHARES_TRAIT]->(trait)
"""},
    {"input": "PG - e70092 LID - 10.1002/gps.70092 [doi] AB - OBJECTIVES: Despite established links between apathy, cardiovascular disease, and dementia, it remains unclear if cardiovascular risk factors (CVRF) play a mediating role in the association between apathy and dementia. If apathy increases dementia risk via lifestyle-related dementia risk factors, targeted lifestyle interventions could help high-risk individuals. METHODS: We used data from the preDIVA study including 3303 individuals aged 70-78 years. Apathy was assessed using the geriatric depression scale, and CVRF (cardiovascular risk factors) (systolic blood pressure, cholesterol, diabetes, body mass index (BMI), smoking, and physical activity) were considered as potential mediators. Outcome was incident dementia during 12 years of follow-up. We assessed mediation using Multiple Mediation Analysis (MMA). RESULTS: Of the association between apathy and dementia (HR 1.49 [95% CI 0.99-2.41]), 27% was mediated by physical inactivity, BMI and diabetes combined. Of this total, physical inactivity mediated 28% of the effect (HR 1.12, 95% CI 1.03-1.29), diabetes 9% of the effect (HR 1.04, 95% CI 1.02-1.10), and BMI counteracted these effects by -12% (HR 0.95, 95% CI 0.88-0.98). CONCLUSION: The relationship between apathy and dementia is partly mediated by physical inactivity, BMI and diabetes. Apathy is an important clinical marker that signals the existence of potentially modifiable pathways, providing an opportunity for lifestyle interventions. To potentially reduce dementia risk via lifestyle modification in patients with apathy, a tailored approach should be taken to overcome the characterizing symptom of diminished motivation. CI - (c) 2025 The Author(s). International Journal of Geriatric Psychiatry published by John Wiley & Sons Ltd. FAU - Lindhout, Josephine E AU - Lindhout JE AUID- ORCID: 0000-0002-6470-4344 AD - Department of Public and Occupational Health, Amsterdam UMC, Amsterdam, the Netherlands. AD - Department of General Practice, Amsterdam UMC, Amsterdam, the Netherlands. FAU - van Dalen, Jan Willem AU - van Dalen JW AD - Department of Neurology, Amsterdam UMC, Amsterdam, the Netherlands. AD - Department of Neurology, Donders Centre for Brain, Behaviour and Cognition, Radboud University Medical Center, Nijmegen, the Netherlands. FAU - van Gool, Willem A AU - van Gool WA AD - Department of Public and Occupational Health, Amsterdam UMC, Amsterdam, the Netherlands. FAU - Richard, Edo AU - Richard E AD - Department of Public and Occupational Health, Amsterdam UMC, Amsterdam, the Netherlands. AD - Department of Neurology, Donders Centre for Brain, Behaviour and Cognition, Radboud University Medical Center, Nijmegen, the Netherlands. FAU - Hoevenaar-Blom, Marieke P AU - Hoevenaar-Blom MP AD - Department of Public and Occupational Health, Amsterdam UMC, Amsterdam, the Netherlands. AD - Department of General Practice, Amsterdam UMC, Amsterdam, the Netherlands. LA - eng GR - 05-234/Dutch Innovation Fund of Collaborative Health Insurances/ GR - wE.09-2017-08/Alzheimer Nederland/ GR - 62000015/Netherlands Organization for Health Research and Development/ GR - 50-50110-98-020/Dutch Ministry of Health, Welfare and Sports/ PT - Journal Article PL - England TA - Int J Geriatr Psychiatry JT - International journal of geriatric psychiatry JID - 8710629 SB - IM MH - Humans MH - *Apathy MH - Aged MH - Male MH - *Dementia/epidemiology/psychology MH - Female MH - Body Mass Index MH - Mediation Analysis MH - *Heart Disease Risk Factors MH - Risk Factors MH - Sedentary Behavior MH - *Cardiovascular Diseases/epidemiology/psychology MH - Incidence OTO - NOTNLM OT - apathy OT - dementia OT - multiple mediation analysis OT - prevention OT - risk factor EDAT- 2025/05/11 00:46 MHDA- 2025/05/11 00:47 CRDT- 2025/05/10 11:43 PHST- 2025/05/11 00:47 [medline] PHST- 2025/05/11 00:46 [pubmed] PHST- 2025/04/23 00:00 [revised] PHST- 2024/12/06 00:00 [received] PHST- 2025/04/25 00:00 [accepted] PHST- 2025/05/10 11:43 [entrez] AID - 10.1002/gps.70092 [doi] PST - ppublish SO - Int J Geriatr Psychiatry. 2025 May;40(5):e70092. doi: 10.1002/gps.70092.", "output": """
CREATE (apathy:ClinicalSymptom {name: "Apathy"})
CREATE (dementia:Disease {name: "Dementia"})
CREATE (cvRisk:RiskFactorGroup {name: "Cardiovascular Risk Factors"})
CREATE (inactivity:RiskFactor {name: "Physical Inactivity"})
CREATE (diabetes:RiskFactor {name: "Diabetes"})
CREATE (bmi:RiskFactor {name: "BMI"})
CREATE (mma:Methodology {name: "Multiple Mediation Analysis"})
CREATE (intervention:Intervention {name: "Lifestyle Intervention"})

MERGE (apathy)-[:INCREASES_RISK_OF]->(dementia)
MERGE (apathy)-[:MEDIATED_BY {percent: 0.27}]->(cvRisk)
MERGE (cvRisk)-[:INCLUDES]->(inactivity)
MERGE (cvRisk)-[:INCLUDES]->(diabetes)
MERGE (cvRisk)-[:INCLUDES]->(bmi)

MERGE (inactivity)-[:MEDIATES {percent: 0.28}]->(dementia)
MERGE (diabetes)-[:MEDIATES {percent: 0.09}]->(dementia)
MERGE (bmi)-[:COUNTERACTS {percent: -0.12}]->(dementia)

MERGE (mma)-[:USED_IN]->(:Study {name: "preDIVA Study"})
MERGE (apathy)-[:SUGGESTS]->(intervention)
MERGE (intervention)-[:TARGETS]->(inactivity)
MERGE (intervention)-[:TARGETS]->(diabetes)
"""}

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

prefix_graph = (
    "You are an expert in converting PubMed articles into Cypher queries for building a knowledge graph in Neo4j.\n"
    "You must extract core biomedical entities and their relationships based on the article abstract, title, and keywords.\n"
    "Create Cypher queries using nodes (representing concepts like diseases, risk factors, behaviors) and relationships (e.g., INCREASES_RISK_OF, MEDIATED_BY, ASSOCIATED_WITH).\n"
    "Only return the Cypher query string as the answer — no extra explanation or formatting.\n"
    "Use MERGE to ensure uniqueness of nodes and relationships.\n"
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


few_shot_prompt_graph = FewShotPromptTemplate(
    examples=examples_grpah,
    example_prompt=example_prompt,
    prefix=prefix_graph ,
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)




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
   with st.expander("🔍 See Evidence"): 
    id_list = search_pubmed(output_text, retmax=10)
   

    if id_list:
   
        # Fetch detailed article data from PubMed
        all_articles = batch_fetch_details(id_list)
        formatted_prompt_graph = few_shot_prompt_graph.format(input=str(all_articles))
        response_g = model.invoke(formatted_prompt_graph)
        cypher_query = response_g.content.strip() 
        run_query(cypher_query)
        # Extract and display article titles and abstracts
        for articles in all_articles:
            titles_and_articles = extract_titles_and_articles(articles)
            for abstract, title in titles_and_articles:
                st.subheader(title)  # Display title
                st.markdown("---")   # Horizontal separator
                st.write(abstract)   # Display abstract
    else:
     st.info("No articles found matching your query.")
