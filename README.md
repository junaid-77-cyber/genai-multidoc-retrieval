## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
With the growing volume of research papers, manually extracting relevant information from multiple sources is time-consuming. The goal is to create an automated agent that can:

1.Retrieve relevant information from multiple documents.

2.Summarize and synthesize insights from research papers.

3.Provide accurate and concise answers to user queries using LlamaIndex and OpenAI LLMs.

### DESIGN STEPS:

#### Step 1:
Preparation of Documents

Collect multiple research papers from sources like OpenReview.

Save the documents locally in PDF or text format.

Maintain a mapping of paper names and file paths.

#### Step 2: 
Create Retrieval and Summary Tools

For each paper, use get_doc_tools() to generate:

Vector tool: Converts document content into embeddings for semantic search.

Summary tool: Summarizes the document content.

Store these tools in a dictionary for easy access.

#### Step 3: 
Create Object and Vector Index

Combine all tools into a single ObjectIndex.

Use VectorStoreIndex to enable semantic search over multiple documents.

This allows retrieval of relevant sections across all papers for a query.

#### Step 4: 
Initialize the LLM

Load OpenAI LLM (gpt-3.5-turbo) for query understanding and response generation.

This LLM is responsible for generating natural language responses from retrieved information.

#### Step 5: 
Set Up the Agent

Use FunctionCallingAgentWorker to link tools and LLM:

Provide the system prompt describing the task.

Ensure the agent uses only the provided tools to answer queries.

Wrap the worker in AgentRunner for executing queries.

#### Step 6: 
Query and Generate Response

Submit a query to the agent, e.g., summarizing generative AI approaches in specific papers.

The agent uses:

Vector search to find relevant sections.

Summary tools to condense the information.

LLM to synthesize a final response.

Print the response.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/forum?id=FPLNSx1jmL",
    "https://openreview.net/forum?id=y5rLR9xZpn",
    "https://openreview.net/forum?id=kiVIVBmMTP",
    "https://openreview.net/forum?id=IKJyRyHpHV",
    "https://openreview.net/forum?id=GGg2BmcBEp",
]

papers = [
    "25649_Improving_Developer_Emot.pdf",
    "25645_Quantum_Inspired_Image_E.pdf",
    "25642_SAVIOR_Sample_efficient_.pdf",
    "25639_Revisiting_Multilingual_.pdf",
    "25633_One_Shot_Style_Personali.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt="""\You are an agent designed to answer queries over a set of papers from the OpenReview 2026 conference.
Always use the provided tools to retrieve information from the papers. Do not rely on prior knowledge.
Focus on generative AI topics when relevant.
""",
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
 "Summarize the generative AI approaches discussed in the papers on quantum-inspired image encoding and one-shot style personalization from the 2026 OpenReview conference."
)
print(str(response))
```
### OUTPUT:
![alt text](<Screenshot 2025-10-24 115117.png>)
![alt text](<Screenshot 2025-10-24 115157.png>)

### RESULT:
The multidocument retrieval agent using LlamaIndex was successfully implemented. It efficiently extracted and summarized information from multiple research articles and provided accurate, concise, and relevant responses for diverse queries.