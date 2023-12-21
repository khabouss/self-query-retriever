from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.gpt4all import GPT4All
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)

vectorstore = ElasticsearchStore(
    embedding=GPT4AllEmbeddings(),
    index_name="arxiv_data",
    es_url="http://172.16.0.216:9200",
)

available_models = [
    "all-MiniLM-L6-v2-f16.gguf",
    "mpt-7b-chat-merges-q4_0.gguf",
    "nous-hermes-llama2-13b.Q4_0.gguf",
    "orca-mini-3b-gguf2-q4_0.gguf",
    "replit-code-v1_5-3b-q4_0.gguf",
]

# Add details about metadata fields
metadata_field_info = [
    AttributeInfo(
        name="id",
        description="The identifier of the research paper",
        type="string",
    ),
    AttributeInfo(
        name="submitter",
        description="The submitters of the research paper (don't use the operator EQ here)",
        type="string or list of strings",
    ),
    AttributeInfo(
        name="authors",
        description="The authors of the research paper (don't use the operator EQ here)",
        type="string",
    ),
    AttributeInfo(
        name="title",
        description="The title of the research paper",
        type="string",
    ),
    AttributeInfo(
        name="comments",
        description="Comments related to the research paper (don't use the operator EQ here)",
        type="string",
    ),
    AttributeInfo(
        name="journal-ref",
        description="Journal reference of the research paper (don't use the operator EQ here)",
        type="string",
    ),
    AttributeInfo(
        name="doi",
        description="Digital Object Identifier of the research paper",
        type="string",
    ),
    AttributeInfo(
        name="report-no",
        description="Report number of the research paper",
        type="string",
    ),
    AttributeInfo(
        name="categories",
        description="Categories or topics related to the research paper",
        type="string",
    ),
    AttributeInfo(
        name="license",
        description="License information for the research paper",
        type="string",
    ),
    AttributeInfo(
        name="abstract",
        description="Abstract of the research paper",
        type="string",
    ),
    AttributeInfo(
        name="versions",
        description="Versions and creation information of the research paper",
        type="string (formatted as an array of objects)",
    ),
    AttributeInfo(
        name="update_date",
        description="Date when the research paper was last updated",
        type="string (formatted as date)",
    ),
    AttributeInfo(
        name="authors_parsed",
        description="Parsed information about the authors",
        type="string (formatted as an array of arrays)",
    ),
]

document_content_description = "The content of the research paper"

callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model=available_models[1], callbacks=callbacks, verbose=True, temp=0
)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

retriever = SelfQueryRetriever.from_llm(
    llm_chain,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

query = input("prompt>")
r = retriever.get_relevant_documents(query)
print("DOCS8 ", r)
