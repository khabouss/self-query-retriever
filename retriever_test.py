from llm import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

available_models = [
    "all-MiniLM-L6-v2-f16.gguf",
    "mpt-7b-chat-merges-q4_0.gguf",
    "nous-hermes-llama2-13b.Q4_0.gguf",
    "orca-mini-3b-gguf2-q4_0.gguf",
    "replit-code-v1_5-3b-q4_0.gguf",
    "gpt4all-falcon-q4_0.gguf",
]

metadata_field_info = [
    AttributeInfo(
        name="id",
        description="The identifier of the song",
        type="string",
    ),
    AttributeInfo(
        name="singers",
        description="The singers of the song",
        type="string or list of strings",
    ),
    AttributeInfo(
        name="authors",
        description="The authors of the song",
        type="string",
    ),
    AttributeInfo(
        name="title",
        description="The title of the song",
        type="string",
    )
]

SelfQueryRetriever(
    available_models[2], metadata_field_info, "Who sang the song Baby?"
)
