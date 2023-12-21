from gpt4all import GPT4All

available_models = [
    "all-MiniLM-L6-v2-f16.gguf",
    "mpt-7b-chat-merges-q4_0.gguf",
    "nous-hermes-llama2-13b.Q4_0.gguf",
    "orca-mini-3b-gguf2-q4_0.gguf",
    "replit-code-v1_5-3b-q4_0.gguf",
    "gpt4all-falcon-q4_0.gguf"
]

model = GPT4All(available_models[5], device="cpu", verbose=True)
prompt = """
Your goal is to structure the user\'s query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{
    "query": string \\ text string to compare to document contents
    "filter": string \\ logical condition statement for filtering documents
}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | gt | gte | lt | lte | contain | like): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.

<< Example 1. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of "pop", "rock" or "rap""
        }
    }
}
```

User Query:
What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre

Structured Request:
```json
{
    "query": "teenager love",
    "filter": "and(or(eq(\\"artist\\", \\"Taylor Swift\\"), eq(\\"artist\\", \\"Katy Perry\\")), lt(\\"length\\", 180), eq(\\"genre\\", \\"pop\\"))"
}
```


<< Example 2. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of "pop", "rock" or "rap""
        }
    }
}
```

User Query:
What are songs that were not published on Spotify

Structured Request:
```json
{
    "query": "",
    "filter": "NO_FILTER"
}
```


<< Example 3. >>
Data Source:
```json
{
    "content": "The content of the research paper",
    "attributes": {
    "id": {
        "description": "The identifier of the research paper",
        "type": "string"
    },
    "submitter": {
        "description": "The submitters of the research paper (don\'t use the operator EQ here)",
        "type": "string or list of strings"
    },
    "authors": {
        "description": "The authors of the research paper (don\'t use the operator EQ here)",
        "type": "string"
    },
    "title": {
        "description": "The title of the research paper",
        "type": "string"
    },
    "comments": {
        "description": "Comments related to the research paper (don\'t use the operator EQ here)",
        "type": "string"
    },
    "journal-ref": {
        "description": "Journal reference of the research paper (don\'t use the operator EQ here)",
        "type": "string"
    },
    "doi": {
        "description": "Digital Object Identifier of the research paper",
        "type": "string"
    },
    "report-no": {
        "description": "Report number of the research paper",
        "type": "string"
    },
    "categories": {
        "description": "Categories or topics related to the research paper",
        "type": "string"
    },
    "license": {
        "description": "License information for the research paper",
        "type": "string"
    },
    "abstract": {
        "description": "Abstract of the research paper",
        "type": "string"
    },
    "versions": {
        "description": "Versions and creation information of the research paper",
        "type": "string (formatted as an array of objects)"
    },
    "update_date": {
        "description": "Date when the research paper was last updated",
        "type": "string (formatted as date)"
    },
    "authors_parsed": {
        "description": "Parsed information about the authors",
        "type": "string (formatted as an array of arrays)"
    }
}
}
```

User Query:
give me the id of the document with roi "10.1016/j.cpc.2007.05.015"
"""
with model.chat_session():
    response = model.generate(prompt=prompt, temp=0)
    print(response)