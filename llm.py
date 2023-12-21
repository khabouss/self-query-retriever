from gpt4all import GPT4All
from langchain.prompts import PromptTemplate
from typing import Optional, Union
import os


class SelfQueryRetriever:
    def __init__(
        self,
        model_name: str,
        metadata_field_info,
        user_prompt,
        model_path: Optional[Union[str, os.PathLike[str]]] = None,
        model_type: Optional[str] = None,
        allow_download: bool = True,
        n_threads: Optional[int] = None,
        device: Optional[str] = "cpu",
        verbose: bool = False,
        
    ):
        """
        """

        model = GPT4All(
            model_name,
            model_path,
            model_type,
            allow_download,
            n_threads,
            device,
            verbose
        )

        """
        Example:
        
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
            ),
        ]
        """

        prompt = PromptTemplate.from_template(
        """
        Your goal is to structure the user\'s query to match the request schema provided below.

        << Structured Request Schema >>
        When responding use a markdown code snippet with a JSON object formatted in the following schema:

        ```json
        {{
            "query": text string to compare to document contents
            "filter": logical condition statement for filtering documents
        }}
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
        {{
            "content": "Lyrics of a song",
            "attributes": {{
                "artist": {{
                    "type": "string",
                    "description": "Name of the song artist"
                }},
                "length": {{
                    "type": "integer",
                    "description": "Length of the song in seconds"
                }},
                "genre": {{
                    "type": "string",
                    "description": "The song genre, one of "pop", "rock" or "rap""
                }}
            }}
        }}
        ```

        User Query:
        What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre

        Structured Request:
        ```json
        {{
            "query": "teenager love",
            "filter": "and(or(eq("artist", "Taylor Swift"), eq("artist", "Katy Perry")), lt("length", 180), eq("genre", "pop"))"
        }}
        ```


        << Example 2. >>
        Data Source:
        ```json
        {{
            "content": "Lyrics of a song",
            "attributes": {{
                "artist": {{
                    "type": "string",
                    "description": "Name of the song artist"
                }},
                "length": {{
                    "type": "integer",
                    "description": "Length of the song in seconds"
                }},
                "genre": {{
                    "type": "string",
                    "description": "The song genre, one of "pop", "rock" or "rap""
                }}
            }}
        }}
        ```

        User Query:
        What are songs that were not published on Spotify

        Structured Request:
        ```json
        {{
            "query": "",
            "filter": "NO_FILTER"
        }}
        ```


        << Example 3. >>
        Data Source:
        ```json
        {metadata_field_info}
        ```

        User Query:
        {user_prompt}
        """
        )

        with model.chat_session():
            response = model.generate(
                prompt=prompt.format(
                    metadata_field_info=metadata_field_info, user_prompt=user_prompt
                ),
                temp=0,
            )
            print(response)
