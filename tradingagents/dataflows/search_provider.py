from google import genai
from google.genai.types import ThinkingConfig, Tool, GenerateContentConfig, GoogleSearch
from openai import OpenAI
from abc import ABC, abstractmethod
from ..utils.text_utils import trim_llm_invocation_result


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str) -> str:
        pass


class GoogleSearchProvider(SearchProvider):
    def __init__(
        self,
        model: str,
        thinking_budget: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = 0.7,
    ):
        self.client = genai.Client()
        self.model = model
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.temperature = temperature

    def search(self, query: str) -> str:
        google_search_tool = Tool(
            google_search=GoogleSearch()
        )

        config_params = {
            "tools": [google_search_tool],
            "response_modalities": ["TEXT"],
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.thinking_budget is not None:
            config_params["thinking_config"] = ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=query
            + " If no data found is in your knowledge case, proceed immediately with Google Search and do not simulate web searches."
            " If you have additional sources to search, please do so and include them in the report."
            " You don't need to ask for additional information, just search the web and return the results."
            " Avoid duplicate whitespaces and random characters in the response.",
            config=GenerateContentConfig(**config_params),
        )

        result_text = ""
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if (
                    hasattr(candidate, "content")
                    and hasattr(candidate.content, "parts")
                    and candidate.content.parts is not None
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            result_text += trim_llm_invocation_result(part.text)
        except Exception as e:
            print(f"Error extracting text from GoogleSearchProvider response: {e}")
            result_text = ""

        return result_text if result_text else "No search results found."
    

class OpenAISearchProvider(SearchProvider):
    def __init__(self, model: str, backend_url: str):
        self.client = OpenAI(base_url=backend_url)
        self.model = model
    
    def search(self, query: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": query
                        }
                    ],
                }
            ],
            text={"format": {"type": "text"}},
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "low",
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )

        return response.output[1].content[0].text