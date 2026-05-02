import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        return "search_long_term_memory"

    @property
    def description(self) -> str:
        return ("Searches long-term memories about the user based on a query."
                "Use this tool to find relevant information from stored memories." 
                "The query can be a question or keywords related to the information you want to retrieve." 
                "The tool returns the most relevant memories in markdown format, including content, category and topics (if present)." 
                "Use this tool when you want to recall specific information about the user that may have been stored in long-term memory."
                )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)

        result = await self.memory_store.search_memories(
            query=query,
            top_k=top_k,
            api_key=tool_call_params.api_key
        )

        if not result:
            final_result = "No memories found."
        else:
            final_result = ""
            for memory in result:
                content = memory.content
                category = memory.category
                importance = memory.importance
                topics = ", ".join(memory.topics) if memory.topics else "No topics"
                final_result += f"**Content:** {content}\n**Category:** {category}\n**Importance:** {importance}\n**Topics:** {topics}\n\n"

        stage = tool_call_params.stage
        stage.append_content(f"```text\n\r{final_result}\n\r```\n\r")

        return final_result
