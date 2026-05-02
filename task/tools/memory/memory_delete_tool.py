from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class DeleteMemoryTool(BaseTool):
    """
    Tool for deleting all long-term memories about the user.

    This permanently removes all stored memories from the system.
    Use with caution - this action cannot be undone.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "delete_long_term_memory"

    @property
    def description(self) -> str:
        return "Deletes all long-term memories about the user. This action is irreversible and should be used with caution. Use this tool when you want to permanently remove all stored memories from the system and only when user asks for it explicitly."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        result = await self.memory_store.delete_all_memories(api_key=tool_call_params.api_key)
        stage = tool_call_params.stage
        stage.append_content(result)

        return result