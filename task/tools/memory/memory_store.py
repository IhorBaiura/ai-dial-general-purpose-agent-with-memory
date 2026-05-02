import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache: dict[str, MemoryCollection] = {}

        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        bucket_with_app_home = await dial_client.my_appdata_home()
        return f"files/{bucket_with_app_home}/__long-memories/data.json"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        """Load memories from DIAL bucket or cache."""

        client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version="2025-01-01-preview")
        memory_file_path = await self._get_memory_file_path(client)
        if memory_file_path in self.cache:
            return self.cache[memory_file_path]
        
        try:
            response = await client.files.download(memory_file_path)
            content = response.get_content().decode('utf-8')
            data = json.loads(content)
            memories = MemoryCollection.model_validate(data)
        except Exception as e:
            print(f"Warning: Could not load memories, initializing empty collection: {e}")
            memories = MemoryCollection()

        self.cache[memory_file_path] = memories

        return memories

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version="2025-01-01-preview")
        memory_file_path = await self._get_memory_file_path(client)
        memories.updated_at = datetime.now(UTC)

        memories_json = memories.model_dump_json()
        self.cache[memory_file_path] = memories
        await client.files.upload(memory_file_path, memories_json.encode('utf-8'))
        
        print(f"Memories saved successfully to {memory_file_path}.")

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""

        memory_collection: MemoryCollection = await self._load_memories(api_key)
        memories = memory_collection.memories

        embedding = self.embedding_model.encode([content])[0].tolist()

        new_memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics
            ),
            embedding=embedding
        )

        memories.append(new_memory)
        await self._save_memories(api_key, memory_collection)

        return f"Memory added successfully with ID: {new_memory.data.id}"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        memory_collection: MemoryCollection = await self._load_memories(api_key)

        if not memory_collection.memories:
            return []
        
        if self._needs_deduplication(memory_collection):
            memory_collection = await self._deduplicate_and_save(api_key, memory_collection)

        embeddings = np.array([memory.embedding for memory in memory_collection.memories]).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        query_embedding = self.model.encode([query]).astype('float32')
        query_norm = np.linalg.norm(query_embedding, keepdims=True)
        normalized_query = query_embedding / query_norm

        k = min(top_k, len(memory_collection.memories))
        distance, indices = index.search(normalized_query, k)

        results = [memory_collection.memories[i].data for i in indices[0]]

        return results

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        if len(collection.memories) <= 10:
            return False
        
        if collection.last_deduplicated_at is None:
            return True
        
        time_since_last = datetime.now(UTC) - collection.last_deduplicated_at
        return time_since_last > timedelta(hours=self.DEDUP_INTERVAL_HOURS)

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """

        try:
            original_count = len(collection.memories)

            if original_count < 2:
                return collection

            deduplicated_memories = self._deduplicate_fast(collection.memories)

            collection.memories = deduplicated_memories
            collection.last_deduplicated_at = datetime.now(UTC)

            await self._save_memories(api_key, collection)

            removed_count = original_count - len(deduplicated_memories)
            print(f"Deduplication completed: {removed_count} duplicates removed, {len(deduplicated_memories)} memories remain.")

            return collection

        except Exception as e:
            print(f"Error during deduplication: {e}")
            return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        if len(memories) < 2:
            return memories

        embeddings = np.array([memory.embedding for memory in memories]).astype('float32')
        embedding_size = len(embeddings)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        k = min(10, embedding_size)
        distance, indices = index.search(normalized_embeddings, k)

        duplicates_set = set()

        for i in range(embedding_size):
            if i in duplicates_set:
                continue

            for j in range(1, k):
                neighbor_idx = indices[i][j]

                if neighbor_idx in duplicates_set:
                    continue

                if distance[i][j] > 0.75:
                    if memories[i].data.importance >= memories[neighbor_idx].data.importance:
                        duplicates_set.add(neighbor_idx)
                    else:
                        duplicates_set.add(i)
                        break

        deduplicated = [m for i, m in enumerate(memories) if i not in duplicates_set]
        return deduplicated

    async def delete_all_memories(self, api_key: str, ) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        try:
            client = AsyncDial(base_url=self.endpoint, api_key=api_key)
            memory_file_path = await self._get_memory_file_path(client)

            try:
                await client.files.delete(memory_file_path)
                print(f"Memory file {memory_file_path} deleted successfully.")
            except Exception as e:
                print(f"Warning: Could not delete memory file: {e}")

            if memory_file_path in self.cache:
                del self.cache[memory_file_path]
                print(f"Cache for {memory_file_path} cleared successfully.")
        except Exception as e:
            error = f"Warning: Could not delete memories: {e}"
            print(error)
            return error

        return "All memories have been deleted successfully."
