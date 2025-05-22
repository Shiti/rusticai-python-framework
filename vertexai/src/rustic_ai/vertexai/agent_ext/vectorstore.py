from typing import Optional, Dict, List

from google.cloud import aiplatform

from rustic_ai.core.agents.commons.media import Document
from rustic_ai.core.guild.agent_ext.depends.vectorstore import VectorStore, VectorSearchResults, UpsertResponse
from rustic_ai.core.guild.agent_ext.depends.vectorstore.vectorstore import DEFAULT_K
from rustic_ai.vertexai.client import VertexAIBase


class VertexAIVectorSearch(VertexAIBase, VectorStore):
    def __init__(self, project: str, location: str, index_id: str) -> None:
        super().__init__(project=project, location=location)
        self.index = aiplatform.MatchingEngineIndex(
            index_name=index_id,
            project=project,
            location=location
        )

    def upsert(self, documents: List[Document]) -> UpsertResponse:
        self.index.update_metadata()

    def delete(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        pass

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        pass

    def similarity_search(self, query: str, k: int = DEFAULT_K, metadata_filter: Optional[Dict[str, str]] = None,
                          where_documents: Optional[Dict[str, str]] = None) -> VectorSearchResults:
        pass

