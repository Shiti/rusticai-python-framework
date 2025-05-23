from typing import List, Optional

from pydantic.config import JsonDict
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from rustic_ai.core.guild.agent_ext.depends import DependencyResolver
from rustic_ai.core.guild.agent_ext.depends.embeddings import Embeddings
from rustic_ai.vertexai.client import VertexAIBase, VertexAIConf


class VertexAIEmbeddingConf(VertexAIConf):
    auto_truncate: bool = False
    task_type: Optional[str] = None
    output_dimensionality: Optional[int] = None


DEFAULT_MODEL: str = "text-embedding-005"


class VertexAIEmbeddings(VertexAIBase, Embeddings):
    def __init__(self, model: str, conf: VertexAIEmbeddingConf):
        super().__init__(conf.project_id, conf.location, conf.credentials)
        self.conf = conf
        self.model = TextEmbeddingModel.from_pretrained(model)

    def embed(self, text: List[str]) -> List[List[float]]:
        print(f"\ntokens according to model are {self.model.count_tokens(text)}\n")
        result: List[List[float]] = []
        for t in text:
            text_embedding_input = [TextEmbeddingInput(
                task_type=self.conf.task_type, text=t
            )]
            embeddings = self.model.get_embeddings(text_embedding_input, auto_truncate=self.conf.auto_truncate,
                                                   output_dimensionality=self.conf.output_dimensionality)
            for embedding in embeddings:
                result.append(embedding.values)
        return result


class VertexAIEmbeddingsResolver(DependencyResolver[Embeddings]):
    memoize_resolution: bool = False

    def __init__(
            self,
            model_name: str = DEFAULT_MODEL,
            conf: JsonDict = {},
    ):
        super().__init__()
        embedding_conf = VertexAIEmbeddingConf.model_validate(conf)

        self.embedding = VertexAIEmbeddings(
            model=model_name,
            conf=embedding_conf,
        )

    def resolve(self, guild_id: str, agent_id: str) -> Embeddings:
        return self.embedding
