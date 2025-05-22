from google.cloud import aiplatform

from rustic_ai.vertexai.client import VertexAIBase


class VertexAISearchIndex(VertexAIBase):
    """
    A class representing a Vertex AI Search Index.
    """

    def __init__(self, project_id: str, location: str, index_id: str):
        """
        Initializes a new instance of the VertexAISearchIndex class.

        Parameters:
            project_id: The ID of the Google Cloud project.
            location: The location of the index.
            index_id: The ID of the index.
        """
        super().__init__(project_id, location)
        self.index_id = index_id

    # Create Index
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        contents_delta_uri=gcs_uri,
        description="Matching Engine Index",
        dimensions=100,
        approximate_neighbors_count=150,
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=7,
        index_update_method="STREAM_UPDATE",  # Options: STREAM_UPDATE, BATCH_UPDATE
        distance_measure_type=aiplatform.matching_engine.matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
    )

    # Create Index Endpoint
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=display_name,
        description="Matching Engine Index Endpoint",
    )

    # Create the index endpoint instance from an existing endpoint.
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=index_endpoint_name
    )

    # Deploy Index to Endpoint
    index_endpoint = index_endpoint.deploy_index(
        index=index, deployed_index_id=deployed_index_id
    )



