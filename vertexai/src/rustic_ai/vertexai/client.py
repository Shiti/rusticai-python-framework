from typing import Optional
import os

from google.cloud import aiplatform
from google.cloud.aiplatform.initializer import global_config
from google.auth.credentials import Credentials as GoogleCredentialsObject
from pydantic import BaseModel, ConfigDict


class VertexAIConf(BaseModel):
    """Configuration class for Vertex AI.

    This class is used to encapsulate configuration details required for
    interacting with Vertex AI.

    Attributes:
        project_id (Optional[str]): The Google Cloud project ID associated with
            Vertex AI.
        location (Optional[str]): The region or location where the Vertex AI
            resources are hosted.
        credentials (Optional[GoogleCredentialsObject]): Authentication
            credentials used for interacting with Vertex AI.
    """
    project_id: Optional[str] = None
    location: Optional[str] = None
    credentials: Optional[GoogleCredentialsObject] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VertexAIBase:
    """Base class for Google Cloud Vertex AI services.
    
    This class handles the common initialization pattern for the Vertex AI SDK,
    ensuring that it's only initialized once per runtime. Other Vertex AI
    service implementations can inherit from this class to reuse this logic.
    """
    _is_vertexai_initialized = False

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[GoogleCredentialsObject] = None,
    ):
        """Initialize Vertex AI SDK if not already initialized.
        
        Args:
            project_id (Optional[str]): The Google Cloud project ID. If None,
                attempts to use the VERTEXAI_PROJECT environment variable.
            location (Optional[str]): The region or location where the Vertex AI
                resources are hosted. If None, attempts to use the VERTEXAI_LOCATION
                environment variable.
            credentials (Optional[auth_credentials.Credentials]): Authentication
                credentials used for interacting with Vertex AI.

        Raises:
            ValueError: If project_id or location cannot be determined from
                parameters or environment variables.
        """
        if not self._is_vertexai_initialized:
            if project_id is None:
                project_id = os.environ.get("VERTEXAI_PROJECT")

            if location is None:
                location = os.environ.get("VERTEXAI_LOCATION")

            aiplatform.init(
                project=project_id,
                location=location,
                credentials=credentials,
            )
            self._is_vertexai_initialized = True
