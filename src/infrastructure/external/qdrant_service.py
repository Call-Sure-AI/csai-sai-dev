# src/infrastructure/external/qdrant_service.py
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.interfaces.external import IVectorStoreService

logger = logging.getLogger(__name__)

class QdrantVectorService(IVectorStoreService):
    """Qdrant vector store service implementation"""
    
    def __init__(self, host: str, port: int, api_key: str):
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=True
        )
        self.vector_size = 1536  # text-embedding-3-small dimension
    
    async def search(
        self, 
        company_id: str, 
        query_embedding: List[float],
        agent_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search vector store"""
        try:
            collection_name = f"company_{company_id}"
            
            # Build filter
            search_filter = None
            if agent_id:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="agent_id",
                            match=models.MatchValue(value=agent_id)
                        )
                    ]
                )
            
            # Perform search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "agent_id": point.payload.get("agent_id")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []
    
    async def add_documents(
        self, 
        company_id: str, 
        agent_id: str,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Add documents to vector store"""
        try:
            collection_name = f"company_{company_id}"
            
            # Ensure collection exists
            await self._ensure_collection(collection_name)
            
            # Prepare points
            points = []
            for i, doc in enumerate(documents):
                point = models.PointStruct(
                    id=doc.get("id", f"{agent_id}_{i}"),
                    vector=doc.get("embedding", [0.0] * self.vector_size),
                    payload={
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "agent_id": agent_id,
                        "company_id": company_id
                    }
                )
                points.append(point)
            
            # Upsert points
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {e}")
            return False
    
    async def delete_agent_data(self, company_id: str, agent_id: str) -> bool:
        """Delete agent data from vector store"""
        try:
            collection_name = f"company_{company_id}"
            
            # Delete points with agent_id filter
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="agent_id",
                                match=models.MatchValue(value=agent_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted agent data for {agent_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting agent data from Qdrant: {e}")
            return False
    
    async def _ensure_collection(self, collection_name: str) -> None:
        """Ensure collection exists"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {e}")
            raise