# src/infrastructure/storage/__init__.py
"""
Storage infrastructure for cloud storage and caching services.
"""

from .s3_storage import S3StorageService
from .redis_cache import RedisCacheService

__all__ = [
    "S3StorageService",
    "RedisCacheService"
]

# src/infrastructure/storage/s3_storage.py
"""
AWS S3 storage service implementation.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
import aioboto3
from botocore.exceptions import ClientError

from core.interfaces.external import IStorageService

logger = logging.getLogger(__name__)

class S3StorageService(IStorageService):
    """AWS S3 implementation of storage service."""
    
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1"
    ):
        self.bucket_name = bucket_name
        self.session = aioboto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    
    async def upload_file(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload file and return URL."""
        try:
            async with self.session.client('s3') as s3:
                put_kwargs = {
                    'Bucket': self.bucket_name,
                    'Key': key,
                    'Body': data
                }
                
                if content_type:
                    put_kwargs['ContentType'] = content_type
                
                if metadata:
                    put_kwargs['Metadata'] = metadata
                
                await s3.put_object(**put_kwargs)
                
                # Return the S3 URL
                return f"https://{self.bucket_name}.s3.amazonaws.com/{key}"
                
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise
    
    async def download_file(self, key: str) -> bytes:
        """Download file by key."""
        try:
            async with self.session.client('s3') as s3:
                response = await s3.get_object(
                    Bucket=self.bucket_name,
                    Key=key
                )
                return await response['Body'].read()
                
        except ClientError as e:
            logger.error(f"Error downloading file from S3: {e}")
            raise
    
    async def delete_file(self, key: str) -> bool:
        """Delete file by key."""
        try:
            async with self.session.client('s3') as s3:
                await s3.delete_object(
                    Bucket=self.bucket_name,
                    Key=key
                )
                return True
                
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {e}")
            return False
    
    async def file_exists(self, key: str) -> bool:
        """Check if file exists."""
        try:
            async with self.session.client('s3') as s3:
                await s3.head_object(
                    Bucket=self.bucket_name,
                    Key=key
                )
                return True
                
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking file existence in S3: {e}")
            raise
    
    async def get_file_info(self, key: str) -> Dict[str, Any]:
        """Get file metadata."""
        try:
            async with self.session.client('s3') as s3:
                response = await s3.head_object(
                    Bucket=self.bucket_name,
                    Key=key
                )
                
                return {
                    "size": response.get('ContentLength'),
                    "last_modified": response.get('LastModified'),
                    "content_type": response.get('ContentType'),
                    "metadata": response.get('Metadata', {}),
                    "etag": response.get('ETag')
                }
                
        except ClientError as e:
            logger.error(f"Error getting file info from S3: {e}")
            raise
    
    async def list_files(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List files with optional prefix filter."""
        try:
            async with self.session.client('s3') as s3:
                list_kwargs = {'Bucket': self.bucket_name}
                
                if prefix:
                    list_kwargs['Prefix'] = prefix
                
                if limit:
                    list_kwargs['MaxKeys'] = limit
                
                response = await s3.list_objects_v2(**list_kwargs)
                
                files = []
                for obj in response.get('Contents', []):
                    files.append({
                        "key": obj['Key'],
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'],
                        "etag": obj['ETag']
                    })
                
                return files
                
        except ClientError as e:
            logger.error(f"Error listing files from S3: {e}")
            return []
    
    async def generate_presigned_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        method: str = "GET"
    ) -> str:
        """Generate presigned URL for direct access."""
        try:
            async with self.session.client('s3') as s3:
                if method.upper() == "GET":
                    url = await s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.bucket_name, 'Key': key},
                        ExpiresIn=expiration_seconds
                    )
                elif method.upper() == "PUT":
                    url = await s3.generate_presigned_url(
                        'put_object',
                        Params={'Bucket': self.bucket_name, 'Key': key},
                        ExpiresIn=expiration_seconds
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                return url
                
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise
    
    async def copy_file(self, source_key: str, dest_key: str) -> bool:
        """Copy file from source to destination."""
        try:
            async with self.session.client('s3') as s3:
                copy_source = {
                    'Bucket': self.bucket_name,
                    'Key': source_key
                }
                
                await s3.copy_object(
                    CopySource=copy_source,
                    Bucket=self.bucket_name,
                    Key=dest_key
                )
                
                return True
                
        except ClientError as e:
            logger.error(f"Error copying file in S3: {e}")
            return False