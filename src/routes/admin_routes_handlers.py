# api/routes/admin_routes.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from pydantic import BaseModel
import json
from datetime import datetime
import uuid
import os

from database.config import get_db
from database.models import Company, Agent, Document, DatabaseIntegration, DocumentType
from services.vector_store.qdrant_service import QdrantService
from services.rag.rag_service import RAGService
from services.vector_store.document_embedding import DocumentEmbeddingService
from services.vector_store.image_embedding import ImageEmbeddingService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/companies")
async def create_company(
    company_data: dict,
    db: Session = Depends(get_db)
):
    try:
        # Generate a unique API key if not provided
        if 'api_key' not in company_data or not company_data['api_key']:
            company_data['api_key'] = str(uuid.uuid4())
        logger.info(f"Generated API key: {company_data['api_key']}")
        
        company = Company(**company_data)
        db.add(company)
        db.commit()
        db.refresh(company)
        return company
    except Exception as e:
        logger.error(f"Error creating company: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating company: {str(e)}")

@router.post("/companies/login")
async def login_company(
    login_data: dict,
    db: Session = Depends(get_db)
):
    try:
        # Validate email and API key
        company = db.query(Company).filter(
            Company.email == login_data.get("email"),
            Company.api_key == login_data.get("api_key")
        ).first()
        
        if not company:
            raise HTTPException(status_code=401, detail="Invalid email or API key")
        
        # Return company details upon successful login
        return {
            "id": company.id,
            "name": company.name,
            "email": company.email,
            "api_key": company.api_key,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")

def is_image_file(content_type: str) -> bool:
    """Check if the file is an image based on content type"""
    return content_type.startswith('image/')

@router.post("/agents")
async def create_agent_with_documents(
    name: str = Form(...),
    type: str = Form(...),
    company_id: str = Form(...),
    prompt: str = Form(...),
    is_active: bool = Form(True),
    additional_context: Optional[str] = Form(None),  # JSON string
    advanced_settings: Optional[str] = Form(None),  # JSON string
    file_urls: Optional[str] = Form(None),  # JSON string of file URLs
    user_id: str = Form(...),  # User ID for the agent
    id: Optional[str] = Form(None),  # Accept existing ID to prevent duplication
    db: Session = Depends(get_db)
):
    """
    Create a new agent with support for multiple file URLs (documents and images)
    """
    try:
        logger.info(f"Creating agent with name: {name}, type: {type}, company_id: {company_id}")
        
        # Check if company_id is "undefined" or not a valid UUID
        if company_id == "undefined" or not is_valid_uuid(company_id):
            raise HTTPException(status_code=400, detail="Invalid company_id. A valid company ID is required.")
        
        # Verify company exists in the database
        company = db.query(Company).filter_by(id=company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
        
        # Parse additional_context if provided
        additional_context_dict = {}
        if additional_context:
            try:
                additional_context_dict = json.loads(additional_context)
            except json.JSONDecodeError:
                logger.warning("Invalid additional_context JSON provided")
        
        # Parse advanced_settings if provided
        advanced_settings_dict = {}
        if advanced_settings:
            try:
                advanced_settings_dict = json.loads(advanced_settings)
            except json.JSONDecodeError:
                logger.warning("Invalid advanced_settings JSON provided")
        
        # Use provided ID or generate a new one
        agent_id = id if id else str(uuid.uuid4())
        logger.info(f"Using agent ID: {agent_id} ({'provided' if id else 'generated'})")
        
        # Create agent with fields that match the Prisma schema
        agent = Agent(
            id=agent_id,  # Use the provided ID or a new one
            user_id=user_id,
            name=name,
            type=AgentType(type.lower()),
            company_id=company_id,
            prompt=prompt,
            additional_context=additional_context_dict,
            advanced_settings=advanced_settings_dict,
            is_active=is_active,
            # Fields with default values
            knowledge_base_ids=[],
            database_integration_ids=[],
            search_config={
                'score_threshold': 0.7,
                'limit': 5,
                'include_metadata': True
            },
            confidence_threshold=0.7,
            max_response_tokens=200,
            temperature=0.7,
            total_interactions=0,
            average_confidence=0.0,
            success_rate=0.0,
            average_response_time=0.0,
            image_processing_enabled=False,
            image_processing_config={
                'max_images': 1000,
                'confidence_threshold': 0.7,
                'enable_auto_description': True
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        db.add(agent)
        db.commit()
        db.refresh(agent)

        document_ids = []
        image_ids = []
        files = []  # List to store file URLs for agent.files field

        if file_urls:
            # Parse file URLs
            try:
                urls_list = json.loads(file_urls)
                logger.info(f"File URLs received: {urls_list}")
                # Store file URLs in files field
                files = urls_list
            except json.JSONDecodeError:
                logger.error("Invalid file URLs JSON provided")
                raise HTTPException(status_code=400, detail="Invalid file URLs format")

            # Initialize services
            qdrant_service = QdrantService()
            document_embedding_service = DocumentEmbeddingService(qdrant_service)
            image_embedding_service = ImageEmbeddingService(qdrant_service)
            
            # Ensure collection exists
            await qdrant_service.setup_collection(company_id)
            
            # Process each file URL
            image_files = []
            document_files = []
            
            for url in urls_list:
                try:
                    # List objects in bucket to verify exact key
                    filename = url.split('/')[-1]
                    logger.info(f"Original filename from URL: {filename}")
                    
                    # Try downloading directly using boto3
                    content, content_type = download_from_s3_direct(url)
                    
                    if not content:
                        logger.warning(f"Failed to download content for URL: {url}")
                        continue
                    
                    logger.info(f"Downloaded file: {filename}, size: {len(content)} bytes, content type: {content_type}")
                    
                    # Determine if file is an image
                    # Determine if file is an image
                    is_image = is_image_file(content_type)
                    
                    # Create document record with proper content handling
                    if is_image:
                        document = Document(
                            company_id=company_id,
                            agent_id=agent.id,
                            name=filename,
                            content=f"[Image: {filename}]",  # Placeholder text for images
                            image_content=content,  # Binary data here
                            is_image=True,
                            file_size=len(content),
                            original_filename=filename,
                            file_type=content_type,
                            type=DocumentType.image,
                            metadata={
                                "original_filename": filename,
                                "file_url": url,
                                "uploaded_at": datetime.now().isoformat()
                            }
                        )
                    else:
                        # For documents: decode content to text
                        try:
                            if isinstance(content, bytes):
                                text_content = content.decode('utf-8')
                            else:
                                text_content = str(content)
                        except UnicodeDecodeError:
                            text_content = content.decode('utf-8', errors='ignore')
                        
                        document = Document(
                            company_id=company_id,
                            agent_id=agent.id,
                            name=filename,
                            content=text_content,  # Decoded text
                            is_image=False,
                            file_size=len(content) if isinstance(content, bytes) else len(content.encode('utf-8')),
                            original_filename=filename,
                            file_type=content_type,
                            type=DocumentType.custom,
                            metadata={
                                "original_filename": filename,
                                "file_url": url,
                                "uploaded_at": datetime.now().isoformat()
                            }
                        )
                    
                    db.add(document)
                    db.commit()
                    db.refresh(document)
                    
                    # Add to appropriate list based on file type
                    if is_image:
                        image_files.append({
                            'id': document.id,
                            'content': content,
                            'filename': filename,
                            'content_type': content_type,
                            'metadata': {
                                'agent_id': agent.id,
                                'original_filename': filename,
                                'file_url': url
                            }
                        })
                        image_ids.append(document.id)
                    else:
                        document_files.append({
                            'id': document.id,
                            'content': content,
                            'metadata': {
                                'agent_id': agent.id,
                                'filename': filename,
                                'file_type': content_type,
                                'file_url': url
                            }
                        })
                        document_ids.append(document.id)
                            
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    continue
            
            # Update agent with files field
            agent.files = files
            db.commit()
            
            # Embed documents if any
            if document_files:
                success = await document_embedding_service.embed_documents(
                    company_id=company_id,
                    agent_id=agent.id,
                    documents=document_files
                )
                if not success:
                    logger.error("Failed to embed documents")
            
            # Embed images if any
            if image_files:
                success = await image_embedding_service.embed_images(
                    company_id=company_id,
                    agent_id=agent.id,
                    images=image_files
                )
                if not success:
                    logger.error("Failed to embed images")

        return {
            "status": "success",
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "company_id": agent.company_id,
                "prompt": agent.prompt,
                "is_active": agent.is_active,
                "additional_context": agent.additional_context,
                "advanced_settings": agent.advanced_settings,
                "files": agent.files
            },
            "documents": {
                "total": len(document_ids) + len(image_ids),
                "document_ids": document_ids,
                "image_ids": image_ids
            }
        }

    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to validate UUID
def is_valid_uuid(val):
    """Check if string is a valid UUID"""
    try:
        uuid_obj = uuid.UUID(str(val))
        return str(uuid_obj) == val
    except (ValueError, AttributeError, TypeError):
        return False


def list_s3_bucket_objects(bucket_name, prefix=''):
    """
    List objects in an S3 bucket with an optional prefix
    """
    try:
        import boto3
        logger.info(f"AWS Access Key ID present: {'Yes' if os.environ.get('aws_access_key_id') else 'No'}")
        logger.info(f"AWS Secret Access Key present: {'Yes' if os.environ.get('aws_secret_access_key') else 'No'}")
        s3_client = boto3.client(
            's3',
            region_name='ap-south-1',
            
            aws_access_key_id = os.environ.get("aws_access_key_id"),
            aws_secret_access_key = os.environ.get("aws_secret_access_key")
        )
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=100
        )
        
        if 'Contents' in response:
            objects = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Found {len(objects)} objects in bucket {bucket_name} with prefix '{prefix}'")
            logger.info(f"Objects: {objects[:10]}")
            return objects
        else:
            logger.warning(f"No objects found in bucket {bucket_name} with prefix '{prefix}'")
            return []
    except Exception as e:
        logger.error(f"Error listing S3 bucket objects: {str(e)}")
        return []

def download_from_s3_direct(url):
    """
    Download an S3 object directly using boto3, with special handling for URL formats
    """
    # Input validation
    if not url or not isinstance(url, str):
        logger.error("Invalid URL provided")
        return None, None
    
    try:
        import boto3
        from urllib.parse import urlparse, unquote
        from botocore.exceptions import ClientError, NoCredentialsError
        
        logger.info(f"Attempting to download directly from URL: {url}")
        
        # Parse URL to get bucket and key
        parsed_url = urlparse(url)
        bucket_name, key = None, None
        
        # Check for different S3 URL formats
        if not parsed_url.netloc.endswith('.amazonaws.com'):
            logger.error(f"URL is not an S3 URL: {url}")
            return None, None
            
        # Format: https://bucket-name.s3.region.amazonaws.com/key
        if '.s3.' in parsed_url.netloc:
            parts = parsed_url.netloc.split('.s3.')
            bucket_name = parts[0]
            key = unquote(parsed_url.path.lstrip('/'))
        # Format: https://s3.region.amazonaws.com/bucket-name/key
        elif parsed_url.netloc.startswith('s3.'):
            path_parts = parsed_url.path.lstrip('/').split('/', 1)
            if len(path_parts) >= 2:
                bucket_name = path_parts[0]
                key = unquote(path_parts[1])
            else:
                logger.error(f"Invalid S3 URL format (path): {url}")
                return None, None
        else:
            logger.error(f"Unrecognized S3 URL format: {url}")
            return None, None
        
        # Validate parsed values
        if not bucket_name or not key:
            logger.error(f"Failed to parse bucket/key from URL: {url}")
            return None, None
            
        logger.info(f"Parsed S3 URL: bucket={bucket_name}, key={key}")
        
        # Check AWS credentials
        aws_access_key = os.environ.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            logger.error("AWS credentials not found in environment variables")
            return None, None
        
        # Set up S3 client with proper error handling
        try:
            s3_client = boto3.client(
                's3',
                region_name='ap-south-1',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        except NoCredentialsError:
            logger.error("Invalid AWS credentials")
            return None, None
        except Exception as e:
            logger.error(f"Failed to create S3 client: {str(e)}")
            return None, None
        
        # Try to get the object directly first
        content, content_type = _try_download_object(s3_client, bucket_name, key)
        if content is not None:
            return content, content_type
        
        # If direct download failed, try with fallback strategies
        logger.info(f"Direct download failed, trying fallback strategies for: {key}")
        return _try_download_with_fallback(s3_client, bucket_name, key)
            
    except Exception as e:
        logger.error(f"Error in download_from_s3_direct: {str(e)}")
        return None, None


def _try_download_object(s3_client, bucket_name, key):
    """Try to download object directly"""
    try:
        logger.info(f"Getting object from S3: bucket={bucket_name}, key={key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read()
        content_type = response.get('ContentType', 'application/octet-stream')
        
        # Validate content
        if not content:
            logger.warning(f"Downloaded empty content for key: {key}")
            return None, None
            
        logger.info(f"Successfully downloaded S3 object: {key} ({len(content)} bytes)")
        return content, content_type
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.warning(f"S3 object not found: {key}")
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied for S3 object: {key}")
        else:
            logger.error(f"S3 client error for {key}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error downloading S3 object {key}: {str(e)}")
        return None, None


def _try_download_with_fallback(s3_client, bucket_name, original_key):
    """Try to download using fallback strategies"""
    try:
        # Strategy 1: Check if objects exist with similar names
        similar_objects = list_s3_bucket_objects(bucket_name, original_key.split('/')[-1].split('-')[0])
        
        # Strategy 2: If no similar objects and key has dashes, try base name
        if not similar_objects and '-' in original_key:
            base_name = original_key.split('/')[-1].split('-')[0]
            similar_objects = list_s3_bucket_objects(bucket_name, base_name)
        
        # Try downloading similar objects
        for similar_key in similar_objects[:5]:  # Limit to first 5 matches
            logger.info(f"Trying alternative key: {similar_key}")
            content, content_type = _try_download_object(s3_client, bucket_name, similar_key)
            if content is not None:
                return content, content_type
        
        logger.warning(f"No downloadable alternatives found for: {original_key}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error in fallback download: {str(e)}")
        return None, None
    
def get_content_type_from_extension(extension: str) -> str:
    """
    Get MIME type from file extension
    """
    extension_to_mime = {
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
        'csv': 'text/csv',
        'xls': 'application/vnd.ms-excel',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'ppt': 'application/vnd.ms-powerpoint',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'svg': 'image/svg+xml',
        'html': 'text/html',
        'json': 'application/json'
    }
    
    return extension_to_mime.get(extension, 'application/octet-stream')


@router.post("/documents/upload")
async def upload_documents(
    company_id: str,
    agent_id: str,
    files: List[UploadFile] = File(...),
    descriptions: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload and process documents and images for an existing agent
    """
    try:
        # Check if agent exists
        agent = db.query(Agent).filter_by(id=agent_id, company_id=company_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Parse descriptions if provided
        descriptions_dict = {}
        if descriptions:
            try:
                descriptions_dict = json.loads(descriptions)
            except json.JSONDecodeError:
                logger.warning("Invalid descriptions JSON provided")
        
        # Initialize services
        qdrant_service = QdrantService()
        document_embedding_service = DocumentEmbeddingService(qdrant_service)
        image_embedding_service = ImageEmbeddingService(qdrant_service)
        
        # Ensure collection exists
        await qdrant_service.setup_collection(company_id)
        
        # Process files
        document_ids = []
        image_ids = []
        document_files = []
        image_files = []
        
        for file in files:
            try:
                content = await file.read()
                
                # Determine if file is an image
                is_image = is_image_file(file.content_type)
                
                # Create document record
                document = Document(
                    company_id=company_id,
                    agent_id=agent_id,
                    name=file.filename,
                    content=content,
                    file_type=file.content_type,
                    type=DocumentType.image if is_image else DocumentType.custom,
                    metadata={
                        "description": descriptions_dict.get(file.filename) if is_image else None,
                        "original_filename": file.filename,
                        "uploaded_at": datetime.now().isoformat()
                    }
                )
                
                db.add(document)
                db.commit()
                db.refresh(document)
                
                # Add to appropriate list based on file type
                if is_image:
                    image_files.append({
                        'id': document.id,
                        'content': content,
                        'description': descriptions_dict.get(file.filename),
                        'filename': file.filename,
                        'content_type': file.content_type,
                        'metadata': {
                            'agent_id': agent_id,
                            'original_filename': file.filename
                        }
                    })
                    image_ids.append(document.id)
                else:
                    document_files.append({
                        'id': document.id,
                        'content': content,
                        'metadata': {
                            'agent_id': agent_id,
                            'filename': file.filename,
                            'file_type': file.content_type
                        }
                    })
                    document_ids.append(document.id)
                        
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        # Process documents if any
        if document_files:
            success = await document_embedding_service.embed_documents(
                company_id=company_id,
                agent_id=agent_id,
                documents=document_files
            )
            if not success:
                logger.error("Failed to embed documents")
        
        # Process images if any
        if image_files:
            success = await image_embedding_service.embed_images(
                company_id=company_id,
                agent_id=agent_id,
                images=image_files
            )
            if not success:
                logger.error("Failed to embed images")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(document_ids) + len(image_ids)} files",
            "documents": {
                "total": len(document_ids) + len(image_ids),
                "document_ids": document_ids,
                "image_ids": image_ids
            }
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{company_id}")
async def get_company_agents(company_id: str, db: Session = Depends(get_db)):
    try:
        agents = db.query(Agent).filter_by(company_id=company_id, is_active=True).all()
        return [{
            "id": agent.id,
            "name": agent.name,
            "type": agent.type,
            "prompt": agent.prompt,
            "documents": len(agent.documents)
        } for agent in agents]
    except Exception as e:
        logger.error(f"Error getting agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))