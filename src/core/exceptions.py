"""
Domain-specific exceptions for the AI voice calling system.

These exceptions represent business rule violations and domain-specific errors.
They are framework-agnostic and can be raised from any layer of the application.
"""

from typing import Optional, Dict, Any


class DomainException(Exception):
    """Base exception for all domain-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ClientNotFoundException(DomainException):
    """Raised when a client cannot be found."""
    
    def __init__(self, client_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        message = f"Client not found: {client_id}"
        super().__init__(message, "CLIENT_NOT_FOUND", details or {"client_id": client_id})


class ConversationNotFoundException(DomainException):
    """Raised when a conversation cannot be found."""
    
    def __init__(self, conversation_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        message = f"Conversation not found: {conversation_id}"
        super().__init__(message, "CONVERSATION_NOT_FOUND", details or {"conversation_id": conversation_id})


class AgentNotFoundException(DomainException):
    """Raised when an agent cannot be found."""
    
    def __init__(self, agent_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        message = f"Agent not found: {agent_id}"
        super().__init__(message, "AGENT_NOT_FOUND", details or {"agent_id": agent_id})


class InvalidClientStateException(DomainException):
    """Raised when a client is in an invalid state for the requested operation."""
    
    def __init__(
        self, 
        client_id: str, 
        current_state: str, 
        required_state: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Client {client_id} is in state '{current_state}', but '{required_state}' is required"
        exception_details = {
            "client_id": client_id,
            "current_state": current_state,
            "required_state": required_state
        }
        if details:
            exception_details.update(details)
        super().__init__(message, "INVALID_CLIENT_STATE", exception_details)


class MaxConnectionsExceededException(DomainException):
    """Raised when the maximum number of connections is exceeded."""
    
    def __init__(
        self, 
        current_connections: int, 
        max_connections: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Maximum connections exceeded: {current_connections}/{max_connections}"
        exception_details = {
            "current_connections": current_connections,
            "max_connections": max_connections
        }
        if details:
            exception_details.update(details)
        super().__init__(message, "MAX_CONNECTIONS_EXCEEDED", exception_details)


class AuthenticationFailedException(DomainException):
    """Raised when client authentication fails."""
    
    def __init__(
        self, 
        reason: str = "Invalid credentials",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Authentication failed: {reason}"
        super().__init__(message, "AUTHENTICATION_FAILED", details)


class VoiceCallException(DomainException):
    """Raised when voice call operations fail."""
    
    def __init__(
        self, 
        operation: str,
        reason: str,
        client_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Voice call {operation} failed: {reason}"
        exception_details = {"operation": operation, "reason": reason}
        if client_id:
            exception_details["client_id"] = client_id
        if details:
            exception_details.update(details)
        super().__init__(message, "VOICE_CALL_ERROR", exception_details)


class AnalyticsException(DomainException):
    """Raised when analytics operations fail."""
    
    def __init__(
        self, 
        operation: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Analytics {operation} failed: {reason}"
        exception_details = {"operation": operation, "reason": reason}
        if details:
            exception_details.update(details)
        super().__init__(message, "ANALYTICS_ERROR", exception_details)


class RateLimitExceededException(DomainException):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        client_id: str,
        limit_type: str,
        current_rate: int,
        max_rate: int,
        reset_time: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Rate limit exceeded for {client_id}: {current_rate}/{max_rate} {limit_type}"
        exception_details = {
            "client_id": client_id,
            "limit_type": limit_type,
            "current_rate": current_rate,
            "max_rate": max_rate
        }
        if reset_time:
            exception_details["reset_time"] = reset_time
        if details:
            exception_details.update(details)
        super().__init__(message, "RATE_LIMIT_EXCEEDED", exception_details)


class InvalidMessageFormatException(DomainException):
    """Raised when message format is invalid."""
    
    def __init__(
        self,
        message_type: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Invalid {message_type} format: {reason}"
        exception_details = {"message_type": message_type, "reason": reason}
        if details:
            exception_details.update(details)
        super().__init__(message, "INVALID_MESSAGE_FORMAT", exception_details)


class ResourceExhaustedException(DomainException):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self,
        resource_type: str,
        current_usage: Optional[float] = None,
        max_capacity: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Resource exhausted: {resource_type}"
        if current_usage is not None and max_capacity is not None:
            message += f" ({current_usage}/{max_capacity})"
        
        exception_details = {"resource_type": resource_type}
        if current_usage is not None:
            exception_details["current_usage"] = current_usage
        if max_capacity is not None:
            exception_details["max_capacity"] = max_capacity
        if details:
            exception_details.update(details)
        
        super().__init__(message, "RESOURCE_EXHAUSTED", exception_details)