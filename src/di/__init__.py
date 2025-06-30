# src/di/__init__.py
"""
Dependency Injection Container

This module provides dependency injection for the AI voice calling system,
ensuring loose coupling and testability across all layers.
"""

from .container import Container
from .providers import ServiceProviders
from .registry import ServiceRegistry

__all__ = [
    "Container",
    "ServiceProviders", 
    "ServiceRegistry"
]