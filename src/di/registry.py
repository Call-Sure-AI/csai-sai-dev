# src/di/registry.py
"""
Service registry for managing and discovering services.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, List
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceInfo:
    """Information about a registered service."""
    
    def __init__(
        self,
        name: str,
        service_type: Type,
        instance: Any,
        dependencies: List[str] = None,
        health_check: Optional[callable] = None
    ):
        self.name = name
        self.service_type = service_type
        self.instance = instance
        self.dependencies = dependencies or []
        self.health_check = health_check
        self.registered_at = datetime.utcnow()
        self.last_health_check = None
        self.is_healthy = True

class ServiceRegistry:
    """Registry for managing service instances and dependencies."""
    
    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._service_instances: Dict[Type, Any] = {}
        self._health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
    
    def register_service(
        self,
        name: str,
        service_type: Type[T],
        instance: T,
        dependencies: List[str] = None,
        health_check: Optional[callable] = None
    ) -> None:
        """Register a service instance."""
        service_info = ServiceInfo(
            name=name,
            service_type=service_type,
            instance=instance,
            dependencies=dependencies,
            health_check=health_check
        )
        
        self._services[name] = service_info
        self._service_instances[service_type] = instance
        
        logger.info(f"Registered service: {name} ({service_type.__name__})")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get service instance by name."""
        service_info = self._services.get(name)
        return service_info.instance if service_info else None
    
    def get_service_by_type(self, service_type: Type[T]) -> Optional[T]:
        """Get service instance by type."""
        return self._service_instances.get(service_type)
    
    def unregister_service(self, name: str) -> bool:
        """Unregister a service."""
        if name in self._services:
            service_info = self._services.pop(name)
            self._service_instances.pop(service_info.service_type, None)
            logger.info(f"Unregistered service: {name}")
            return True
        return False
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services."""
        services = []
        for name, info in self._services.items():
            services.append({
                "name": name,
                "type": info.service_type.__name__,
                "dependencies": info.dependencies,
                "registered_at": info.registered_at.isoformat(),
                "is_healthy": info.is_healthy,
                "last_health_check": info.last_health_check.isoformat() if info.last_health_check else None
            })
        return services
    
    def get_service_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a service."""
        service_info = self._services.get(name)
        return service_info.dependencies if service_info else []
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all service dependencies are satisfied."""
        missing_dependencies = []
        
        for name, info in self._services.items():
            for dependency in info.dependencies:
                if dependency not in self._services:
                    missing_dependencies.append(f"{name} requires {dependency}")
        
        return missing_dependencies
    
    async def start_health_monitoring(self) -> None:
        """Start background health monitoring for services."""
        if self._health_check_task:
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started service health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped service health monitoring")
    
    async def check_service_health(self, name: str) -> bool:
        """Check health of a specific service."""
        service_info = self._services.get(name)
        if not service_info or not service_info.health_check:
            return True
        
        try:
            if asyncio.iscoroutinefunction(service_info.health_check):
                is_healthy = await service_info.health_check(service_info.instance)
            else:
                is_healthy = service_info.health_check(service_info.instance)
            
            service_info.is_healthy = is_healthy
            service_info.last_health_check = datetime.utcnow()
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for service {name}: {e}")
            service_info.is_healthy = False
            service_info.last_health_check = datetime.utcnow()
            return False
    
    async def check_all_services_health(self) -> Dict[str, bool]:
        """Check health of all services."""
        health_status = {}
        
        for name in self._services:
            health_status[name] = await self.check_service_health(name)
        
        return health_status
    
    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                health_status = await self.check_all_services_health()
                
                unhealthy_services = [name for name, healthy in health_status.items() if not healthy]
                if unhealthy_services:
                    logger.warning(f"Unhealthy services detected: {unhealthy_services}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        total_services = len(self._services)
        healthy_services = sum(1 for info in self._services.values() if info.is_healthy)
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
            "services": self.list_services(),
            "missing_dependencies": self.validate_dependencies()
        }

# Global service registry instance
service_registry = ServiceRegistry()

# Health check functions for different service types
async def check_database_health(service) -> bool:
    """Health check for database services."""
    try:
        # Attempt a simple query
        if hasattr(service, 'session'):
            result = service.session.execute("SELECT 1")
            return True
        return True
    except Exception:
        return False

async def check_external_api_health(service) -> bool:
    """Health check for external API services."""
    try:
        if hasattr(service, 'validate_api_key'):
            return await service.validate_api_key()
        return True
    except Exception:
        return False

async def check_storage_health(service) -> bool:
    """Health check for storage services."""
    try:
        if hasattr(service, 'file_exists'):
            # Try to check if a test file exists (doesn't matter if it doesn't)
            await service.file_exists("health_check_test")
            return True
        return True
    except Exception:
        return False

async def check_cache_health(service) -> bool:
    """Health check for cache services."""
    try:
        if hasattr(service, 'set') and hasattr(service, 'get'):
            test_key = "health_check_test"
            await service.set(test_key, "test", ttl=10)
            value = await service.get(test_key)
            await service.delete(test_key)
            return value == "test"
        return True
    except Exception:
        return False