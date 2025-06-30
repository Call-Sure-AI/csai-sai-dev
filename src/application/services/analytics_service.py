# src/application/services/analytics_service.py
"""
Analytics service for collecting and processing system metrics.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta

from core.interfaces.services import IAnalyticsService
from core.interfaces.repositories import IAnalyticsRepository
from core.entities.analytics import SessionAnalytics, ConversationAnalytics, SystemMetrics, AnalyticsTimeframe
from core.entities.client import ClientSession

logger = logging.getLogger(__name__)

class AnalyticsService(IAnalyticsService):
    """Service for analytics collection and reporting."""
    
    def __init__(self, analytics_repository: IAnalyticsRepository):
        self.analytics_repository = analytics_repository
    
    async def record_session_start(self, session: ClientSession) -> None:
        """Record the start of a client session."""
        try:
            analytics = SessionAnalytics.create_from_session(
                session_id=session.session_id,
                client_id=session.client_id,
                company_id=session.company_id,
                start_time=session.connection_time
            )
            
            await self.analytics_repository.save_session_analytics(analytics)
            logger.debug(f"Recorded session start: {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error recording session start: {e}")
    
    async def record_session_end(self, session: ClientSession) -> None:
        """Record the end of a client session."""
        try:
            # Get existing analytics
            analytics = await self.analytics_repository.get_session_analytics(session.session_id)
            
            if not analytics:
                # Create new analytics if not found
                analytics = SessionAnalytics.create_from_session(
                    session_id=session.session_id,
                    client_id=session.client_id,
                    company_id=session.company_id,
                    start_time=session.connection_time
                )
            
            # Update with completion data
            end_time = datetime.utcnow()
            analytics = analytics.with_completion(end_time)
            
            # Update with session metrics
            analytics = analytics.with_message_stats(
                total_messages=session.metrics.message_count,
                user_messages=session.metrics.message_count // 2,  # Approximate
                agent_messages=session.metrics.message_count // 2,
                error_messages=session.metrics.error_count
            )
            
            # Add voice data if applicable
            if session.metrics.voice_minutes > 0:
                analytics = analytics.with_voice_data(
                    duration_minutes=session.metrics.voice_minutes,
                    quality_score=0.8  # Placeholder
                )
            
            # Add performance data
            if session.metrics.request_times:
                avg_time = session.metrics.average_request_time
                max_time = max(session.metrics.request_times)
                min_time = min(session.metrics.request_times)
                
                analytics = analytics.with_performance_data(
                    avg_response_time=avg_time,
                    max_response_time=max_time,
                    min_response_time=min_time
                )
            
            await self.analytics_repository.save_session_analytics(analytics)
            logger.debug(f"Recorded session end: {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error recording session end: {e}")
    
    async def record_conversation_analytics(
        self,
        conversation_id: str,
        client_id: str,
        agent_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record conversation analytics."""
        try:
            analytics = ConversationAnalytics.create_from_conversation(
                conversation_id=conversation_id,
                client_id=client_id,
                agent_id=agent_id
            )
            
            if metrics:
                # Add metrics data
                if "response_time" in metrics:
                    analytics = analytics.with_response_time(metrics["response_time"])
            
            await self.analytics_repository.save_conversation_analytics(analytics)
            logger.debug(f"Recorded conversation analytics: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error recording conversation analytics: {e}")
    
    async def record_system_metrics(
        self,
        active_connections: int,
        active_voice_calls: int,
        processing_load: float,
        error_rate: float,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record system-wide metrics."""
        try:
            metrics = SystemMetrics.create_snapshot(AnalyticsTimeframe.REAL_TIME)
            
            # Update with provided data
            metrics = metrics.with_connections(
                active=active_connections,
                total_today=0,  # Would need to calculate
                peak_concurrent=active_connections,
                success_rate=1.0 - error_rate
            )
            
            # Add additional metrics if provided
            if additional_metrics:
                for key, value in additional_metrics.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
            
            await self.analytics_repository.save_system_metrics(metrics)
            logger.debug("Recorded system metrics")
            
        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")
    
    async def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for a specific session."""
        try:
            analytics = await self.analytics_repository.get_session_analytics(session_id)
            return analytics.to_dict() if analytics else None
        except Exception as e:
            logger.error(f"Error getting session analytics: {e}")
            return None
    
    async def get_conversation_analytics(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for a specific conversation."""
        try:
            analytics = await self.analytics_repository.get_conversation_analytics(conversation_id)
            return analytics.to_dict() if analytics else None
        except Exception as e:
            logger.error(f"Error getting conversation analytics: {e}")
            return None
    
    async def get_company_analytics(
        self,
        company_id: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Get aggregated analytics for a company."""
        try:
            start_time = datetime.combine(start_date, datetime.min.time())
            end_time = datetime.combine(end_date, datetime.max.time())
            
            analytics = await self.analytics_repository.get_company_analytics(
                company_id=company_id,
                start_time=start_time,
                end_time=end_time
            )
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting company analytics: {e}")
            return {}
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            # Get recent system metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            metrics_list = await self.analytics_repository.get_system_metrics(
                start_time=start_time,
                end_time=end_time,
                timeframe="hourly"
            )
            
            if not metrics_list:
                return {"status": "no_data", "timestamp": end_time.isoformat()}
            
            latest_metrics = metrics_list[-1]
            
            return {
                "status": "healthy" if not latest_metrics.is_overloaded else "warning",
                "timestamp": latest_metrics.timestamp.isoformat(),
                "active_connections": latest_metrics.active_connections,
                "active_voice_calls": latest_metrics.active_voice_calls,
                "error_rate": latest_metrics.error_rate,
                "average_response_time": latest_metrics.average_response_time,
                "system_health_score": latest_metrics.system_health_score,
                "resource_usage": {
                    "cpu": latest_metrics.cpu_usage_percent,
                    "memory": latest_metrics.memory_usage_percent,
                    "disk": latest_metrics.disk_usage_percent
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system health report: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_performance_trends(
        self,
        metric_type: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get performance trends for a specific metric."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            trends = await self.analytics_repository.get_performance_trends(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                granularity="daily"
            )
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return []
    
    async def cleanup_old_analytics(self, retention_days: int = 90) -> int:
        """Clean up old analytics data."""
        try:
            deleted_count = await self.analytics_repository.cleanup_old_analytics(retention_days)
            logger.info(f"Cleaned up {deleted_count} old analytics records")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up analytics: {e}")
            return 0