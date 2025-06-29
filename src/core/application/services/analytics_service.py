# src/application/services/analytics_service.py
import logging
from datetime import datetime, date
from typing import Dict, Any

from core.interfaces.services import IAnalyticsService
from core.interfaces.repositories import IAnalyticsRepository
from core.entities.client import ClientSession
from core.entities.analytics import SessionEvent, EventType, DailyMetrics, LiveStats

logger = logging.getLogger(__name__)

class AnalyticsService(IAnalyticsService):
    """Service for analytics and metrics collection"""
    
    def __init__(self, analytics_repository: IAnalyticsRepository):
        self.analytics_repository = analytics_repository
    
    async def record_connection(self, session: ClientSession) -> None:
        """Record client connection event"""
        if not session.company:
            return
        
        try:
            event = SessionEvent(
                id=f"{session.client_id}_connect_{int(datetime.utcnow().timestamp())}",
                company_id=session.company["id"],
                client_id=session.client_id,
                event_type=EventType.CONNECTION
            )
            
            await self.analytics_repository.record_event(event)
            await self._update_daily_metrics(session.company["id"], total_connections=1)
            
        except Exception as e:
            logger.error(f"Error recording connection: {e}")
    
    async def record_disconnection(self, session: ClientSession) -> None:
        """Record client disconnection with session data"""
        if not session.company:
            return
        
        try:
            event = SessionEvent(
                id=f"{session.client_id}_disconnect_{int(datetime.utcnow().timestamp())}",
                company_id=session.company["id"],
                client_id=session.client_id,
                event_type=EventType.DISCONNECTION,
                session_duration=session.get_session_duration(),
                message_count=session.message_count,
                token_count=session.total_tokens
            )
            
            await self.analytics_repository.record_event(event)
            await self._update_daily_metrics(
                session.company["id"],
                connection_hours=session.get_session_duration() / 3600,
                total_messages=session.message_count,
                total_tokens=session.total_tokens
            )
            
        except Exception as e:
            logger.error(f"Error recording disconnection: {e}")
    
    async def record_message(
        self, 
        client_id: str, 
        tokens: int, 
        response_time: float
    ) -> None:
        """Record message processing metrics"""
        # Implementation would get company_id from client session
        company_id = await self._get_company_id_for_client(client_id)
        if not company_id:
            return
        
        try:
            await self._update_daily_metrics(
                company_id,
                total_messages=1,
                total_tokens=tokens,
                avg_response_time=response_time
            )
        except Exception as e:
            logger.error(f"Error recording message: {e}")
    
    async def record_voice_call(self, client_id: str, duration: float) -> None:
        """Record voice call completion"""
        company_id = await self._get_company_id_for_client(client_id)
        if not company_id:
            return
        
        try:
            event = SessionEvent(
                id=f"{client_id}_voice_{int(datetime.utcnow().timestamp())}",
                company_id=company_id,
                client_id=client_id,
                event_type=EventType.VOICE_END,
                voice_duration=duration
            )
            
            await self.analytics_repository.record_event(event)
            await self._update_daily_metrics(
                company_id,
                total_voice_calls=1,
                total_voice_minutes=duration / 60
            )
        except Exception as e:
            logger.error(f"Error recording voice call: {e}")
    
    async def record_error(self, company_id: str, error_message: str) -> None:
        """Record error occurrence"""
        try:
            event = SessionEvent(
                id=f"error_{company_id}_{int(datetime.utcnow().timestamp())}",
                company_id=company_id,
                client_id="system",
                event_type=EventType.ERROR,
                error_message=error_message
            )
            
            await self.analytics_repository.record_event(event)
            await self._update_daily_metrics(company_id, errors_count=1)
            
        except Exception as e:
            logger.error(f"Error recording error: {e}")
    
    async def get_live_stats(self) -> LiveStats:
        """Get current live system statistics"""
        # This would integrate with connection service to get live data
        stats = LiveStats()
        # Implementation would populate from active connections
        return stats
    
    async def get_company_usage_report(
        self, 
        company_id: str, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, Any]:
        """Get usage report for company"""
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        
        return await self.analytics_repository.get_company_usage_report(
            company_id, start, end
        )
    
    async def _update_daily_metrics(self, company_id: str, **kwargs) -> None:
        """Update daily metrics for company"""
        today = datetime.utcnow().date()
        
        # Get or create metrics
        metrics = await self.analytics_repository.get_daily_metrics(company_id, today)
        if not metrics:
            metrics = DailyMetrics(company_id=company_id, date=today)
        
        # Update metrics
        metrics.update_metrics(**kwargs)
        
        # Save back
        await self.analytics_repository.update_daily_metrics(metrics)
    
    async def _get_company_id_for_client(self, client_id: str) -> Optional[str]:
        """Get company ID for client (placeholder)"""
        # This would be injected from connection service
        return None