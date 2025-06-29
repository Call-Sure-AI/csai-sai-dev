# src/infrastructure/database/repositories/analytics_repository.py
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
from datetime import date, datetime

from core.interfaces.repositories import IAnalyticsRepository
from core.entities.analytics import SessionEvent, DailyMetrics
from database.models import SessionEvent as SessionEventModel, CompanyUsageMetrics

logger = logging.getLogger(__name__)

class AnalyticsRepository(IAnalyticsRepository):
    """Repository for analytics data persistence"""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def record_event(self, event: SessionEvent) -> None:
        """Record analytics event"""
        try:
            db_event = SessionEventModel(
                id=event.id,
                company_id=event.company_id,
                client_id=event.client_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                session_duration=event.session_duration,
                message_count=event.message_count,
                token_count=event.token_count,
                voice_duration=event.voice_duration,
                error_message=event.error_message
            )
            
            self.session.add(db_event)
            self.session.commit()
            logger.debug(f"Recorded event: {event.event_type.value} for {event.client_id}")
            
        except SQLAlchemyError as e:
            logger.error(f"Error recording event: {e}")
            self.session.rollback()
            raise
    
    async def get_daily_metrics(self, company_id: str, date: date) -> Optional[DailyMetrics]:
        """Get daily metrics for company"""
        try:
            db_metrics = self.session.query(CompanyUsageMetrics).filter_by(
                company_id=company_id,
                date=date
            ).first()
            
            if not db_metrics:
                return None
            
            return DailyMetrics(
                company_id=db_metrics.company_id,
                date=db_metrics.date,
                total_connections=db_metrics.total_connections,
                peak_concurrent_connections=db_metrics.peak_concurrent_connections,
                connection_hours=db_metrics.connection_hours,
                total_messages=db_metrics.total_messages,
                total_tokens=db_metrics.total_tokens,
                total_voice_calls=db_metrics.total_voice_calls,
                total_voice_minutes=db_metrics.total_voice_minutes,
                tickets_created=db_metrics.tickets_created,
                errors_count=db_metrics.errors_count,
                avg_response_time=db_metrics.avg_response_time,
                created_at=db_metrics.created_at,
                updated_at=db_metrics.updated_at
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting daily metrics: {e}")
            raise
    
    async def update_daily_metrics(self, metrics: DailyMetrics) -> None:
        """Update daily metrics"""
        try:
            db_metrics = self.session.query(CompanyUsageMetrics).filter_by(
                company_id=metrics.company_id,
                date=metrics.date
            ).first()
            
            if not db_metrics:
                # Create new metrics record
                db_metrics = CompanyUsageMetrics(
                    id=f"{metrics.company_id}_{metrics.date}",
                    company_id=metrics.company_id,
                    date=metrics.date
                )
                self.session.add(db_metrics)
            
            # Update all fields
            db_metrics.total_connections = metrics.total_connections
            db_metrics.peak_concurrent_connections = metrics.peak_concurrent_connections
            db_metrics.connection_hours = metrics.connection_hours
            db_metrics.total_messages = metrics.total_messages
            db_metrics.total_tokens = metrics.total_tokens
            db_metrics.total_voice_calls = metrics.total_voice_calls
            db_metrics.total_voice_minutes = metrics.total_voice_minutes
            db_metrics.tickets_created = metrics.tickets_created
            db_metrics.errors_count = metrics.errors_count
            db_metrics.avg_response_time = metrics.avg_response_time
            db_metrics.updated_at = metrics.updated_at
            
            self.session.commit()
            logger.debug(f"Updated daily metrics for {metrics.company_id} on {metrics.date}")
            
        except SQLAlchemyError as e:
            logger.error(f"Error updating daily metrics: {e}")
            self.session.rollback()
            raise
    
    async def get_company_usage_report(
        self, 
        company_id: str, 
        start_date: date, 
        end_date: date
    ) -> Dict[str, Any]:
        """Get usage report for date range"""
        try:
            metrics = self.session.query(CompanyUsageMetrics).filter(
                CompanyUsageMetrics.company_id == company_id,
                CompanyUsageMetrics.date >= start_date,
                CompanyUsageMetrics.date <= end_date
            ).order_by(CompanyUsageMetrics.date).all()
            
            if not metrics:
                return {"error": "No data found for the specified period"}
            
            # Calculate totals
            totals = {
                "total_connections": sum(m.total_connections for m in metrics),
                "total_messages": sum(m.total_messages for m in metrics),
                "total_tokens": sum(m.total_tokens for m in metrics),
                "total_voice_calls": sum(m.total_voice_calls for m in metrics),
                "total_voice_minutes": sum(m.total_voice_minutes for m in metrics),
                "total_connection_hours": sum(m.connection_hours for m in metrics),
                "tickets_created": sum(m.tickets_created for m in metrics),
                "errors_count": sum(m.errors_count for m in metrics)
            }
            
            # Daily breakdown
            daily_data = [
                {
                    "date": m.date.isoformat(),
                    "connections": m.total_connections,
                    "messages": m.total_messages,
                    "tokens": m.total_tokens,
                    "voice_calls": m.total_voice_calls,
                    "voice_minutes": m.total_voice_minutes,
                    "connection_hours": m.connection_hours,
                    "avg_response_time": m.avg_response_time,
                    "errors": m.errors_count
                }
                for m in metrics
            ]
            
            return {
                "company_id": company_id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": len(metrics)
                },
                "totals": totals,
                "daily_breakdown": daily_data
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting usage report: {e}")
            raise