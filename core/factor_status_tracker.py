"""
factor_status_tracker.py

Factor status tracking system for monitoring the progress of each factor
through different stages of analysis and optimization.

Features:
- Detailed status tracking (pending, checking, stage1, stage2, optimizing, completed, failed)
- Status history with timestamps
- Status query and update methods
- Integration with ContextManager
- Status statistics and reporting
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class FactorStatus(Enum):
    """Factor status enumeration."""
    PENDING = "pending"                    # Not started
    CHECKING = "checking"                  # Checking data quality
    STAGE1_RUNNING = "stage1_running"      # Stage 1 grid search in progress
    STAGE1_COMPLETED = "stage1_completed"  # Stage 1 completed
    STAGE1_FAILED = "stage1_failed"        # Stage 1 failed
    STAGE2_RUNNING = "stage2_running"      # Stage 2 LLM optimization in progress
    STAGE2_COMPLETED = "stage2_completed"  # Stage 2 completed
    STAGE2_FAILED = "stage2_failed"        # Stage 2 failed
    OPTIMIZING = "optimizing"              # Additional optimization in progress
    COMPLETED = "completed"                # Fully completed
    FAILED = "failed"                      # Overall failed
    SKIPPED = "skipped"                    # Skipped (e.g., didn't meet thresholds)


class FactorStatusTracker:
    """
    Tracks detailed status of each factor through analysis pipeline.
    
    Status flow:
    pending -> checking -> stage1_running -> stage1_completed -> stage2_running -> stage2_completed -> completed
    Or: pending -> checking -> stage1_running -> stage1_failed -> failed
    Or: pending -> checking -> stage1_completed -> skipped (if thresholds not met)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize factor status tracker.
        
        :param db_path: Path to SQLite database file
                       If None, uses default: data/context/factor_status.db
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent
            db_dir = project_root / "data" / "context"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "factor_status.db")
        
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        
        logging.info(f"Factor Status Tracker initialized")
        logging.info(f"  Database: {db_path}")
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
        self._create_indexes()
        
        logging.info("Status database initialized successfully")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main status table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS factor_status (
            factor_id TEXT PRIMARY KEY,
            factor_name TEXT NOT NULL,
            asset TEXT,
            current_status TEXT NOT NULL,
            previous_status TEXT,
            stage1_result TEXT,
            stage2_result TEXT,
            best_sharpe REAL DEFAULT 0.0,
            best_calmar REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT
        )
        """)
        
        # Status history table (for tracking status changes)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id TEXT NOT NULL,
            status TEXT NOT NULL,
            previous_status TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (factor_id) REFERENCES factor_status(factor_id)
        )
        """)
        
        self.conn.commit()
    
    def _create_indexes(self):
        """Create indexes for performance."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status_current_status 
        ON factor_status(current_status)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status_history_factor_id 
        ON status_history(factor_id)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status_history_timestamp 
        ON status_history(timestamp)
        """)
        
        self.conn.commit()
    
    def register_factor(self, factor_id: str, factor_name: str, asset: str = None):
        """
        Register a new factor with initial status.
        
        :param factor_id: Unique factor identifier (e.g., "BTC_sopr")
        :param factor_name: Factor name
        :param asset: Asset symbol
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT OR IGNORE INTO factor_status 
        (factor_id, factor_name, asset, current_status)
        VALUES (?, ?, ?, ?)
        """, (factor_id, factor_name, asset, FactorStatus.PENDING.value))
        
        self.conn.commit()
        
        # Record status change
        self._record_status_change(factor_id, FactorStatus.PENDING.value, None)
        
        logging.debug(f"Registered factor: {factor_id}")
    
    def update_status(self, 
                     factor_id: str,
                     new_status: FactorStatus,
                     metadata: Dict = None,
                     stage_result: Dict = None,
                     error_message: str = None):
        """
        Update factor status.
        
        :param factor_id: Factor identifier
        :param new_status: New status (FactorStatus enum)
        :param metadata: Optional metadata dictionary
        :param stage_result: Optional stage result (for stage1/stage2)
        :param error_message: Optional error message (for failed status)
        """
        cursor = self.conn.cursor()
        
        # Get current status
        cursor.execute("SELECT current_status FROM factor_status WHERE factor_id = ?", (factor_id,))
        row = cursor.fetchone()
        previous_status = row['current_status'] if row else None
        
        # Prepare update data
        update_fields = {
            'current_status': new_status.value,
            'previous_status': previous_status,
            'last_updated': datetime.now()
        }
        
        # Update stage results
        if stage_result:
            if new_status in [FactorStatus.STAGE1_COMPLETED, FactorStatus.STAGE1_FAILED]:
                update_fields['stage1_result'] = json.dumps(stage_result)
            elif new_status in [FactorStatus.STAGE2_COMPLETED, FactorStatus.STAGE2_FAILED]:
                update_fields['stage2_result'] = json.dumps(stage_result)
            
            # Update best metrics if available
            if 'best_result' in stage_result:
                best = stage_result['best_result']
                if 'sharpe_ratio' in best:
                    update_fields['best_sharpe'] = best['sharpe_ratio']
                if 'calmar_ratio' in best:
                    update_fields['best_calmar'] = best['calmar_ratio']
        
        # Update completed_at if status is completed
        if new_status == FactorStatus.COMPLETED:
            update_fields['completed_at'] = datetime.now()
        
        # Update error message if failed
        if error_message:
            update_fields['error_message'] = error_message
        
        # Build and execute update query
        set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
        values = list(update_fields.values()) + [factor_id]
        
        cursor.execute(f"""
        UPDATE factor_status 
        SET {set_clause}
        WHERE factor_id = ?
        """, values)
        
        self.conn.commit()
        
        # Record status change in history
        self._record_status_change(factor_id, new_status.value, previous_status, metadata)
        
        logging.info(f"Status updated: {factor_id} -> {new_status.value}")
    
    def _record_status_change(self, 
                             factor_id: str,
                             status: str,
                             previous_status: Optional[str],
                             metadata: Dict = None):
        """Record status change in history table."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT INTO status_history 
        (factor_id, status, previous_status, metadata)
        VALUES (?, ?, ?, ?)
        """, (factor_id, status, previous_status, 
              json.dumps(metadata) if metadata else None))
        
        self.conn.commit()
    
    def get_status(self, factor_id: str) -> Optional[Dict]:
        """
        Get current status of a factor.
        
        :param factor_id: Factor identifier
        :return: Status dictionary or None if not found
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM factor_status WHERE factor_id = ?
        """, (factor_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        result = dict(row)
        
        # Parse JSON fields
        if result.get('stage1_result'):
            result['stage1_result'] = json.loads(result['stage1_result'])
        if result.get('stage2_result'):
            result['stage2_result'] = json.loads(result['stage2_result'])
        
        return result
    
    def get_status_history(self, factor_id: str, limit: int = 10) -> List[Dict]:
        """
        Get status change history for a factor.
        
        :param factor_id: Factor identifier
        :param limit: Maximum number of history records
        :return: List of status history records
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM status_history 
        WHERE factor_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (factor_id, limit))
        
        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            record = dict(row)
            if record.get('metadata'):
                record['metadata'] = json.loads(record['metadata'])
            results.append(record)
        
        return results
    
    def get_factors_by_status(self, status: FactorStatus) -> List[Dict]:
        """
        Get all factors with a specific status.
        
        :param status: Status to filter by
        :return: List of factor status records
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM factor_status 
        WHERE current_status = ?
        ORDER BY last_updated DESC
        """, (status.value,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_status_statistics(self) -> Dict:
        """
        Get statistics about factor statuses.
        
        :return: Dictionary with status counts and percentages
        """
        cursor = self.conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM factor_status")
        total = cursor.fetchone()['total']
        
        if total == 0:
            return {
                'total': 0,
                'by_status': {},
                'percentages': {}
            }
        
        # Get counts by status
        cursor.execute("""
        SELECT current_status, COUNT(*) as count
        FROM factor_status
        GROUP BY current_status
        """)
        
        by_status = {}
        for row in cursor.fetchall():
            by_status[row['current_status']] = row['count']
        
        # Calculate percentages
        percentages = {
            status: (count / total * 100) 
            for status, count in by_status.items()
        }
        
        return {
            'total': total,
            'by_status': by_status,
            'percentages': percentages
        }
    
    def get_all_statuses(self, limit: int = None) -> List[Dict]:
        """
        Get all factor statuses.
        
        :param limit: Optional limit on number of results
        :return: List of all factor status records
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM factor_status ORDER BY last_updated DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_in_progress_factors(self) -> List[Dict]:
        """
        Get all factors currently in progress.
        
        :return: List of factors with running/active statuses
        """
        in_progress_statuses = [
            FactorStatus.CHECKING.value,
            FactorStatus.STAGE1_RUNNING.value,
            FactorStatus.STAGE2_RUNNING.value,
            FactorStatus.OPTIMIZING.value
        ]
        
        cursor = self.conn.cursor()
        
        placeholders = ",".join(["?" for _ in in_progress_statuses])
        cursor.execute(f"""
        SELECT * FROM factor_status 
        WHERE current_status IN ({placeholders})
        ORDER BY last_updated DESC
        """, in_progress_statuses)
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logging.info("Status tracker database connection closed")


# Global instance (singleton pattern)
_tracker_instance = None

def get_status_tracker(db_path: str = None) -> FactorStatusTracker:
    """
    Get global status tracker instance.
    
    :param db_path: Optional database path
    :return: FactorStatusTracker instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = FactorStatusTracker(db_path)
    return _tracker_instance


if __name__ == "__main__":
    # Test the status tracker
    print("Testing Factor Status Tracker...")
    
    tracker = FactorStatusTracker()
    
    # Test: Register a factor
    tracker.register_factor("BTC_sopr", "sopr", "BTC")
    
    # Test: Update status
    tracker.update_status("BTC_sopr", FactorStatus.CHECKING)
    tracker.update_status("BTC_sopr", FactorStatus.STAGE1_RUNNING)
    
    # Test: Get status
    status = tracker.get_status("BTC_sopr")
    print(f"Current status: {status}")
    
    # Test: Get statistics
    stats = tracker.get_status_statistics()
    print(f"Statistics: {stats}")
    
    tracker.close()
    print("✅ Status tracker tests completed!")
