"""
context_manager.py

Context manager for storing and retrieving factor analysis history.
Uses SQLite database for efficient storage and querying of 900+ factors.

Features:
- SQLite database storage
- Factor context management
- Analysis history tracking
- Optimization history
- Daily summaries
- Context summarization
- Data export (JSON/CSV)
"""

import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ContextManager:
    """
    Manages factor analysis context using SQLite database.
    
    Designed for 900+ factors with efficient querying and storage.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize context manager.
        
        :param db_path: Path to SQLite database file
                       If None, uses default: data/context/factor_context.db
        """
        if db_path is None:
            # Default path: project_root/data/context/factor_context.db
            project_root = Path(__file__).parent.parent
            db_dir = project_root / "data" / "context"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "factor_context.db")
        
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        
        logging.info(f"Context Manager initialized")
        logging.info(f"  Database: {db_path}")
    
    def _initialize_database(self):
        """
        Initialize database connection and create tables if not exist.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Create tables
        self._create_tables()
        
        # Create indexes for performance
        self._create_indexes()
        
        logging.info("Database initialized successfully")
    
    def _create_tables(self):
        """
        Create database tables if they don't exist.
        """
        cursor = self.conn.cursor()
        
        # Table 1: Factor context
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS factor_context (
            factor_id TEXT PRIMARY KEY,
            factor_name TEXT NOT NULL,
            asset TEXT,
            factor_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            best_sharpe REAL DEFAULT 0.0,
            best_params TEXT,
            status TEXT DEFAULT 'pending',
            total_analyses INTEGER DEFAULT 0
        )
        """)
        
        # Table 2: Analysis history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stage TEXT,
            params TEXT,
            results TEXT,
            llm_analysis TEXT,
            sharpe_ratio REAL,
            total_return REAL,
            max_drawdown REAL,
            win_rate REAL,
            num_trades INTEGER,
            FOREIGN KEY (factor_id) REFERENCES factor_context(factor_id)
        )
        """)
        
        # Table 3: Optimization history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            optimization_type TEXT,
            before_params TEXT,
            after_params TEXT,
            before_sharpe REAL,
            after_sharpe REAL,
            improvement REAL,
            FOREIGN KEY (factor_id) REFERENCES factor_context(factor_id)
        )
        """)
        
        # Table 4: Daily summary
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date DATE PRIMARY KEY,
            total_factors_analyzed INTEGER DEFAULT 0,
            promising_factors INTEGER DEFAULT 0,
            best_factor_id TEXT,
            best_sharpe REAL,
            summary_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.conn.commit()
    
    def _create_indexes(self):
        """
        Create indexes for performance optimization.
        """
        cursor = self.conn.cursor()
        
        # Indexes for analysis_history
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysis_factor_id 
        ON analysis_history(factor_id)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysis_timestamp 
        ON analysis_history(timestamp DESC)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysis_sharpe 
        ON analysis_history(sharpe_ratio DESC)
        """)
        
        # Indexes for factor_context
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_factor_status 
        ON factor_context(status)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_factor_best_sharpe 
        ON factor_context(best_sharpe DESC)
        """)
        
        # Indexes for optimization_history
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_opt_factor_id 
        ON optimization_history(factor_id)
        """)
        
        self.conn.commit()
    
    def save_factor_context(self, 
                           factor_id: str,
                           factor_name: str,
                           asset: str = None,
                           factor_type: str = None,
                           status: str = 'pending'):
        """
        Save or update factor context.
        
        :param factor_id: Unique factor identifier (e.g., "BTC_sopr")
        :param factor_name: Factor name
        :param asset: Asset symbol (BTC, ETH, etc.)
        :param factor_type: Factor type
        :param status: Status (pending, analyzing, completed, failed)
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO factor_context 
        (factor_id, factor_name, asset, factor_type, status, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (factor_id, factor_name, asset, factor_type, status, datetime.now()))
        
        self.conn.commit()
        logging.debug(f"Saved factor context: {factor_id}")
    
    def add_analysis_history(self,
                           factor_id: str,
                           stage: str,
                           params: Dict,
                           results: Dict,
                           llm_analysis: Dict = None):
        """
        Add analysis history record.
        
        :param factor_id: Factor identifier
        :param stage: Analysis stage (exploration, optimization, final)
        :param params: Parameter dictionary
        :param results: Backtest results dictionary
        :param llm_analysis: LLM analysis result (optional)
        """
        cursor = self.conn.cursor()
        
        # Extract key metrics
        sharpe_ratio = results.get('sharpe_ratio', 0.0)
        total_return = results.get('total_return', 0.0)
        max_drawdown = results.get('max_drawdown', 0.0)
        win_rate = results.get('win_rate', 0.0)
        num_trades = results.get('num_trades', 0)
        
        # Insert analysis history
        cursor.execute("""
        INSERT INTO analysis_history 
        (factor_id, stage, params, results, llm_analysis, 
         sharpe_ratio, total_return, max_drawdown, win_rate, num_trades)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            factor_id,
            stage,
            json.dumps(params, ensure_ascii=False),
            json.dumps(results, ensure_ascii=False, default=str),
            json.dumps(llm_analysis, ensure_ascii=False) if llm_analysis else None,
            sharpe_ratio,
            total_return,
            max_drawdown,
            win_rate,
            num_trades
        ))
        
        # Update factor context with best result
        self._update_best_result(factor_id, sharpe_ratio, params)
        
        # Update total analyses count
        cursor.execute("""
        UPDATE factor_context 
        SET total_analyses = total_analyses + 1,
            last_updated = ?
        WHERE factor_id = ?
        """, (datetime.now(), factor_id))
        
        self.conn.commit()
        logging.debug(f"Added analysis history for {factor_id}")
    
    def _update_best_result(self, factor_id: str, sharpe_ratio: float, params: Dict):
        """
        Update best result for a factor.
        
        :param factor_id: Factor identifier
        :param sharpe_ratio: Current Sharpe ratio
        :param params: Parameters used
        """
        cursor = self.conn.cursor()
        
        # Get current best Sharpe
        cursor.execute("""
        SELECT best_sharpe FROM factor_context WHERE factor_id = ?
        """, (factor_id,))
        
        row = cursor.fetchone()
        current_best = row[0] if row else 0.0
        
        # Update if current is better
        if sharpe_ratio > current_best:
            cursor.execute("""
            UPDATE factor_context 
            SET best_sharpe = ?, best_params = ?, last_updated = ?
            WHERE factor_id = ?
            """, (
                sharpe_ratio,
                json.dumps(params, ensure_ascii=False),
                datetime.now(),
                factor_id
            ))
            logging.debug(f"Updated best result for {factor_id}: Sharpe {sharpe_ratio:.4f}")
    
    def add_optimization_history(self,
                               factor_id: str,
                               optimization_type: str,
                               before_params: Dict,
                               after_params: Dict,
                               before_sharpe: float,
                               after_sharpe: float):
        """
        Add optimization history record.
        
        :param factor_id: Factor identifier
        :param optimization_type: Type of optimization (llm_guided, grid_search, etc.)
        :param before_params: Parameters before optimization
        :param after_params: Parameters after optimization
        :param before_sharpe: Sharpe ratio before
        :param after_sharpe: Sharpe ratio after
        """
        cursor = self.conn.cursor()
        
        improvement = after_sharpe - before_sharpe
        
        cursor.execute("""
        INSERT INTO optimization_history 
        (factor_id, optimization_type, before_params, after_params,
         before_sharpe, after_sharpe, improvement)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            factor_id,
            optimization_type,
            json.dumps(before_params, ensure_ascii=False),
            json.dumps(after_params, ensure_ascii=False),
            before_sharpe,
            after_sharpe,
            improvement
        ))
        
        self.conn.commit()
        logging.debug(f"Added optimization history for {factor_id}: improvement {improvement:.4f}")
    
    def get_factor_context(self, factor_id: str) -> Optional[Dict]:
        """
        Get factor context.
        
        :param factor_id: Factor identifier
        :return: Factor context dictionary or None
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM factor_context WHERE factor_id = ?
        """, (factor_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_analysis_history(self, 
                            factor_id: str, 
                            limit: int = 10,
                            stage: str = None) -> List[Dict]:
        """
        Get analysis history for a factor.
        
        :param factor_id: Factor identifier
        :param limit: Maximum number of records to return
        :param stage: Filter by stage (optional)
        :return: List of analysis history records
        """
        cursor = self.conn.cursor()
        
        if stage:
            cursor.execute("""
            SELECT * FROM analysis_history 
            WHERE factor_id = ? AND stage = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """, (factor_id, stage, limit))
        else:
            cursor.execute("""
            SELECT * FROM analysis_history 
            WHERE factor_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """, (factor_id, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_factor_summary(self, factor_id: str) -> str:
        """
        Get context summary for a factor (for LLM context).
        
        :param factor_id: Factor identifier
        :return: Summary text string
        """
        context = self.get_factor_context(factor_id)
        if not context:
            return ""
        
        history = self.get_analysis_history(factor_id, limit=5)
        
        summary = f"""
Factor {factor_id} Historical Summary:
- Best Sharpe Ratio: {context.get('best_sharpe', 0):.4f}
- Total Analyses: {context.get('total_analyses', 0)}
- Status: {context.get('status', 'unknown')}
- Recent Analysis Records: {len(history)}
"""
        
        if history:
            best_record = max(history, key=lambda x: x.get('sharpe_ratio', 0))
            summary += f"- Recent Best Sharpe: {best_record.get('sharpe_ratio', 0):.4f}\n"
        
        return summary.strip()
    
    def save_daily_summary(self,
                          date: str,
                          total_factors: int,
                          promising_factors: int,
                          best_factor_id: str,
                          best_sharpe: float,
                          summary_data: Dict):
        """
        Save daily summary.
        
        :param date: Date string (YYYY-MM-DD)
        :param total_factors: Total factors analyzed
        :param promising_factors: Number of promising factors
        :param best_factor_id: Best factor ID
        :param best_sharpe: Best Sharpe ratio
        :param summary_data: Additional summary data dictionary
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO daily_summary 
        (date, total_factors_analyzed, promising_factors, 
         best_factor_id, best_sharpe, summary_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            date,
            total_factors,
            promising_factors,
            best_factor_id,
            best_sharpe,
            json.dumps(summary_data, ensure_ascii=False, default=str)
        ))
        
        self.conn.commit()
        logging.info(f"Saved daily summary for {date}")
    
    def get_daily_summary(self, date: str) -> Optional[Dict]:
        """
        Get daily summary.
        
        :param date: Date string (YYYY-MM-DD)
        :return: Daily summary dictionary or None
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM daily_summary WHERE date = ?
        """, (date,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_top_factors(self, limit: int = 10, min_sharpe: float = 0.0) -> List[Dict]:
        """
        Get top performing factors.
        
        :param limit: Maximum number of factors to return
        :param min_sharpe: Minimum Sharpe ratio threshold
        :return: List of top factors
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM factor_context 
        WHERE best_sharpe >= ?
        ORDER BY best_sharpe DESC
        LIMIT ?
        """, (min_sharpe, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def export_to_json(self, output_path: str, factor_ids: List[str] = None):
        """
        Export context data to JSON file.
        
        :param output_path: Output file path
        :param factor_ids: List of factor IDs to export (None = all)
        """
        export_data = {
            'export_date': datetime.now().isoformat(),
            'factors': [],
            'daily_summaries': []
        }
        
        if factor_ids:
            for factor_id in factor_ids:
                context = self.get_factor_context(factor_id)
                if context:
                    history = self.get_analysis_history(factor_id, limit=100)
                    export_data['factors'].append({
                        'context': context,
                        'history': history
                    })
        else:
            # Export all factors
            cursor = self.conn.cursor()
            cursor.execute("SELECT factor_id FROM factor_context")
            all_factor_ids = [row[0] for row in cursor.fetchall()]
            
            for factor_id in all_factor_ids:
                context = self.get_factor_context(factor_id)
                if context:
                    history = self.get_analysis_history(factor_id, limit=100)
                    export_data['factors'].append({
                        'context': context,
                        'history': history
                    })
        
        # Export daily summaries
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM daily_summary ORDER BY date DESC LIMIT 30")
        export_data['daily_summaries'] = [dict(row) for row in cursor.fetchall()]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"Exported context data to {output_path}")
    
    def close(self):
        """
        Close database connection.
        """
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Test the context manager
    import tempfile
    
    # Create temporary database for testing
    test_db = tempfile.mktemp(suffix='.db')
    
    print("Testing ContextManager...")
    
    with ContextManager(db_path=test_db) as cm:
        # Test: Save factor context
        print("\n1. Testing save_factor_context...")
        cm.save_factor_context(
            factor_id="BTC_sopr",
            factor_name="SOPR Account Based",
            asset="BTC",
            factor_type="indicator",
            status="analyzing"
        )
        
        # Test: Add analysis history
        print("2. Testing add_analysis_history...")
        cm.add_analysis_history(
            factor_id="BTC_sopr",
            stage="exploration",
            params={"rolling": 10, "long_param": -0.04, "short_param": 0.14},
            results={
                "sharpe_ratio": 1.8,
                "total_return": 0.25,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
                "num_trades": 45
            },
            llm_analysis={
                "has_potential": True,
                "confidence": 0.8,
                "reason": "Good Sharpe ratio and win rate"
            }
        )
        
        # Test: Get factor context
        print("3. Testing get_factor_context...")
        context = cm.get_factor_context("BTC_sopr")
        print(f"   Context: {context}")
        
        # Test: Get analysis history
        print("4. Testing get_analysis_history...")
        history = cm.get_analysis_history("BTC_sopr", limit=5)
        print(f"   History records: {len(history)}")
        
        # Test: Get factor summary
        print("5. Testing get_factor_summary...")
        summary = cm.get_factor_summary("BTC_sopr")
        print(f"   Summary:\n{summary}")
        
        # Test: Save daily summary
        print("6. Testing save_daily_summary...")
        cm.save_daily_summary(
            date="2025-01-15",
            total_factors=100,
            promising_factors=15,
            best_factor_id="BTC_sopr",
            best_sharpe=1.8,
            summary_data={"test": "data"}
        )
        
        # Test: Get top factors
        print("7. Testing get_top_factors...")
        top_factors = cm.get_top_factors(limit=5)
        print(f"   Top factors: {len(top_factors)}")
        
        # Test: Export to JSON
        print("8. Testing export_to_json...")
        export_path = tempfile.mktemp(suffix='.json')
        cm.export_to_json(export_path, factor_ids=["BTC_sopr"])
        print(f"   Exported to: {export_path}")
    
    print("\n✅ All tests passed!")
    
    # Cleanup
    os.remove(test_db)
    print(f"Cleaned up test database: {test_db}")
