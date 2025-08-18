"""
Risk Manager for managing trading risks.
Basic implementation for risk monitoring and control.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from config.trading_config import DEFAULT_INITIAL_CAPITAL

class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskRule:
    """Risk rule configuration"""
    name: str
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    max_leverage: float = 1.0
    enabled: bool = True

class RiskManager:
    """
    Basic risk manager for monitoring and controlling trading risks.
    
    Features:
    - Position size limits
    - Daily loss limits
    - Drawdown monitoring
    - Risk alerts
    """
    
    def __init__(self, initial_capital: float = DEFAULT_INITIAL_CAPITAL):
        """Initialize risk manager"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Risk rules
        self.risk_rules = {
            "default": RiskRule(
                name="default",
                max_position_size=0.1,  # 10% of capital
                max_daily_loss=0.05,    # 5% of capital
                max_drawdown=0.15,      # 15% of capital
                max_leverage=1.0
            )
        }
        
        # Risk alerts
        self.risk_alerts = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for risk manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _reset_daily_metrics(self):
        """Reset daily metrics if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logging.info("Daily metrics reset")
    
    def update_capital(self, pnl_change: float):
        """
        Update capital and track metrics.
        
        :param pnl_change: P&L change (positive for profit, negative for loss)
        """
        self._reset_daily_metrics()
        
        self.current_capital += pnl_change
        self.daily_pnl += pnl_change
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        logging.info(f"Capital updated: {self.current_capital:.2f} (Change: {pnl_change:.2f})")
    
    def check_position_size(self, symbol: str, position_value: float) -> Dict[str, Any]:
        """
        Check if position size is within limits.
        
        :param symbol: Trading symbol
        :param position_value: Position value in base currency
        :return: Risk check result
        """
        rule = self.risk_rules.get("default")
        max_position_value = self.current_capital * rule.max_position_size
        
        risk_level = RiskLevel.LOW
        if position_value > max_position_value:
            risk_level = RiskLevel.HIGH
        
        result = {
            "allowed": position_value <= max_position_value,
            "risk_level": risk_level,
            "position_value": position_value,
            "max_allowed": max_position_value,
            "percentage": (position_value / self.current_capital) * 100
        }
        
        if not result["allowed"]:
            self._add_risk_alert(f"Position size limit exceeded for {symbol}: {position_value:.2f} > {max_position_value:.2f}")
        
        return result
    
    def check_daily_loss(self) -> Dict[str, Any]:
        """
        Check if daily loss is within limits.
        
        :return: Risk check result
        """
        rule = self.risk_rules.get("default")
        max_daily_loss = self.initial_capital * rule.max_daily_loss
        
        risk_level = RiskLevel.LOW
        if self.daily_pnl < -max_daily_loss:
            risk_level = RiskLevel.CRITICAL
        elif self.daily_pnl < -max_daily_loss * 0.8:
            risk_level = RiskLevel.HIGH
        
        result = {
            "allowed": self.daily_pnl >= -max_daily_loss,
            "risk_level": risk_level,
            "daily_pnl": self.daily_pnl,
            "max_daily_loss": -max_daily_loss,
            "percentage": (self.daily_pnl / self.initial_capital) * 100
        }
        
        if not result["allowed"]:
            self._add_risk_alert(f"Daily loss limit exceeded: {self.daily_pnl:.2f} < {max_daily_loss:.2f}")
        
        return result
    
    def check_drawdown(self) -> Dict[str, Any]:
        """
        Check if current drawdown is within limits.
        
        :return: Risk check result
        """
        rule = self.risk_rules.get("default")
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        max_drawdown = rule.max_drawdown
        
        risk_level = RiskLevel.LOW
        if current_drawdown > max_drawdown:
            risk_level = RiskLevel.CRITICAL
        elif current_drawdown > max_drawdown * 0.8:
            risk_level = RiskLevel.HIGH
        
        result = {
            "allowed": current_drawdown <= max_drawdown,
            "risk_level": risk_level,
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "peak_capital": self.peak_capital,
            "current_capital": self.current_capital
        }
        
        if not result["allowed"]:
            self._add_risk_alert(f"Drawdown limit exceeded: {current_drawdown:.2%} > {max_drawdown:.2%}")
        
        return result
    
    def check_overall_risk(self) -> Dict[str, Any]:
        """
        Perform comprehensive risk check.
        
        :return: Overall risk assessment
        """
        position_check = {"status": "not_applicable"}
        daily_loss_check = self.check_daily_loss()
        drawdown_check = self.check_drawdown()
        
        # Determine overall risk level
        risk_levels = [daily_loss_check["risk_level"], drawdown_check["risk_level"]]
        overall_risk = max(risk_levels, key=lambda x: x.value)
        
        result = {
            "overall_risk": overall_risk,
            "daily_loss_check": daily_loss_check,
            "drawdown_check": drawdown_check,
            "position_check": position_check,
            "capital": self.current_capital,
            "daily_pnl": self.daily_pnl,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _add_risk_alert(self, message: str):
        """Add risk alert"""
        alert = {
            "timestamp": datetime.now(),
            "message": message,
            "level": "WARNING"
        }
        self.risk_alerts.append(alert)
        logging.warning(f"Risk Alert: {message}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        risk_check = self.check_overall_risk()
        
        summary = {
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "peak": self.peak_capital,
                "daily_pnl": self.daily_pnl
            },
            "risk_metrics": {
                "daily_loss_percentage": (self.daily_pnl / self.initial_capital) * 100,
                "drawdown_percentage": ((self.peak_capital - self.current_capital) / self.peak_capital) * 100,
                "overall_risk_level": risk_check["overall_risk"].value
            },
            "risk_checks": risk_check,
            "alerts": len(self.risk_alerts),
            "last_updated": datetime.now().isoformat()
        }
        
        return summary
    
    def should_stop_trading(self) -> bool:
        """
        Determine if trading should be stopped due to risk limits.
        
        :return: True if trading should be stopped
        """
        risk_check = self.check_overall_risk()
        
        # Stop trading if any critical risk level is reached
        if risk_check["overall_risk"] == RiskLevel.CRITICAL:
            logging.critical("Trading stopped due to critical risk level")
            return True
        
        # Stop trading if daily loss limit exceeded
        if not risk_check["daily_loss_check"]["allowed"]:
            logging.critical("Trading stopped due to daily loss limit")
            return True
        
        # Stop trading if drawdown limit exceeded
        if not risk_check["drawdown_check"]["allowed"]:
            logging.critical("Trading stopped due to drawdown limit")
            return True
        
        return False
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent risk alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.risk_alerts
            if alert["timestamp"] > cutoff_time
        ]
        return recent_alerts
    
    def add_risk_rule(self, rule_name: str, rule: RiskRule):
        """Add a new risk rule"""
        self.risk_rules[rule_name] = rule
        logging.info(f"Added risk rule: {rule_name}")
    
    def update_risk_rule(self, rule_name: str, **kwargs):
        """Update an existing risk rule"""
        if rule_name in self.risk_rules:
            rule = self.risk_rules[rule_name]
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logging.info(f"Updated risk rule: {rule_name}")
        else:
            logging.error(f"Risk rule {rule_name} not found") 