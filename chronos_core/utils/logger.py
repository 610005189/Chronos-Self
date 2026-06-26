"""
Chronos-Self Logging System
===========================

This module provides a comprehensive logging system for the Chronos-Self project.
It supports multi-level logging with both file and console output, including
timestamps, module names, and log levels.

Features:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Simultaneous file and console output
- Structured logging with contextual information
- Rotation and backup of log files
- Integration with the configuration system
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ChronosFormatter(logging.Formatter):
    """
    Custom formatter for Chronos-Self logs with enhanced formatting.
    
    Format includes:
    - Timestamp with milliseconds
    - Log level with color coding (for console)
    - Module name
    - Function name
    - Line number
    - Message
    - Optional context dictionary
    """
    
    # ANSI color codes for console output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        """Initialize the formatter."""
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        # Create timestamp with milliseconds
        timestamp = datetime.fromtimestamp(record.created).strftime(
            '%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Get log level
        level = record.levelname
        
        # Get module, function, and line info
        module = record.name
        function = record.funcName if record.funcName != '<module>' else 'module'
        line = record.lineno
        
        # Get the base message
        message = record.getMessage()
        
        # Format the main log line
        if self.use_colors:
            color = self.COLORS.get(level, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            log_line = (
                f"{timestamp} | {color}{level:8s}{reset} | "
                f"{module:30s} | {function:20s} | line {line:4d} | {message}"
            )
        else:
            log_line = (
                f"{timestamp} | {level:8s} | {module:30s} | "
                f"{function:20s} | line {line:4d} | {message}"
            )
        
        # Add context if available
        if hasattr(record, 'context') and record.context:
            context_str = json.dumps(record.context, indent=2, ensure_ascii=False)
            log_line += f"\n    Context: {context_str}"
        
        # Add exception info if available
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"
        
        return log_line


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging to files.
    
    Outputs log records as JSON objects for easier parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """
    Filter that adds contextual information to log records.
    
    This allows adding extra context to log messages without
    changing the logging call signature.
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize the filter with optional context."""
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the record."""
        if not hasattr(record, 'context'):
            record.context = {}
        record.context.update(self.context)
        return True
    
    def update_context(self, **kwargs) -> None:
        """Update the context dictionary."""
        self.context.update(kwargs)


class ChronosLogger:
    """
    Main logging class for Chronos-Self.
    
    Provides a unified interface for logging with both file and console output,
    contextual information, and configurable log levels.
    """
    
    def __init__(
        self,
        name: str = "ChronosSelf",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_logging: bool = True,
        file_logging: bool = True,
        log_rotation_size: int = 10,  # MB
        log_backup_count: int = 5,
        use_colors: bool = True,
        use_json_format: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Chronos logger.
        
        Args:
            name: Logger name
            log_level: Minimum log level to record
            log_file: Path to log file (if None, uses default)
            console_logging: Enable console output
            file_logging: Enable file output
            log_rotation_size: Maximum log file size in MB before rotation
            log_backup_count: Number of backup log files to keep
            use_colors: Use colored output for console
            use_json_format: Use JSON format for file logs
            context: Default context dictionary for all logs
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Add context filter
        self.context_filter = ContextFilter(context)
        self.logger.addFilter(self.context_filter)
        
        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(ChronosFormatter(use_colors=use_colors))
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_logging:
            if log_file is None:
                log_file = "logs/chronos_self.log"
            
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=log_rotation_size * 1024 * 1024,  # Convert MB to bytes
                backupCount=log_backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            
            if use_json_format:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(ChronosFormatter(use_colors=False))
            
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **context) -> None:
        """Log a debug message."""
        if context:
            old_context = self.context_filter.context.copy()
            self.context_filter.update_context(**context)
            self.logger.debug(message)
            self.context_filter.context = old_context
        else:
            self.logger.debug(message)
    
    def info(self, message: str, **context) -> None:
        """Log an info message."""
        if context:
            old_context = self.context_filter.context.copy()
            self.context_filter.update_context(**context)
            self.logger.info(message)
            self.context_filter.context = old_context
        else:
            self.logger.info(message)
    
    def warning(self, message: str, **context) -> None:
        """Log a warning message."""
        if context:
            old_context = self.context_filter.context.copy()
            self.context_filter.update_context(**context)
            self.logger.warning(message)
            self.context_filter.context = old_context
        else:
            self.logger.warning(message)
    
    def error(self, message: str, **context) -> None:
        """Log an error message."""
        if context:
            old_context = self.context_filter.context.copy()
            self.context_filter.update_context(**context)
            self.logger.error(message)
            self.context_filter.context = old_context
        else:
            self.logger.error(message)
    
    def critical(self, message: str, **context) -> None:
        """Log a critical message."""
        if context:
            old_context = self.context_filter.context.copy()
            self.context_filter.update_context(**context)
            self.logger.critical(message)
            self.context_filter.context = old_context
        else:
            self.logger.critical(message)
    
    def exception(self, message: str, **context) -> None:
        """Log an exception message with traceback."""
        if context:
            old_context = self.context_filter.context.copy()
            self.context_filter.update_context(**context)
            self.logger.exception(message)
            self.context_filter.context = old_context
        else:
            self.logger.exception(message)
    
    def update_context(self, **kwargs) -> None:
        """Update the default context for all log messages."""
        self.context_filter.update_context(**kwargs)
    
    def set_level(self, level: str) -> None:
        """Set the log level."""
        self.logger.setLevel(getattr(logging, level.upper()))
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, level.upper()))
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """Get a child logger for a specific module."""
        return self.logger.getChild(module_name)


# Global logger instance
_global_logger: Optional[ChronosLogger] = None


def get_logger() -> ChronosLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ChronosLogger()
    return _global_logger


def set_logger(logger: ChronosLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger


def init_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_logging: bool = True,
    file_logging: bool = True,
    use_colors: bool = True,
    use_json_format: bool = False,
    **kwargs
) -> ChronosLogger:
    """
    Initialize the global logger.
    
    Args:
        log_level: Minimum log level to record
        log_file: Path to log file
        console_logging: Enable console output
        file_logging: Enable file output
        use_colors: Use colored output for console
        use_json_format: Use JSON format for file logs
        **kwargs: Additional context for all logs
    
    Returns:
        Initialized logger instance
    """
    global _global_logger
    _global_logger = ChronosLogger(
        log_level=log_level,
        log_file=log_file,
        console_logging=console_logging,
        file_logging=file_logging,
        use_colors=use_colors,
        use_json_format=use_json_format,
        context=kwargs
    )
    return _global_logger


# Convenience functions for module-level logging
def debug(message: str, **context) -> None:
    """Log a debug message using the global logger."""
    get_logger().debug(message, **context)


def info(message: str, **context) -> None:
    """Log an info message using the global logger."""
    get_logger().info(message, **context)


def warning(message: str, **context) -> None:
    """Log a warning message using the global logger."""
    get_logger().warning(message, **context)


def error(message: str, **context) -> None:
    """Log an error message using the global logger."""
    get_logger().error(message, **context)


def critical(message: str, **context) -> None:
    """Log a critical message using the global logger."""
    get_logger().critical(message, **context)


def exception(message: str, **context) -> None:
    """Log an exception message using the global logger."""
    get_logger().exception(message, **context)