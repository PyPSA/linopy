"""
Remote execution handlers for linopy models.

This module provides different handlers for executing optimization models
on remote systems:

- RemoteHandler: SSH-based remote execution using paramiko
- OetcHandler: Cloud-based execution via OET Cloud service
"""

from linopy.remote.ssh import RemoteHandler

try:
    from linopy.remote.oetc import OetcCredentials, OetcHandler, OetcSettings
except ImportError:
    pass

__all__ = [
    "RemoteHandler",
    "OetcHandler",
    "OetcSettings",
    "OetcCredentials",
]
