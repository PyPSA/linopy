#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 23:06:41 2022.

@author: fabian
"""

from getpass import getpass

from netmiko import ConnectHandler

# Just pick an 'invalid' device_type
cisco1 = {
    "device_type": "linux",
    "host": "gridlock.fias.uni-frankfurt.de",
    "username": "hofmann",
    "password": "?grysVAC0",
}

net_connect = ConnectHandler(**cisco1)
# net_connect.disconnect()
