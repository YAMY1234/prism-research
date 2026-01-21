#!/usr/bin/env python3
# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism Multi-Model Server Launch Script.

Usage:
    python -m prism.launch --model-config-file models.json --enable-elastic-memory
    
Or:
    prism-server --model-config-file models.json --enable-elastic-memory
"""

import sys


def main():
    """Main entry point for Prism server."""
    # Apply patches first
    import prism
    
    # Import and launch server
    from prism.multi_model import prepare_server_args, launch_multi_model_server
    
    args = prepare_server_args(sys.argv[1:])
    launch_multi_model_server(args)


if __name__ == "__main__":
    main()
