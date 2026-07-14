#!/usr/bin/env python
"""Utilitário de linha de comando do Django."""

import os
import sys


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

    try:
        from django.core.management import execute_from_command_line
    except ImportError as erro:
        raise ImportError(
            'Django não encontrado. Ative o ambiente virtual e rode: '
            'pip install -r requirements.txt'
        ) from erro

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
