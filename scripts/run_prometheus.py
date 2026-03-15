# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Wrapper to run Prometheus on Windows without GBK encoding errors.

Rich's Win32 console writer bypasses PYTHONIOENCODING and uses the native
Windows console code page (GBK/cp936), causing UnicodeEncodeError on Unicode
box-drawing characters. This wrapper replaces sys.stdout/stderr with UTF-8
TextIOWrappers BEFORE Rich's Console is initialized, forcing it to use the
plain file-based writer instead.
"""

import sys

from prometheus.scriptlib import setup_io

setup_io()

# Now import and run prometheus — Rich will see non-console file handles
sys.argv = ["prometheus"] + sys.argv[1:]
from prometheus.cli import main  # noqa: E402

main()
