# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Wrapper to run Abliterix on Windows without GBK encoding errors.

Rich's Win32 console writer bypasses PYTHONIOENCODING and uses the native
Windows console code page (GBK/cp936), causing UnicodeEncodeError on Unicode
box-drawing characters. This wrapper replaces sys.stdout/stderr with UTF-8
TextIOWrappers BEFORE Rich's Console is initialized, forcing it to use the
plain file-based writer instead.
"""

import sys

from abliterix.scriptlib import setup_io

setup_io()

# Now import and run abliterix — Rich will see non-console file handles
sys.argv = ["abliterix"] + sys.argv[1:]
from abliterix.cli import main  # noqa: E402

main()
