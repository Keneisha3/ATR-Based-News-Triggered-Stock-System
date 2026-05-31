"""Make the project root importable so `import atr_news_alert` works in tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
