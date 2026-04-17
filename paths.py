import os

CWD = os.getcwd()
CUDART_DIR = os.path.abspath(os.path.dirname(__file__))
PYSRC = os.path.join(CUDART_DIR, "pysrc")
SCRIPTS = os.path.join(CUDART_DIR, "scripts")
DOCS = os.path.join(CUDART_DIR, "docs")
SRCS = os.path.join(CUDART_DIR, "src")