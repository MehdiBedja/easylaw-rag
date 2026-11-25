@echo off
rem French titles and references for every text (resumable).
cd /d G:\easylaw\easylaw-rag
.venv\Scripts\python.exe -u -m easylaw_rag.ingest.joradp_search --years 1962-2026 >> data\raw\joradp\search.log 2>&1
