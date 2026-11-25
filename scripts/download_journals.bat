@echo off
rem Journal PDFs, one request at a time (resumable). Order: Arabic gap years, modern French (text layer,
rem no OCR needed), then the older French scans.
cd /d G:\easylaw\easylaw-rag
.venv\Scripts\python.exe -u -m easylaw_rag.ingest.joradp --lang ar --years 1970-1973 >> data\raw\joradp\download.log 2>&1
.venv\Scripts\python.exe -u -m easylaw_rag.ingest.joradp --lang fr --years 2001-2026 >> data\raw\joradp\download.log 2>&1
.venv\Scripts\python.exe -u -m easylaw_rag.ingest.joradp --lang fr >> data\raw\joradp\download.log 2>&1
