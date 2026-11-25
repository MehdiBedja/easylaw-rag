@echo off
rem OCR of the Arabic journals of 1970-1973 (resumable: finished journals are skipped).
cd /d G:\easylaw\easylaw-rag
.venv\Scripts\python.exe -u -m easylaw_rag.ingest.pdf_pages --lang ar --years 1970-1973 --workers 12 >> data\interim\pages_ar.log 2>&1
