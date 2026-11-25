@echo off
rem French 1962-2000 (scanned journals, OCR). Starts once the modern French batch is indexed (one GPU job
rem at a time) and the older journals are downloaded. Safe to re-run: each step resumes.
cd /d G:\easylaw\easylaw-rag
.venv\Scripts\python.exe -u -m easylaw_rag.pipeline --lang fr --years 1962-2000 --wait-for data\raw\joradp\download.log "fr: finished#2" --wait-for data\pipeline.log "fr 2001-2026: indexed" >> data\pipeline_old_french.log 2>&1
