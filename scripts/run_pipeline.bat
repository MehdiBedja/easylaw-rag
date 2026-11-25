@echo off
rem Everything after the downloads, in order (one GPU job at a time). Safe to re-run: each step resumes.
rem   1. Arabic 1970-1973 (missing from the original scrape): cut texts, embed, load into easylaw_ar
rem   2. French 2001-2026: page text, cut texts, embed, load into easylaw_fr
cd /d G:\easylaw\easylaw-rag
.venv\Scripts\python.exe -u -m easylaw_rag.pipeline --lang ar --years 1970-1973 --skip-pages --wait-for data\interim\pages_ar.log "ar: done" >> data\pipeline.log 2>&1
.venv\Scripts\python.exe -u -m easylaw_rag.pipeline --lang fr --years 2001-2026 --wait-for data\raw\joradp\download.log "fr: finished" --wait-for data\raw\joradp\search.log "finished:" >> data\pipeline.log 2>&1
