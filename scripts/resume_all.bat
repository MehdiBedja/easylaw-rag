@echo off
rem Long job of the moment, started through Task Scheduler (task "EasyLawResume") or by double-click so
rem that it does not depend on another program staying open. Safe to re-run: every step resumes.
rem
rem Current job: answer evaluation (30 questions x French and Arabic through the full system, then the LLM judge).
cd /d G:\easylaw\easylaw-rag
set PYTHONIOENCODING=utf-8
echo === %date% %time% answer evaluation started >> data\eval_answers.log
.venv\Scripts\python.exe -u -m easylaw_rag.eval.answers generate >> data\eval_answers.log 2>&1
.venv\Scripts\python.exe -u -m easylaw_rag.eval.answers judge >> data\eval_answers.log 2>&1
.venv\Scripts\python.exe -u -m easylaw_rag.eval.answers generate >> data\eval_answers.log 2>&1
.venv\Scripts\python.exe -u -m easylaw_rag.eval.answers judge >> data\eval_answers.log 2>&1
.venv\Scripts\python.exe -u -m easylaw_rag.eval.answers report >> data\eval_answers.log 2>&1
echo === %date% %time% answer evaluation exited with code %errorlevel% >> data\eval_answers.log
