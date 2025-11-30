@echo off
cd backend
uvicorn app.main:app --reload
