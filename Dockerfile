FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# agent 主程序 + API 层 + RAG 数据文件一起打进镜像
COPY agent_v04.py server.py rag_v03_overlap.py ./ 
COPY corpus/ ./corpus/
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
