"""UNC-SEW 算法服务（A038 复现成果的服务器化）。

平台契约（lym-lab/algorithm_research_frontend）：
- POST /process  multipart/form-data：input_file(文件) + method_id + parameters(JSON 字符串)
- 200 → {"message": str, "artifacts": [{"kind","label","filename","url"}, ...]}（1-20 个）
- 失败 → 非 200 + {"error": "用户可理解说明"}
- 结果文件独立 URL 可下载，全链 CORS；同步等待（无轮询）。

运维：
- GPU 串行锁（A038_CONCURRENCY，默认 1）；排队满返回 429
- 结果 TTL 清理（A038_TTL_H，默认 1 小时；后台线程周期删除）
- 时长护栏 3-30s（A038_MAX_SECONDS 可调）；codec 由 A038_CODEC 选择（mock|semanticodec）
- 嵌入产物为 float32 WAV——PCM16/MP3 等重量化会摧毁 LSB 水印（论文口径：理想信道）

启动：uvicorn app:app --host 0.0.0.0 --port 8710
冒烟：python smoke_test.py
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import service

BASE = Path(__file__).parent
OUTDIR = BASE / "out"
OUTDIR.mkdir(exist_ok=True)
METHOD_ID = "a038_unc_sew"
TTL_H = float(__import__("os").environ.get("A038_TTL_H", "1"))
MAX_MB = 20
GPU_SEM = threading.Semaphore(int(__import__("os").environ.get("A038_CONCURRENCY", "1")))
QUEUE_FULL = 2  # 等待队列上限（含正在处理的一个）

app = FastAPI(title="UNC-SEW watermark service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    expose_headers=["Content-Disposition"])


# ---- 契约错误格式：{"error": ...} 而非 FastAPI 默认 {"detail": ...} ----
@app.exception_handler(HTTPException)
async def _http_err(request: Request, exc: HTTPException):
    return JSONError(exc.status_code, str(exc.detail))


def JSONError(status: int, message: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONError(500, f"服务内部错误：{type(exc).__name__}")


# ---- TTL 清理线程 ----
def _cleaner():
    while True:
        time.sleep(600)
        deadline = time.time() - TTL_H * 3600
        for d in OUTDIR.iterdir():
            if d.is_dir() and d.stat().st_mtime < deadline:
                shutil.rmtree(d, ignore_errors=True)


threading.Thread(target=_cleaner, daemon=True).start()


@app.post("/process")
async def process(
    input_file: UploadFile = File(...),
    method_id: str = Form(...),
    parameters: str = Form("{}"),
):
    if method_id != METHOD_ID:
        raise HTTPException(400, f"方法不匹配：期望 {METHOD_ID}，收到 {method_id}")
    try:
        params = json.loads(parameters) if parameters and parameters != "null" else {}
        if not isinstance(params, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "parameters 必须是 JSON 对象字符串")

    data = await input_file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"文件超过 {MAX_MB} MB 限制")
    if not data:
        raise HTTPException(400, "上传文件为空")

    if not GPU_SEM.acquire(blocking=False):
        raise HTTPException(429, "服务正忙（GPU 串行），请稍后重试")
    token = uuid.uuid4().hex[:12]
    out = OUTDIR / token
    out.mkdir(exist_ok=True)
    try:
        result = await _run_in_thread(service.process, data, input_file.filename or "input.wav",
                                      params, out)
    except ValueError as e:      # 算法层护栏（时长/格式/参数值）→ 400 可读错误
        shutil.rmtree(out, ignore_errors=True)
        raise HTTPException(400, str(e))
    finally:
        GPU_SEM.release()
    return {
        "message": result["message"],
        "artifacts": [
            {"kind": a["kind"], "label": a["label"], "filename": a["filename"],
             "url": f"/files/{token}/{a['filename']}"}
            for a in result["artifacts"]],
    }


async def _run_in_thread(func, *args):
    import anyio
    return await anyio.to_thread.run_sync(lambda: func(*args))


@app.get("/files/{token}/{filename}")
async def files(token: str, filename: str):
    if "/" in token or "/" in filename or ".." in (token, filename):
        raise HTTPException(400, "非法路径")
    p = OUTDIR / token / filename
    if not p.is_file():
        raise HTTPException(404, "结果不存在或已过期（结果保留 1 小时）")
    return FileResponse(p, filename=filename)


@app.get("/health")
async def health():
    return {"status": "ok", "method": METHOD_ID, "codec": service.CODEC_NAME}
