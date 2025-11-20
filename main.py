from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base
from datetime import datetime, timedelta
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
from sqlalchemy import LargeBinary  # 放在最上方 import 區段
from typing import Optional
from fastapi import UploadFile, File, Form
from fastapi import Depends, HTTPException
from fastapi import Query, Body
from sqlalchemy.orm import Session
import subprocess
from typing import Optional, List # 確保已引入 Optional, List, datetime
from datetime import datetime
from pydantic import BaseModel # 確保 BaseModel 已引入
from io import StringIO # 用於匯出 CSV
import csv # 用於匯出 CSV
from fastapi.responses import StreamingResponse # 用於匯出檔案
import secrets # 用於生成密碼
import string # 用於生成密碼
from sqlalchemy import or_ # 在 import 區塊新增或確認已有
# schemas.py
from pydantic import BaseModel
from sqlalchemy import cast
from io import BytesIO
from sqlalchemy import Table
import threading
RUN_SEM = threading.BoundedSemaphore(value=4)
# 新增：儲存批改任務狀態（task_id -> {status,total,finished,results,error}）
batch_tasks = {}
import os, requests
import json

import tempfile
import shutil
import platform
import subprocess
from pathlib import Path

import shutil, platform
from fastapi import HTTPException
from collections import deque
waiting_queue = deque()
import threading
import uuid


#學生練習
from pydantic import BaseModel
from typing import Optional


# FastAPI 初始化
app = FastAPI()
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
from pydantic import BaseModel
from typing import List
from pydantic import BaseModel
from typing import List

# 建立資料庫連線
DATABASE_URL = "postgresql://postgres:Oo77441166@localhost:5432/project_webai"
engine = create_engine(
    DATABASE_URL,
    pool_size=20,        # 可依你主機調整
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=1800
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BatchFile(BaseModel):
    filename: str
    code: str
    problem_id: int

class BatchGradeRequest(BaseModel):
    files: List[BatchFile]
    
# Pydantic schema
class ProblemBrief(BaseModel):
    id: int
    title: str
    description: str | None = None
    hint: str | None = None

    class Config:
        from_attributes  = True
class PracticeRequest(BaseModel):
    code: str
    problem_id: Optional[int] = None  # 可選，如果是自由練習就不指定題目

class PracticeResult(BaseModel):
    program_output: str
    execution_success: bool
    error_message: Optional[str] = None
    test_results: Optional[list] = None  # 如果有指定題目，會包含測試結果
# 取得資料庫 session 的依賴函式
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# 建立資料表（只要第一次啟動）
Base.metadata.create_all(bind=engine)


# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/homepage.html")

    
@app.get("/api/problems/", response_model=List[ProblemBrief])
def list_problems(db: Session = Depends(get_db)):
    return db.query(Problem).all()

# 資料庫初始化
DATABASE_URL = "postgresql://postgres:Oo77441166@localhost:5432/project_webai"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


#多對多關聯表
teacher_class_table = Table(
    "teacher_class",
    Base.metadata,
    Column("teacher_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("class_id", Integer, ForeignKey("classes.id"), primary_key=True)
)


class Class(Base):
    __tablename__ = "classes"
    id = Column(Integer, primary_key=True)
    name = Column(String)

class ProblemBrief(BaseModel):
    id: int
    title: str
    description: str | None = None
    hint: str | None = None

    class Config:
        from_attributes  = True
# 資料表
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    conversations = relationship("Conversation", back_populates="user")
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=True)
    class_name = Column(String, nullable=True)
    real_name = Column(String, nullable=True)
    student_grade = Column(Integer, nullable=True)  # 年級
    student_id = Column(Integer, unique=True, nullable=True)  # 學號
    identity = Column(Integer, nullable=True)
    teaching_classes = relationship(
        "Class",
        secondary=teacher_class_table,
        backref="teachers"
    )



# JWT 與加密
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        username = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]).get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token decode error")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/student_history")
def get_submission_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submissions = db.query(CodeSubmission).filter(CodeSubmission.user_id == user.id).all()
    return [
        {
            "problem_id": s.problem_id,
            "problem_title": s.problem.title if s.problem else None,
            "code": s.code,
            "ai_score": s.ai_score,
            "ai_feedback": s.ai_feedback,
            "overall_correct": s.overall_correct,
            "detailed_results": (
                json.loads(s.detailed_results) if isinstance(s.detailed_results, str) else s.detailed_results
            ),
            "created_at": s.created_at,
        }
        for s in submissions
    ]


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")


class Problem(Base):
    __tablename__ = "problems"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(Text)
    sample_solution = Column(Text, nullable=True)
    solution_file_content = Column(LargeBinary, nullable=True)
    pdf_file_content = Column(LargeBinary, nullable=True)
    hint = Column(String, nullable=True)
    answer_output = Column(Text, nullable=True)
    test_input = Column(Text, nullable=True)  # ✅ 新增欄位：測資輸入
    created_at = Column(DateTime, default=datetime.utcnow)
    
class CourseMemberResponse(BaseModel):
    id: int
    name: Optional[str] = None
    account: Optional[str] = None
    student_id: Optional[int] = None  # 新增學號
    student_grade: Optional[int] = None  # 新增年級
    className: Optional[str] = None

    class Config:
        from_attributes  = True

class MemberCreate(BaseModel):
    """定義新增成員時前端傳來的資料格式"""
    name: str
    student_id: Optional[int] = None
    student_grade: Optional[int] = None
    className: Optional[str] = None
    account: str
    # 密碼由後端自動生成，狀態預設啟用

class MemberUpdate(BaseModel):
    """定義更新成員時前端傳來的資料格式"""
    name: Optional[str] = None
    student_id: Optional[int] = None
    student_grade: Optional[int] = None
    className: Optional[str] = None
    # 更新時不應修改密碼

Base.metadata.create_all(bind=engine)

# Pydantic 模型
class RegisterForm(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str

class NewConversationRequest(BaseModel):
    title: str

class RenameRequest(BaseModel):
    title: str
from pydantic import BaseModel

class GradeRequest(BaseModel):
    code: str


# 工具函式
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(pwd): return pwd_context.hash(pwd)
def verify_password(p, h): return pwd_context.verify(p, h)
def create_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def _find_compiler(candidates: list[str]) -> str | None:
    for name in candidates:
        p = shutil.which(name)
        if p:
            return p
    return None

def select_toolchain_by_ext(filename: str):
    """
    回傳 (ext, compile_cmd, run_cmd_builder, write_ext)
    - Python：不需要編譯
    - C / C++：自動偵測 clang++ / g++，若都沒有，回報清楚錯誤
    """
    ext = Path(filename).suffix.lower() if filename else ".py"

    if ext == ".c":
        compiler = _find_compiler(["gcc", "clang"])
        if not compiler:
            raise HTTPException(
                status_code=503,
                detail="找不到 C 編譯器（gcc/clang）。請安裝 MinGW-w64 或 LLVM 並加入 PATH。"
            )
        def run_cmd_builder(exe_path, src_path): return [str(exe_path)]
        return ".c", lambda src, out: [compiler, "-O2", "-std=c11", str(src), "-o", str(out)], run_cmd_builder, ".c"

    if ext in [".cc", ".cpp", ".cxx"]:
        compiler = _find_compiler(["clang++", "g++"])  # 先試 clang++，再 g++
        if not compiler:
            raise HTTPException(
                status_code=503,
                detail="找不到 C++ 編譯器（clang++/g++）。請安裝 LLVM 或 MinGW-w64 並加入 PATH。"
            )
        def run_cmd_builder(exe_path, src_path): return [str(exe_path)]
        return ".cpp", lambda src, out: [compiler, "-O2", "-std=c++17", str(src), "-o", str(out)], run_cmd_builder, ".cpp"

    # 預設 Python
    def run_cmd_builder(exe_path, src_path): return ["python", str(src_path)]
    return ".py", None, run_cmd_builder, ".py"
# ========== LLM 評分工具：組 prompt 與抽分數 ==========
import re
import json

def build_llm_prompt(problem, filename: str, student_code: str,
                     expected_output: str, program_output: str,
                     detailed_results: list[dict]) -> str:
    """
    用你的原始 prompt 為基礎，同時把逐筆測資比對結果也傳給模型（更穩定）。
    """
    # 把 detailed_results 轉成更好讀的文字塊（每筆一段）
    cases_text_lines = []
    for i, r in enumerate(detailed_results, start=1):
        ok = "是" if r.get("correct") else "否"
        cases_text_lines.append(
            f"- 測資 #{i}｜正確：{ok}\n"
            f"  [輸入]\n{r.get('input','')}\n"
            f"  [輸出]\n{r.get('output','')}\n"
            f"  [期望]\n{r.get('expected','')}\n"
        )
    cases_text = "\n".join(cases_text_lines) if cases_text_lines else "（無逐筆比對資料）"

    prompt = f"""
[題目資訊]
標題：{problem.title}
提示：{problem.hint or '無提示'}
敘述：
{problem.description or ''}

標準答案（若有）：
{problem.sample_solution or '無'}

[測資執行摘要]
總測資數量：{len(detailed_results)}
每個測資預期輸出（逐行匯總）：
{expected_output}

[實際執行輸出（逐行匯總）]
{program_output}

[逐筆測資比對結果]
{cases_text}

[學生檔案與程式碼]
檔名：{filename}
{student_code}

請閱讀學生程式與測資結果，依據正確性與程式品質（命名、可讀性、邏輯性）提供回饋，並以以下格式回覆（勿新增其他區塊標題）：
【評分】（分數 1～100）
【總結】
【優點】
【缺點】
【建議】

請用繁體中文，且務必包含「【評分】」這一行，分數使用整數或一位小數皆可。
"""
    return prompt.strip()


def extract_score_from_feedback(text: str) -> float:
    """
    從 LLM 回覆中抓出「【評分】..」的數字，並截斷到 0~100 區間。
    """
    m = re.search(r"[【\[]評分[\]】]?\s*[:：]?\s*(\d+(?:\.\d+)?)", text)
    score = float(m.group(1)) if m else 0.0
    # 安全界線
    if score < 0:
        score = 0.0
    if score > 100:
        score = 100.0
    return score
# =====================================================


# ---------- 解析測資與答案：一定要先定義 ----------
def parse_cases(test_input_text: str) -> list[str]:
    """
    將資料庫中的 test_input（多行）切成每個 case 一個字串，
    並保留換行當作 stdin（避免 input() 卡住）。
    """
    text = (test_input_text or "").strip()
    if not text:
        return []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # 每個 case 都補上換行，讓程式用 input() 能讀到
    return [ln + "\n" for ln in lines]

def parse_expected(ans_text: str) -> list[str]:
    """
    將 answer_output 切成每個 case 的預期輸出（逐行），移除前後空白。
    """
    txt = (ans_text or "").strip()
    if not txt:
        return []
    return [ln.strip() for ln in txt.replace("\r\n", "\n").replace("\r", "\n").split("\n")]



# 使用者註冊/登入
@app.post("/register")
def register(form: RegisterForm, db: Session = Depends(get_db)):
    if db.query(User).filter_by(username=form.username).first():
        raise HTTPException(status_code=400, detail="User already exists")
    user = User(username=form.username, hashed_password=get_password_hash(form.password))
    db.add(user)
    db.commit()
    return {"message": "註冊成功"}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="帳號或密碼錯誤")
    token = create_token({"sub": user.username})
    return {
    "access_token": token,
    "token_type": "bearer",
    "identity": user.identity  # ⭐ 加這行
}

class GradeRequest(BaseModel):
    code: str
    filename: Optional[str] = None   # e.g. "main.py" / "main.c" / "main.cpp"

class DetailedResult(BaseModel):
    input: str
    output: str
    expected: str
    correct: bool
class CombinedGradeResult(BaseModel):
    ai_score: float
    ai_feedback: str
    overall_correct: bool
    detailed_results: list[DetailedResult]
    program_output: str        # ← 這要加
    expected_output: str       # ← 這也要加


from sqlalchemy import Boolean, Float, JSON, Text, DateTime

class CodeSubmission(Base):
    __tablename__ = "code_submissions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    problem_id = Column(Integer, ForeignKey("problems.id"))       # 有時候也要知道是哪一題
    code = Column(Text, nullable=False)
    ai_score = Column(Float, nullable=False)
    ai_feedback = Column(Text, nullable=False)
    overall_correct = Column(Boolean, nullable=False)
    detailed_results = Column(JSON, nullable=False)              # 需 import JSON
    program_output = Column(Text, nullable=True)
    expected_output = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")
    problem = relationship("Problem")


Base.metadata.create_all(bind=engine)

from fastapi import HTTPException, Depends, Query, Body
from sqlalchemy.orm import Session
import subprocess, requests, tempfile, os, sys, re
from collections import Counter

# ================== LLM 呼叫工具：Ollama / fallback ==================
def call_codellama(prompt: str, temperature: float = 0.1, model: str = "codellama:13b") -> str:
    """
    嘗試呼叫本機 Ollama；失敗就回傳一段格式正確的 fallback。
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
    except Exception:
        # 保證格式正確的離線回覆，避免前端報「格式錯誤」
        return (
            "【評分】8.0\n"
            "【總結】（離線模式）模型不可用，提供範例格式。\n"
            "【優點】\n- 有基本結構\n- 易於閱讀\n"
            "【缺點】\n- 未實際分析程式\n- 分數僅示意\n"
            "【建議】\n- 確認 Ollama 埠號\n- 檢查模型已載入\n"
        )

def call_codellama_multiple(prompt: str, n: int = 3) -> list[str]:
    return [call_codellama(prompt) for _ in range(n)]

def select_most_common_feedback(responses: list[str]) -> str:
    """
    先挑包含 5 個段落標題的回覆；若都沒有，挑最長的一份。
    """
    need = ("【評分】", "【總結】", "【優點】", "【缺點】", "【建議】")
    candidates = [s for s in responses if isinstance(s, str)]
    for s in candidates:
        if all(k in s for k in need):
            return s
    return max(candidates, key=len) if candidates else ""


GEMINI_API_KEY = "AIzaSyAcEfnUVeBechvocsLchBccPSmYpCEwptY"

def _gemini_generate(url: str, prompt: str, temperature: float) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature}
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        if not text:
            raise RuntimeError("empty text in candidates[0]")
        return text.strip()
    except Exception:
        raise RuntimeError(f"unexpected response: {data}")

def call_gemini(prompt: str, model: str = "gemini-2.5-flash", temperature: float = 0.1) -> str:
    """
    使用 Google Gemini 模型（預設 gemini-2.5-flash）。
    若失敗則嘗試 v1beta，若都失敗會回傳具體錯誤。
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
        raise RuntimeError("GEMINI_API_KEY is empty（請確認 main.py 已填入真實金鑰）")

    versions = ["v1", "v1beta"]
    last_err = None

    for ver in versions:
        url = f"https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent?key={GEMINI_API_KEY}"
        try:
            return _gemini_generate(url, prompt, temperature)
        except requests.HTTPError as he:
            if he.response is not None and he.response.status_code == 404:
                last_err = he
                continue  # 試下一個版本
            last_err = he
            raise  # 其他 HTTP 錯誤直接丟出
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Gemini call failed for model={model}: {last_err}")
# ===================================================================

def call_llm_multiple(prompt: str, provider: str = "codellama", n: int = 3) -> list[str]:
    if provider == "gemini":
        return [call_gemini(prompt) for _ in range(n)]
    return [call_codellama(prompt) for _ in range(n)]
# =====================================


@app.post("/combined_grade", response_model=CombinedGradeResult)
def combined_grade(
    problem_id: int = Query(...),
    data: GradeRequest = Body(...),
    model: str = Query("codellama", pattern="^(codellama|gemini)$"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    """
    單檔批改：給練習或舊前端用。
    注意：前端現在兩種模式都已改走 /batch_grade；這支保留但要是『單檔自給自足』，不能再夾雜 batch 的變數。
    """
    # 1) 取題目
    problem = db.query(Problem).filter(Problem.id == problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="找不到該題目")

    # 2) 切測資/答案
    def _nl(s: str) -> str:
        return (s or "").replace("\r\n", "\n").replace("\r", "\n")

    expected_output_text = _nl(problem.answer_output or "")
    expected_lines = [ln.strip() for ln in expected_output_text.split("\n")] if expected_output_text else []

    test_input_text = _nl(problem.test_input or "")
    ti_lines = [(ln + "\n") for ln in test_input_text.split("\n")] if test_input_text else []

    num_cases = min(len(ti_lines), len(expected_lines)) if expected_lines else len(ti_lines)
    ti_lines = ti_lines[:num_cases]
    expected_lines = expected_lines[:num_cases]

    # 3) 依副檔名選工具鏈
    filename = data.filename or "main.py"
    ext, compile_cmd_factory, run_cmd_builder, write_ext = select_toolchain_by_ext(filename)

    # 4) 寫檔、(可選)編譯、逐測資執行
    detailed_results = []
    program_output_lines = []
    overall_correct = True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src_path = tmpdir / f"Main{write_ext}"
        exe_path = tmpdir / ("a.exe" if platform.system().lower().startswith("win") else "a.out")

        # 寫入原始碼
        src_path.write_text(data.code, encoding="utf-8")

        # 需要編譯（C/C++）
        if compile_cmd_factory is not None:
            try:
                cp = subprocess.run(
                    compile_cmd_factory(src_path, exe_path),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=30
                )
            except FileNotFoundError:
                raise HTTPException(status_code=503, detail="找不到編譯器（請安裝並加入 PATH）")
            if cp.returncode != 0:
                compile_err = (cp.stderr or cp.stdout or "").strip()
                # 編譯失敗時也回應清楚資訊
                ai_feedback = f"【評分】0\n【總結】編譯失敗\n【優點】\n【缺點】\n- {compile_err}\n【建議】\n- 請先修正編譯錯誤後再提交"
                ai_score = 0.0
                # 存一次 DB（可選）
                submission = CodeSubmission(
                    user_id=user.id,
                    problem_id=problem_id,
                    code=data.code,
                    ai_score=ai_score,
                    ai_feedback=ai_feedback,
                    overall_correct=False,
                    detailed_results=[],
                    program_output=compile_err,
                    expected_output="\n".join(expected_lines),
                    created_at=datetime.utcnow()
                )
                db.add(submission)
                db.commit()
                return CombinedGradeResult(
                    ai_score=ai_score,
                    ai_feedback=ai_feedback,
                    overall_correct=False,
                    detailed_results=[],
                    program_output=compile_err,
                    expected_output="\n".join(expected_lines)
                )

        # 組執行命令（Python 用 python 跑；C/C++ 用 exe）
        run_cmd = run_cmd_builder(exe_path, src_path)

        # 逐測資執行
        for i in range(num_cases):
            case_input = ti_lines[i] if i < len(ti_lines) else ""
            expected = expected_lines[i] if i < len(expected_lines) else ""
            try:
                run_res = subprocess.run(
                    run_cmd,
                    input=case_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                stdout = (run_res.stdout or "").strip()
                stderr = (run_res.stderr or "").strip()
                ok = (stdout.strip() == expected.strip()) if expected_lines else True
            except subprocess.TimeoutExpired:
                stdout, stderr, ok = "", "Time Limit Exceeded", False

            program_output_lines.append(stdout if not stderr else f"{stdout}\n{stderr}".strip())
            detailed_results.append({
                "input": case_input.rstrip("\n"),
                "output": stdout,
                "expected": expected,
                "correct": ok
            })
            if not ok:
                overall_correct = False

    program_output = "\n".join(program_output_lines)
    expected_output = "\n".join(expected_lines)

    # 5) 組 prompt → 呼叫 LLM → 擷取分數
    prompt = build_llm_prompt(
        problem=problem,
        filename=filename,
        student_code=data.code,
        expected_output=expected_output,
        program_output=program_output,
        detailed_results=detailed_results
    )
    ai_feedback = "⚠️ 無法取得模型回覆。"
    ai_score = 0.0
    try:
        responses = call_llm_multiple(prompt, provider=model, n=3)
        ai_feedback = select_most_common_feedback(responses) or ai_feedback
        ai_score = extract_score_from_feedback(ai_feedback)

    except Exception as e:
        ai_feedback = (ai_feedback + f"\n（LLM 調用失敗：{e}）").strip()

    # 6) 存 DB（單檔一次）
    submission = CodeSubmission(
        user_id=user.id,
        problem_id=problem_id,
        code=data.code,
        ai_score=ai_score,
        ai_feedback=ai_feedback,
        overall_correct=overall_correct,
        detailed_results=detailed_results,
        program_output=program_output,
        expected_output=expected_output,
        created_at=datetime.utcnow()
    )
    db.add(submission)
    db.commit()

    # 7) 回傳
    return CombinedGradeResult(
        ai_score=ai_score,
        ai_feedback=ai_feedback,
        overall_correct=overall_correct,
        detailed_results=detailed_results,
        program_output=program_output,
        expected_output=expected_output
    )



# 假設這是 FastAPI 路由的例子
@app.get("/problems")
def get_all_problems(db: Session = Depends(get_db)):
    problems = db.query(Problem).all()
    return [{"id": problem.id, "title": problem.title} for problem in problems]

@app.post("/upload_problem")
def upload_problem(
    title: str = Form(...),
    description: str = Form(...),
    sample_solution: Optional[str] = Form(None),
    solution_file: Optional[UploadFile] = File(None),
    pdf_file: Optional[UploadFile] = File(None),
    test_input: Optional[str] = Form(None),
    answer_output: Optional[str] = Form(None),
    hint: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 讀取上傳的程式檔案（如果有）
    solution_content_bytes = None
    if solution_file:
        try:
            solution_content_bytes = solution_file.file.read()
        except Exception:
            raise HTTPException(status_code=400, detail="程式檔案讀取失敗")

    # 讀取上傳的 PDF 題目檔案（如果有）
    pdf_content_bytes = None
    if pdf_file:
        try:
            pdf_content_bytes = pdf_file.file.read()
        except Exception:
            raise HTTPException(status_code=400, detail="PDF 檔案讀取失敗")

    # 寫入資料庫
    problem = Problem(
        title=title,
        description=description,
        sample_solution=sample_solution,
        solution_file_content=solution_content_bytes,
        pdf_file_content=pdf_content_bytes,
        hint=hint,
        answer_output=answer_output,
        test_input=test_input  # ✅ 存入資料庫
    )
    db.add(problem)
    db.commit()
    return {"message": "題目已成功上傳並儲存至資料庫"}

# 取得所有作業發布歷史紀錄
@app.get("/assignment_history")
def assignment_history(
    page: int = Query(1, ge=1),  # 頁碼，預設為第 1 頁
    limit: int = Query(10, ge=1),  # 每頁顯示的項目數，預設為 10
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 計算偏移量
    offset = (page - 1) * limit

    # 查詢資料庫，按 created_at 排序，並進行分頁
    problems = db.query(Problem).order_by(Problem.created_at.desc()).offset(offset).limit(limit).all()

    # 總數量，用於判斷是否有下一頁
    total_count = db.query(Problem).count()

    # 構建回應
    return {
        "items": [
            {
                "id": p.id,
                "title": p.title,
                "created_at": p.created_at.strftime("%Y-%m-%d %H:%M:%S")  # 格式化時間
            }
            for p in problems
        ],
        "hasNextPage": offset + limit < total_count  # 是否有下一頁
    }

@app.get("/assignment_detail/{problem_id}")
def assignment_detail(problem_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    problem = db.query(Problem).filter_by(id=problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="找不到該作業")
    return {
        "id": problem.id,
        "title": problem.title,
        "description": problem.description,
        "sample_solution": problem.sample_solution,
        "hint": problem.hint,
        "test_input": problem.test_input,  # 新增測資輸入
        "answer_output": problem.answer_output,  # 新增答案輸出
        "created_at": problem.created_at.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/submission_records/{problem_id}")
def submission_records(problem_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submissions = db.query(CodeSubmission).filter_by(problem_id=problem_id).all()
    # 不要 raise 404，直接回傳空陣列
    return [
        {
            "id": s.id,
            "real_name": s.user.real_name if s.user else None,
            "student_id": s.user.student_id if s.user else None,
            "ai_score": s.ai_score,
            "ai_feedback": s.ai_feedback,
            "program_output": s.program_output,
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for s in submissions
    ]

@app.get("/submission_overview/{problem_id}")
def submission_overview(problem_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from sqlalchemy import func
    # 查每個 user_id 的最高分與最後提交時間（不查 student_name）
    results = (
        db.query(
            CodeSubmission.user_id,
            func.max(CodeSubmission.ai_score).label("max_score"),
            func.max(CodeSubmission.created_at).label("last_submit")
        )
        .filter(CodeSubmission.problem_id == problem_id)
        .group_by(CodeSubmission.user_id)
        .all()
    )

    overview = []
    for r in results:
        # 以 user_id 去查 User.real_name（若 user_id 為 None 或查不到，顯示為 None 或 "(未知)"）
        if r.user_id is not None:
            u = db.query(User).filter(User.id == r.user_id).first()
            real_name = u.real_name if u else "(未知)"
            student_id = u.student_id if u else None
        else:
            real_name = None
            student_id = None

        overview.append({
            "user_id": r.user_id,
            "real_name": real_name,
            "student_id": student_id,
            "max_score": r.max_score,
            "last_submit": r.last_submit.strftime("%Y-%m-%d %H:%M:%S") if r.last_submit else None
        })
    return overview

@app.get("/submission_records/{problem_id}/{user_id}")
def submission_records_by_user(
    problem_id: int,
    user_id: Optional[str],  # 可為 None
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    user_id:
      - 數字字串：查詢該 user_id
      - "null"/"none"/""/None：查詢 user_id 為 None 的紀錄
    """
    uid_raw = (user_id or "").strip().lower()
    if uid_raw in ("null", "none", ""):
        submissions = db.query(CodeSubmission).filter_by(problem_id=problem_id, user_id=None).all()
    else:
        try:
            uid = int(uid_raw)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid user_id: {user_id}")
        submissions = db.query(CodeSubmission).filter_by(problem_id=problem_id, user_id=uid).all()

    return [
        {
            "id": s.id,
            "ai_score": s.ai_score,
            "ai_feedback": s.ai_feedback,
            "program_output": s.program_output,
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for s in submissions
    ]

def is_teacher_or_assistant(user: User = Depends(get_current_user)):
    """
    簡易判斷使用者是否為教師或助教。
    目前使用 User 模型中的 class_name 進行示意性判斷，請務必修改。
    """
    # 如果 User 模型有 role 欄位，可以這樣寫：
    # if user.role not in ['teacher', 'assistant']:
    #     raise HTTPException(status_code=403, detail="只有教師或助教有權限執行此操作")

    # 這裡是一個使用 class_name 的【示意】判斷，可能不適用您的實際情況
    # 假設 class_name 為 None 或 '學生' 代表不是老師或助教
    #if user.class_name is None or user.class_name == '學生':
    #     raise HTTPException(status_code=403, detail="只有教師或助教有權限執行此操作")

    # 如果判斷通過，函數沒有回傳 HTTPException，請求會繼續處理
    pass


@app.get("/api/course_members", response_model=List[CourseMemberResponse])
def get_course_members(
    query: Optional[str] = Query(None, description="依姓名或學號搜尋"),
    className: Optional[str] = Query(None, description="依課程篩選"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    # 只查學生
    query_stmt = db.query(User).filter(User.identity == 1)
    # 如果是老師，過濾自己所屬班級
    if user.identity == 0:
        teacher_class_ids = [str(c.id) for c in user.teaching_classes]
        if teacher_class_ids:
            query_stmt = query_stmt.filter(User.class_id.in_(teacher_class_ids))
        else:
            query_stmt = query_stmt.filter(False) # 沒有班級則不顯示任何學生

    if query:
        search_filter = or_(
            User.real_name.ilike(f"%{query}%"),
            cast(User.student_id, String).ilike(f"%{query}%")
        )
        query_stmt = query_stmt.filter(search_filter)
    if className:
        query_stmt = query_stmt.filter(User.class_name == className)
    members = query_stmt.all()
    return [
        CourseMemberResponse(
            id=member.id,
            name=member.real_name,
            student_id=str(member.student_id).zfill(8) if member.student_id is not None else None,
            student_grade=member.student_grade,
            className=member.class_name
        )
        for member in members
    ]

@app.post("/api/course_members", response_model=CourseMemberResponse)
def create_course_member(
    member_data: MemberCreate,
    teacher_check: None = Depends(is_teacher_or_assistant), # 依賴檢查
    db: Session = Depends(get_db)
):

    # 檢查使用者名稱/學號是否已存在
    db_user = db.query(User).filter(User.username == member_data.account).first()
    if db_user:
        raise HTTPException(status_code=400, detail="使用者帳號/學號已存在")

    # 自動生成密碼（這裡使用簡單方式，實際應用應更安全）
    # 例如：生成隨機密碼並透過郵件發送給使用者
    generated_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(12))
    hashed_password = get_password_hash(generated_password)

    # 創建 User 物件，只使用 User 模型中存在的欄位
    # 新增成員
    new_member = User(
        username=member_data.account,
        hashed_password=hashed_password,
        real_name=member_data.name,
        student_id=member_data.student_id,  # 新增學號
        student_grade=member_data.student_grade,  # 新增年級
        class_name=member_data.className
    )

    db.add(new_member)
    db.commit()
    db.refresh(new_member)

    # 回傳創建的成員資訊 (缺少欄位仍為 None)
    return CourseMemberResponse(
        id=new_member.id,
        name=new_member.real_name,
        account=new_member.username,
        email=None, # 資料庫中無此欄位
        role=None,  # 資料庫中無此欄位
        status=None,# 資料庫中無此欄位
        join_date=None # 資料庫中無此欄位
    )

@app.put("/api/course_members/{member_id}", response_model=CourseMemberResponse)
def update_course_member(
    member_id: int,
    member_data: MemberUpdate,
    teacher_check: None = Depends(is_teacher_or_assistant), # 依賴檢查
    db: Session = Depends(get_db)
):
    """
    更新課程成員資訊。
    注意：前端傳來 email, role, status，但目前 User 模型無此欄位，這些欄位的更新將無效。
    """
    db_member = db.query(User).filter(User.id == member_id).first()
    if not db_member:
        raise HTTPException(status_code=404, detail="找不到該成員")

    # 更新 User 模型中存在的欄位
    if member_data.name is not None:
        db_member.real_name = member_data.name
    if member_data.student_id is not None:
        db_member.student_id = member_data.student_id
    if member_data.student_grade is not None:
        db_member.student_grade = member_data.student_grade

    db.commit()
    db.refresh(db_member)

    # 回傳更新後的成員資訊 (缺少欄位仍為 None)
    return CourseMemberResponse(
        id=db_member.id,
        name=db_member.real_name,
        account=db_member.username,
        email=None, # 資料庫中無此欄位
        role=None,  # 資料庫中無此欄位
        status=None,# 資料庫中無此欄位
        join_date=None # 資料庫中無此欄位
    )

@app.delete("/api/course_members/{member_id}")
def delete_course_member(
    member_id: int,
    teacher_check: None = Depends(is_teacher_or_assistant), # 依賴檢查
    db: Session = Depends(get_db)
):
    """
    刪除課程成員。
    請確保刪除操作符合您的業務邏輯，例如是否級聯刪除相關數據 (如提交紀錄)。
    """
    db_member = db.query(User).filter(User.id == member_id).first()
    if not db_member:
        raise HTTPException(status_code=404, detail="找不到該成員")

    # 執行刪除
    db.delete(db_member)
    db.commit()

    return {"message": "成員已成功刪除"}

@app.post("/api/course_members/import")
async def import_course_members(
    file: UploadFile = File(...),
    teacher_check: None = Depends(is_teacher_or_assistant), # 依賴檢查
    db: Session = Depends(get_db)
):
    """
    從 CSV 檔案匯入課程成員。
    此處提供基礎 CSV 處理框架，需要根據您的檔案格式和需求修改。
    由於 User 模型缺少欄位，匯入的 email, role, status, join_date 將不會被儲存。
    """
    # 檢查檔案類型（這裡僅處理 CSV，如果需要 Excel 需安裝 pandas/openpyxl）
    if not file.filename.lower().endswith('.csv'):
        # 如果前端也允許 .xlsx，需要在這裡處理
        raise HTTPException(status_code=400, detail="目前僅支援匯入 CSV 檔案")

    content = await file.read()
    # 假設檔案是 UTF-8 編碼，如果不是，請嘗試其他編碼如 'big5' 或讓使用者選擇
    try:
        sio = StringIO(content.decode('utf-8'))
    except UnicodeDecodeError:
        try:
            sio = StringIO(content.decode('big5')) # 嘗試繁體中文常見編碼
        except Exception:
             raise HTTPException(status_code=400, detail="檔案編碼錯誤，請使用 UTF-8 或 Big5 編碼")

    reader = csv.reader(sio)
    try:
        header = next(reader) # 讀取標頭
    except StopIteration:
         raise HTTPException(status_code=400, detail="CSV 檔案為空")

    # 期望的標頭欄位 (可以根據您的需求調整)
    # 即使資料庫沒有這些欄位，可以先定義來讀取 CSV
    expected_header_columns = ['姓名', '帳號', '電子郵件', '角色'] # 定義讀取順序
    # 您可以加入更嚴謹的標頭檢查邏輯

    imported_count = 0
    skipped_count = 0
    skipped_details = []

    for row in reader:
        # 確保行資料至少包含期望的欄位數量
        if len(row) < len(expected_header_columns):
            skipped_count += 1
            skipped_details.append(f"行資料欄位不足 ({len(row)} < {len(expected_header_columns)}): {row}")
            continue

        # 根據 expected_header_columns 的順序提取資料
        member_name = row[0].strip()
        member_account = row[1].strip()
        member_email = row[2].strip() # ❗️ 無法儲存 email 到目前 User 模型
        member_role = row[3].strip()  # ❗️ 無法儲存 role 到目前 User 模型

        if not member_account:
            skipped_count += 1
            skipped_details.append(f"帳號為空，跳過該行: {row}")
            continue
         # 姓名非必需，但建議有
        if not member_name:
             member_name = member_account # 如果姓名為空，暫時用帳號代替

        # 檢查使用者帳號/學號是否已存在
        db_user = db.query(User).filter(User.username == member_account).first()
        if db_user:
            skipped_count += 1
            skipped_details.append(f"帳號已存在，跳過: {member_account}")
            continue

        # 生成密碼（同新增成員邏輯）
        generated_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(12))
        hashed_password = get_password_hash(generated_password)

        # 創建 User 物件，只使用 User 模型中存在的欄位
        new_member = User(
            username=member_account,
            hashed_password=hashed_password,
            real_name=member_name,
            # ❗️ 以下欄位在目前的 User 模型中不存在，不會被儲存
            # email=member_email, # 需要新增 email 欄位
            # role=member_role,   # 需要新增 role 欄位
            # status='active',    # 需要新增 status 欄位
            # created_at=datetime.utcnow() # 需要新增 created_at 欄位
            class_id=None, # 請根據您的邏輯設定
            class_name=None # 請根據您的邏輯設定
        )
        db.add(new_member)
        imported_count += 1

    try:
        db.commit()
        return {
            "message": f"匯入完成。成功新增 {imported_count} 位成員，跳過 {skipped_count} 位成員。",
            "skipped_details": skipped_details
        }
    except Exception as e:
        db.rollback()
        print(f"匯入成員時資料庫錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"匯入成員失敗: {e}")

@app.get("/api/course_members/export")
def export_course_members(
    query: Optional[str] = Query(None),
    className: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    query_stmt = db.query(User).filter(User.identity == 1)
    # 如果是老師，過濾自己所屬班級
    if user.identity == 0:
        teacher_class_ids = [str(c.id) for c in user.teaching_classes]
        if teacher_class_ids:
            query_stmt = query_stmt.filter(User.class_id.in_(teacher_class_ids))
        else:
            query_stmt = query_stmt.filter(False)

    if query:
        search_filter = or_(
            User.real_name.ilike(f"%{query}%"),
            cast(User.student_id, String).ilike(f"%{query}%")
        )
        query_stmt = query_stmt.filter(search_filter)
    if className:
        query_stmt = query_stmt.filter(User.class_name == className)
    members = query_stmt.all()

    # 產生 CSV 內容
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['序號', '姓名', '學號', '年級', '課程'])
    for idx, m in enumerate(members, 1):
        writer.writerow([
            idx,
            m.real_name or '',
            m.student_id or '',
            m.student_grade or '',
            m.class_name or ''
        ])
    csv_content = output.getvalue()
    output.close()

    # 加上 BOM
    bom = '\ufeff'
    csv_bytes = (bom + csv_content).encode('utf-8')
    return StreamingResponse(
        BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=course_members.csv"}
    )


@app.get("/api/course_members/{member_id}")
def get_member(member_id: int, db: Session = Depends(get_db)):
    member = db.query(User).filter(User.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="成員不存在")
    return {
        "id": member.id,
        "name": member.real_name,
        "account": member.username,
        "student_id": member.student_id,
        "student_grade": member.student_grade,
        "className": member.class_name
    }

# 在您的 API 路由區塊中新增

@app.get("/api/class_names", response_model=List[str])
def get_unique_class_names(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    # 教師只回傳自己所屬班級
    if user.identity == 0:
        return [c.name for c in user.teaching_classes]
    # 其他身份（如管理員）回傳全部班級
    class_names = (
        db.query(User.class_name)
        .distinct()
        .filter(User.class_name.isnot(None))
        .filter(User.class_name != "")
        .order_by(User.class_name)
        .all()
    )
    return [name[0] for name in class_names]
#學生練習
@app.get("/progress")
def get_progress(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    problems = db.query(Problem).all()
    
    # 取得該使用者所有繳交記錄
    submissions = (
        db.query(CodeSubmission)
        .filter(CodeSubmission.user_id == user.id)
        .order_by(CodeSubmission.created_at.desc())
        .all()
    )
    
    # 先統計每題繳交次數，分類成功/失敗
    stats = {}  # {problem_id: {"passed": int, "failed": int}}
    latest_results = {}  # 紀錄每題最新繳交狀態

    for sub in submissions:
        pid = sub.problem_id
        if pid not in stats:
            stats[pid] = {"passed": 0, "failed": 0}
        if sub.overall_correct:
            stats[pid]["passed"] += 1
        else:
            stats[pid]["failed"] += 1

        # 紀錄最新狀態 (因為 submissions 是按時間排序)
        if pid not in latest_results:
            latest_results[pid] = sub.overall_correct

    progress = []
    for problem in problems:
        pid = problem.id
        if pid in latest_results:
            status = "✅" if latest_results[pid] else "❌"
            passed = stats[pid]["passed"]
            failed = stats[pid]["failed"]
        else:
            status = "⏳"
            passed = 0
            failed = 0
        
        progress.append({
            "problem_id": pid,
            "title": problem.title,
            "status": status,
            "passed_count": passed,
            "failed_count": failed,
        })
    return progress
@app.post("/practice_code", response_model=PracticeResult)
def practice_code(
    data: PracticeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    練習模式：讓學生測試程式碼，不會儲存到資料庫
    """
    import tempfile
    import os
    import sys
    import subprocess
    
    # 如果指定了題目ID，取得題目資訊
    problem = None
    if data.problem_id:
        problem = db.query(Problem).filter_by(id=data.problem_id).first()
        if not problem:
            raise HTTPException(status_code=404, detail="找不到指定的題目")
    
    # 建立臨時檔案執行程式碼
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(data.code)
        tmp_path = tmp_file.name
    
    try:
        # 準備輸入資料
        test_input = ""
        if problem and problem.test_input:
            test_input = problem.test_input.strip()
        
        # 執行程式碼
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=test_input + "\n" if test_input else "",
            capture_output=True,
            timeout=10,  # 練習模式給更長的執行時間
            text=True
        )
        
        execution_success = proc.returncode == 0
        program_output = proc.stdout.strip() if proc.stdout else ""
        error_message = proc.stderr.strip() if proc.stderr else None
        
        # 如果有指定題目，進行測試比對（只比對第一筆）
        test_results = None
        if problem and execution_success:
            test_results = []
            if problem and problem.answer_output:
                expected_lines = problem.answer_output.strip().splitlines()
                output_lines = program_output.splitlines()
                
                # 只比對第一行
                if expected_lines:
                    expected = expected_lines[0]
                    actual = output_lines[0] if len(output_lines) > 0 else "(無輸出)"
                    is_correct = actual.strip() == expected.strip()
                    
                    test_results.append({
                        "test_case": 1,
                        "expected": expected,
                        "actual": actual,
                        "correct": is_correct
                    })
        
        return PracticeResult(
            program_output=program_output,
            execution_success=execution_success,
            error_message=error_message,
            test_results=test_results
        )
        
    except subprocess.TimeoutExpired:
        return PracticeResult(
            program_output="",
            execution_success=False,
            error_message="程式執行超時（超過10秒）"
        )
    except Exception as e:
        return PracticeResult(
            program_output="",
            execution_success=False,
            error_message=f"執行錯誤: {str(e)}"
        )
    finally:
        # 清理臨時檔案
        try:
            os.remove(tmp_path)
        except:
            pass

@app.get("/practice_problems")
def get_practice_problems(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    取得可用於練習的題目列表
    """
    problems = db.query(Problem).all()
    return [
        {
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "hint": p.hint,
            "has_test_case": bool(p.test_input and p.answer_output)
        }
        for p in problems
    ]

@app.get("/practice_problem/{problem_id}")
def get_practice_problem_detail(
    problem_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    取得練習題目詳細資訊
    """
    problem = db.query(Problem).filter_by(id=problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="找不到題目")
    
    return {
        "id": problem.id,
        "title": problem.title,
        "description": problem.description,
        "hint": problem.hint,
        "sample_solution": problem.sample_solution,  # 可選：是否顯示範例解答
        "has_test_case": bool(problem.test_input and problem.answer_output)
    }
#學生練習停止線
from fastapi import Body
from typing import List
class SaveStudentScoreRequest(BaseModel):
    student_name: str
    problem_id: int
    score: float
    feedback: str

@app.post("/save_student_score")
def save_student_score(
    data: List[SaveStudentScoreRequest] = Body(...),
    db: Session = Depends(get_db)
):
    """
    由外部匯入分數時，嘗試以 student_name 對應 User.real_name 或 student_id，
    若找到對應 User，則用 user_id 建立 CodeSubmission；找不到則跳過並回傳 skipped 詳細。
    （避免使用或新增不存在的 student_name 欄位）
    """
    imported_count = 0
    skipped_count = 0
    skipped_details = []

    for item in data:
        found_user = None
        name = (item.student_name or "").strip()

        if name:
            # 先以 real_name 精確比對
            found_user = db.query(User).filter(User.real_name == name).first()

            # 若看起來像學號，嘗試以 student_id 比對
            if not found_user:
                try:
                    sid = int(name)
                    found_user = db.query(User).filter(User.student_id == sid).first()
                except Exception:
                    pass

        if not found_user:
            skipped_count += 1
            skipped_details.append(f"找不到使用者：{item.student_name}")
            continue

        submission = CodeSubmission(
            user_id=found_user.id,
            problem_id=item.problem_id,
            code="",
            ai_score=item.score,
            ai_feedback=item.feedback,
            overall_correct=False,
            detailed_results=[],
            program_output=None,
            expected_output=None,
            created_at=datetime.utcnow()
        )
        db.add(submission)
        imported_count += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"db_write_failed: {e}")

    return {
        "message": f"匯入完成。成功新增 {imported_count} 筆，跳過 {skipped_count} 筆。",
        "skipped_details": skipped_details
    }
@app.post("/batch_grade")
def batch_grade(
    req: BatchGradeRequest,
    model: str = Query("codellama", pattern="^(codellama|gemini)$"),
    current_user: User = Depends(get_current_user),
):
    import threading, uuid, json
    from datetime import datetime
    import platform, subprocess, tempfile
    from pathlib import Path

    user_id = current_user.id
    task_id = str(uuid.uuid4())

    # 初始化任務狀態
    batch_tasks[task_id] = {
        "status": "queued",
        "total": len(req.files),
        "finished": 0,
        "results": [],
        "error": None,
        "user_id": user_id,
        "model": model,  # ← 新增
    }

    def do_batch(task_id: str, user_id: int, req: BatchGradeRequest):
        db = SessionLocal()
        provider = batch_tasks[task_id].get("model", "codellama")
        try:
            batch_tasks[task_id]["status"] = "running"

            for idx, file in enumerate(req.files):
                problem = db.query(Problem).filter(Problem.id == file.problem_id).first()
                if not problem:
                    # 題目不存在：記一筆錯誤結果後繼續
                    batch_tasks[task_id]["results"].append({
                        "filename": getattr(file, "filename", f"file_{idx}"),
                        "program_output": "",
                        "expected_output": "",
                        "overall_correct": False,
                        "detailed_results": [],
                        "ai_score": 0.0,
                        "ai_feedback": "【總結】找不到題目",
                        "submission_id": None,
                        "user_id": user_id
                    })
                    batch_tasks[task_id]["finished"] += 1
                    continue

                # 依副檔名決定工具鏈
                ext, compile_cmd_factory, run_cmd_builder, write_ext = select_toolchain_by_ext(
                    getattr(file, "filename", None) or "main.py"
                )

                # 切測資
                test_input_text = (problem.test_input or '').replace("\r\n", "\n").replace("\r", "\n").strip()
                answer_text     = (problem.answer_output or '').replace("\r\n", "\n").replace("\r", "\n").strip()

                ti_lines = [ln + "\n" for ln in test_input_text.split("\n")] if test_input_text else []
                expected_lines = [ln.strip() for ln in answer_text.split("\n")] if answer_text else []

                num_cases = min(len(ti_lines), len(expected_lines)) if expected_lines else len(ti_lines)
                ti_lines = ti_lines[:num_cases]
                expected_lines = expected_lines[:num_cases]

                detailed_results = []
                program_output_lines = []
                overall_correct = True

                # 寫檔 / 編譯 / 執行
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    src_path = tmpdir / f"Main{write_ext}"
                    exe_path = tmpdir / ("a.exe" if platform.system().lower().startswith("win") else "a.out")

                    # 寫入原始碼
                    src_path.write_text(file.code, encoding="utf-8")

                    # 需要編譯（C/C++）
                    if compile_cmd_factory is not None:
                        try:
                            cp = subprocess.run(
                                compile_cmd_factory(src_path, exe_path),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=30
                            )
                        except FileNotFoundError:
                            compile_err = "找不到編譯器（請安裝並加入 PATH）"
                            batch_tasks[task_id]["results"].append({
                                "filename": file.filename,
                                "program_output": compile_err,
                                "expected_output": "\n".join(expected_lines),
                                "overall_correct": False,
                                "detailed_results": [],
                                "ai_score": 0.0,
                                "ai_feedback": f"【評分】0\n【總結】編譯失敗\n【缺點】{compile_err}\n【建議】請修正環境或 PATH",
                                "submission_id": None,
                                "user_id": user_id
                            })
                            batch_tasks[task_id]["finished"] += 1
                            continue

                        if cp.returncode != 0:
                            compile_err = cp.stderr.strip()
                            batch_tasks[task_id]["results"].append({
                                "filename": file.filename,
                                "program_output": compile_err,
                                "expected_output": "\n".join(expected_lines),
                                "overall_correct": False,
                                "detailed_results": [],
                                "ai_score": 0.0,
                                "ai_feedback": f"【評分】0\n【總結】編譯失敗\n【缺點】{compile_err}\n【建議】請修正編譯錯誤後再提交",
                                "submission_id": None,
                                "user_id": user_id
                            })
                            batch_tasks[task_id]["finished"] += 1
                            continue

                    # 組執行命令（Python 用 python 跑；C/C++ 用 exe）
                    run_cmd = run_cmd_builder(exe_path, src_path)

                    # 跑測資
                    for i in range(num_cases):
                        case_input = ti_lines[i] if i < len(ti_lines) else ""
                        expected = expected_lines[i] if i < len(expected_lines) else ""
                        try:
                            run_res = subprocess.run(
                                run_cmd,
                                input=case_input,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=5
                            )
                            stdout = run_res.stdout.strip()
                            stderr = run_res.stderr.strip()
                            ok = (stdout.strip() == expected.strip()) if expected_lines else True
                        except subprocess.TimeoutExpired:
                            stdout, stderr, ok = "", "Time Limit Exceeded", False

                        program_output_lines.append(stdout if not stderr else f"{stdout}\n{stderr}".strip())
                        detailed_results.append({
                            "input": case_input.rstrip("\n"),
                            "output": stdout,
                            "expected": expected,
                            "correct": ok
                        })
                        if not ok:
                            overall_correct = False

                    program_output = "\n".join(program_output_lines)
                    expected_output = "\n".join(expected_lines)

                    # ======（在 do_batch 的每檔處理內，計算完 program_output / expected_output / detailed_results 之後）======

                    # 1) 組 prompt（用你的風格 + 加入逐筆比對結果）
                    prompt = build_llm_prompt(
                        problem=problem,
                        filename=file.filename,
                        student_code=file.code,
                        expected_output=expected_output,
                        program_output=program_output,
                        detailed_results=detailed_results
                    )

                    # 2) 呼叫你的模型（沿用你原本的呼叫方法）
                    #    假設你已有這兩個工具函式：call_codellama_multiple、select_most_common_feedback
                    ai_feedback = "⚠️ 無法取得模型回覆。"
                    ai_score = 0.0
                    try:
                        responses = call_llm_multiple(prompt, provider=provider, n=3)
                        ai_feedback = select_most_common_feedback(responses) or ai_feedback
                        ai_score = extract_score_from_feedback(ai_feedback)

                    except Exception as e:
                        # 不讓整批失敗；保留錯誤訊息在回饋末尾
                        ai_feedback = (ai_feedback + f"\n（LLM 調用失敗：{e}）").strip()

                    # 3) 寫入資料庫（沿用你現有的 CodeSubmission 模型）
                    from datetime import datetime
                    saved_id = None
                    try:
                        submission = CodeSubmission(
                            user_id=user_id,
                            problem_id=file.problem_id,
                            code=file.code,
                            ai_score=ai_score,
                            ai_feedback=ai_feedback,
                            overall_correct=overall_correct,
                            detailed_results=detailed_results,     # 若此欄位是 JSON/text，SQLAlchemy 會自動轉（或你原本用 json.dumps）
                            program_output=program_output,
                            expected_output=expected_output,
                            created_at=datetime.utcnow()
                        )
                        db.add(submission)
                        db.commit()
                        db.refresh(submission)
                        saved_id = submission.id
                    except Exception as e:
                        db.rollback()
                        # 把 DB 錯誤附註到 ai_feedback，避免丟失資訊
                        ai_feedback = (ai_feedback + f"\n（DB 儲存失敗：{e}）").strip()

                    # 4) 回填到任務結果（前端要顯示）
                    batch_tasks[task_id]["results"].append({
                        "filename": file.filename,
                        "program_output": program_output,
                        "expected_output": expected_output,
                        "overall_correct": overall_correct,
                        "detailed_results": detailed_results,
                        "ai_score": ai_score,
                        "ai_feedback": ai_feedback,
                        "submission_id": saved_id,
                        "user_id": user_id
                    })
                    batch_tasks[task_id]["finished"] += 1
                    # ======（到此為止）======


            batch_tasks[task_id]["status"] = "done"
        except Exception as e:
            batch_tasks[task_id]["status"] = "error"
            batch_tasks[task_id]["error"] = str(e)
        finally:
            try:
                db.close()
            except:
                pass

    # 啟動背景執行
    threading.Thread(target=do_batch, args=(task_id, user_id, req), daemon=True).start()
    return {"task_id": task_id}




@app.get("/batch_grade/status/{task_id}")
def get_batch_grade_status(task_id: str):
    """
    根據 task_id 查詢批改任務的狀態與結果。
    """
    task = batch_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到該任務")
    return task
# 新增：查詢目前排隊人數與 index
@app.get("/queue/position")
def queue_position(task_id: str):
    try:
        idx = list(waiting_queue).index(task_id)
        total = len(waiting_queue)
        return {"position": idx + 1, "waiting": idx, "total": total}
    except ValueError:
        # 不在 queue 代表已完成或不存在
        return {"position": 0, "waiting": 0, "total": len(waiting_queue)}

@app.get("/queue/waiting_count")
def queue_waiting_count():
    """
    回傳目前等待隊列中的人數（任務數）。
    """
    return {"waiting_count": len(waiting_queue)}