
# locust_unique20_samefile.py
import json
from pathlib import Path
from collections import deque
from locust import HttpUser, task, between
from locust.exception import StopUser
import gevent.lock

# ====== 請依你的環境修改 ======
BASE_URL = "http://localhost:8000"
PROBLEM_ID = 7              # 單題測試，改成你的 problem_id
TEST_FILE_PATH = "2the9s.py"  # 同一份測試檔案
# =============================

# 準備 20 組帳密（密碼=帳號）
_accounts = deque([f"test{str(i).zfill(2)}" for i in range(1, 21)])
_lock = gevent.lock.Semaphore()

# 預先讀檔，避免每次 I/O
try:
    CODE_TEXT = Path(TEST_FILE_PATH).read_text(encoding="utf-8")
except Exception as e:
    print(f"[WARN] 無法讀取測試檔案 {TEST_FILE_PATH}: {e}")
    CODE_TEXT = "print(input())"  # 後備內容

class StudentUser(HttpUser):
    host = BASE_URL
    wait_time = between(0.5, 1.0)  # 其實只會跑一次任務

    def on_start(self):
        # 取得唯一帳號
        with _lock:
            if not _accounts:
                raise StopUser("No account available")
            self.username = _accounts.popleft()
            self.password = self.username  # 密碼同帳號

        # 登入
        with self.client.post(
            "/login",
            data={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            name="/login",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                if not self.token:
                    resp.failure("missing access_token")
                    raise StopUser("login failed: no token")
            else:
                resp.failure(f"login failed: {resp.status_code} {resp.text}")
                raise StopUser("login failed")

    @task
    def submit_once_and_quit(self):
        # 提交相同檔案內容一次
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({"code": CODE_TEXT})

        with self.client.post(
            f"/combined_grade?problem_id={PROBLEM_ID}",
            data=payload,
            headers=headers,
            name="/combined_grade (unique, same file)",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"grade failed: {resp.status_code} {resp.text}")
            raise StopUser()
