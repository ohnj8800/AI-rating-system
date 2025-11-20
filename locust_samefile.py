
# locust_samefile.py
# 使用相同測試檔案內容，讓所有使用者提交到 /combined_grade
import json
import random
from pathlib import Path
from locust import HttpUser, task, between, events

# ====== 你可以改這些參數 ======
USERS = [f"test{str(i).zfill(2)}" for i in range(1, 21)]
PASSWORDS = {u: u for u in USERS}
PROBLEM_IDS = [7]  # 把它改成你實際的 problem_id 清單（可多個）
TEST_FILE_PATH = "2 the 9s.py"  # 所有人都交這份
# ==============================

# 預先讀取檔案內容（避免每次請求打開檔案 IO）
try:
    CODE_TEXT = Path(TEST_FILE_PATH).read_text(encoding="utf-8")
except Exception as e:
    CODE_TEXT = "# 讀檔失敗時的預設內容（請確認 TEST_FILE_PATH）\nprint('hello')"
    print(f"[WARN] 無法讀取測試檔案 {TEST_FILE_PATH}: {e}")

class StudentUser(HttpUser):
    wait_time = between(1, 3)
    token = None
    username = None

    def on_start(self):
        # 為每個虛擬使用者挑一個帳號（隨機）
        self.username = random.choice(USERS)
        password = PASSWORDS[self.username]
        with self.client.post(
            "/login",
            data={"username": self.username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
            else:
                resp.failure(f"login failed: {resp.status_code} {resp.text}")

    @task(3)
    def submit_same_file(self):
        if not self.token or not PROBLEM_IDS:
            return

        pid = random.choice(PROBLEM_IDS)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = json.dumps({"code": CODE_TEXT})

        with self.client.post(
            f"/combined_grade?problem_id={pid}",
            data=payload,
            headers=headers,
            name="/combined_grade (same file)",
            catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"grade failed: {resp.status_code} {resp.text}")
