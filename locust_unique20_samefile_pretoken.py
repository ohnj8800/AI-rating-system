
# locust_unique20_samefile_pretoken.py
import json
from pathlib import Path
from collections import deque
from locust import HttpUser, task, between, events
from locust.exception import StopUser
import gevent.lock
import requests
import random, gevent  # 檔案頂部已有就不用再加


BASE_URL = "http://localhost:8000"
PROBLEM_ID = 7
TEST_FILE_PATH = "2the9s.py"
USERS = [f"test{str(i).zfill(2)}" for i in range(1, 21)]
PASSWORDS = {u: u for u in USERS}

_accounts = deque(USERS)
_lock = gevent.lock.Semaphore()
_token_map = {}

@events.test_start.add_listener
def _prefetch_tokens(environment, **kwargs):
    session = requests.Session()
    for u in USERS:
        payload = {"username": u, "password": PASSWORDS[u]}
        try:
            r = session.post(f"{BASE_URL}/login",
                             data=payload,
                             headers={"Content-Type": "application/x-www-form-urlencoded"},
                             timeout=10)
            r.raise_for_status()
            _token_map[u] = r.json().get("access_token")
            if not _token_map[u]:
                print(f"[WARN] login ok but no token for {u}")
        except Exception as e:
            print(f"[ERROR] prefetch login failed for {u}: {e}")
    session.close()

try:
    CODE_TEXT = Path(TEST_FILE_PATH).read_text(encoding="utf-8")
except Exception as e:
    print(f"[WARN] 無法讀取檔案 {TEST_FILE_PATH}: {e}")
    CODE_TEXT = "print(input())"

class StudentUser(HttpUser):
    host = BASE_URL
    wait_time = between(0.5, 1.0)

    def on_start(self):
        with _lock:
            if not _accounts:
                raise StopUser("No account available")
            self.username = _accounts.popleft()

        self.token = _token_map.get(self.username)
        if not self.token:
            with self.client.post(
                "/login",
                data={"username": self.username, "password": self.username},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                name="/login (fallback)",
                catch_response=True,
            ) as resp:
                if resp.status_code == 200:
                    self.token = resp.json().get("access_token")
                else:
                    resp.failure(f"login failed: {resp.status_code} {resp.text}")
                    raise StopUser()

    @task
    def submit_once_and_quit(self):
        gevent.sleep(random.uniform(0, 1.5))  # ★ 起跑抖動，分散 0~1.5 秒
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
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
