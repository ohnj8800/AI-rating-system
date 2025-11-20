from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

for i in range(3, 21):  # 03 到 20
    password = f"test{i:02d}"   # ← 重點：補零到兩位數
    hashed = pwd_context.hash(password)
    print(f"{password} -> {hashed}")
