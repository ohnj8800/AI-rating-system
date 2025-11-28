from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

password = "Student"   # ← 重點：補零到兩位數
hashed = pwd_context.hash(password)
print(f"{password} -> {hashed}")
