from passlib.context import CryptContext
import sys

try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    h = pwd_context.hash("admin123")
    print(f"Hash: {h}")
    v = pwd_context.verify("admin123", h)
    print(f"Verify: {v}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
