
"""
V16.03 Main CLV Patch Layer

Safe integration helper for V16.00 main.py.

Purpose:
- keep main.py unchanged
- expose CLV bridge through one function
- allow later merge into production
"""

from main_bridge_v16_02_1 import run_main_bridge


def attach_clv_to_main():
    return run_main_bridge()


if __name__ == "__main__":
    print("=== V16.03 MAIN CLV PATCH TEST ===")
    print(attach_clv_to_main())
