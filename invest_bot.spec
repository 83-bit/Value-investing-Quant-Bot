# PyInstaller spec — pyinstaller invest_bot.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

icon_file = "assets/invest_bot.ico"

hidden = [
    "global_screener",
    "partial_rerun",
    "refresh_volume_attention",
    "backtest_attention",
    "audit_thesis",
    "invest_bot_launcher",
    "pandas",
    "numpy",
    "yfinance",
    "openai",
    "httpx",
    "httpcore",
    "certifi",
    "dotenv",
    "tenacity",
    "sqlite3",
    "concurrent.futures",
    "tkinter",
    "PIL",
]

a = Analysis(
    ["invest_bot_gui.py"],
    pathex=[],
    binaries=[],
    datas=[("assets/invest_bot.ico", "assets")],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "PyQt5", "PyQt6"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="InvestBot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)
