"""
Invest Bot 視窗版（tkinter）。
長時間任務在背景執行緒跑，log 顯示於視窗內。
"""
from __future__ import annotations

import io
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk


def _resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


def _env_ok() -> tuple[bool, str]:
    from global_screener import DEEPSEEK_API_KEY, FMP_API_KEY

    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not FMP_API_KEY:
        missing.append("FMP_API_KEY")
    if missing:
        return False, "缺少 .env：" + ", ".join(missing)
    return True, "API 金鑰已載入"


class _QueueWriter(io.TextIOBase):
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:
        pass


class InvestBotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Invest Bot — Quant Screener")
        self.root.minsize(720, 520)
        self.root.geometry("860x620")

        icon = _resource_path(os.path.join("assets", "invest_bot.ico"))
        if os.path.isfile(icon):
            try:
                self.root.iconbitmap(icon)
            except Exception:
                pass

        self._log_q: queue.Queue = queue.Queue()
        self._running = False
        self._worker: threading.Thread | None = None

        self._build_ui()
        self._refresh_env_status()

    def _build_ui(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#1e293b")
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#64748b")
        style.configure("Action.TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"), padding=10)

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="Invest Bot", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Quantamental 選股 · Volume Attention · LLM 辯論 · 回測",
            style="Sub.TLabel",
        ).pack(anchor=tk.W)

        self.env_label = ttk.Label(header, text="", style="Sub.TLabel")
        self.env_label.pack(anchor=tk.W, pady=(6, 0))

        opts = ttk.LabelFrame(outer, text="選項", padding=10)
        opts.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(opts, text="模式").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self.mode_var = tk.StringVar(value="both")
        ttk.Combobox(
            opts,
            textvariable=self.mode_var,
            values=["both", "compounder", "growth"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(opts, text="Partial 上限 (0=全部)").grid(row=0, column=2, sticky=tk.W, padx=(24, 8))
        self.limit_var = tk.StringVar(value="0")
        ttk.Entry(opts, textvariable=self.limit_var, width=8).grid(row=0, column=3, sticky=tk.W)

        actions = ttk.LabelFrame(outer, text="功能", padding=10)
        actions.pack(fill=tk.X, pady=(0, 10))

        buttons = [
            ("完整掃描", self._on_scan, "scan"),
            ("重跑 LLM", self._on_partial, "partial"),
            ("重算 Volume", self._on_volume, "volume"),
            ("Tier 回測", self._on_backtest, "backtest"),
            ("Attention 回測", self._on_attention, "attention"),
        ]
        for i, (label, cmd, _key) in enumerate(buttons):
            ttk.Button(actions, text=label, style="Action.TButton", command=cmd).grid(
                row=i // 3, column=i % 3, padx=6, pady=6, sticky=tk.EW
            )
        for c in range(3):
            actions.columnconfigure(c, weight=1)

        util = ttk.Frame(outer)
        util.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(util, text="開啟工作資料夾", command=self._open_workdir).pack(side=tk.LEFT)
        ttk.Button(util, text="重新檢查 .env", command=self._refresh_env_status).pack(side=tk.LEFT, padx=8)

        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(0, 8))

        log_frame = ttk.LabelFrame(outer, text="執行紀錄", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
        )
        self.log.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(outer, text="就緒", style="Sub.TLabel")
        self.status.pack(anchor=tk.W, pady=(8, 0))

    def _append_log(self, text: str):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _refresh_env_status(self):
        ok, msg = _env_ok()
        color = "#15803d" if ok else "#b45309"
        self.env_label.configure(text=msg, foreground=color)

    def _open_workdir(self):
        from global_screener import APP_DIR

        os.startfile(APP_DIR)

    def _mode_list(self) -> list[str]:
        m = self.mode_var.get()
        return ["compounder", "growth"] if m == "both" else [m]

    def _limit_or_none(self) -> int | None:
        raw = self.limit_var.get().strip()
        if not raw or raw == "0":
            return None
        try:
            return max(1, int(raw))
        except ValueError:
            return None

    def _confirm_long(self, title: str, body: str) -> bool:
        return messagebox.askyesno(title, body, icon=messagebox.WARNING)

    def _start_task(self, name: str, target):
        if self._running:
            messagebox.showinfo("忙碌中", "請等待目前任務完成。")
            return

        self._running = True
        self.status.configure(text=f"執行中：{name}")
        self.progress.start(12)
        self._append_log(f"\n{'=' * 60}\n▶ {name}\n{'=' * 60}\n")

        def worker():
            writer = _QueueWriter(self._log_q)
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = writer
            err = None
            try:
                target()
            except Exception as e:
                err = e
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self._log_q.put(("__DONE__", err))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self.root.after(80, self._poll_log)

    def _poll_log(self):
        while True:
            try:
                item = self._log_q.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item[0] == "__DONE__":
                err = item[1]
                self.progress.stop()
                self._running = False
                if err:
                    self._append_log(f"\n❌ 錯誤：{err}\n")
                    messagebox.showerror("錯誤", str(err))
                else:
                    self._append_log("\n✅ 完成\n")
                self.status.configure(text="就緒")
                return
            self._append_log(str(item))

        if self._running:
            self.root.after(80, self._poll_log)

    def _on_scan(self):
        if not self._confirm_long(
            "完整掃描",
            "將掃描全球 universe（數百檔），可能耗時數小時並消耗 API 配額。\n確定要開始？",
        ):
            return

        def run():
            from global_screener import main

            main()

        self._start_task("完整掃描", run)

    def _on_partial(self):
        limit = self._limit_or_none()
        modes = self._mode_list()

        def run():
            from global_screener import partial_rerun_csv

            for m in modes:
                partial_rerun_csv(f"results_{m}.csv", limit=limit, regenerate_one_pagers=False)

        self._start_task("Partial LLM rerun", run)

    def _on_volume(self):
        modes = self._mode_list()
        limit = self._limit_or_none()

        def run():
            from refresh_volume_attention import refresh_csv

            for m in modes:
                refresh_csv(f"results_{m}.csv", limit=limit, fmp_first=False, backup=True)

        self._start_task("Volume Attention 重算", run)

    def _on_backtest(self):
        modes = self._mode_list()

        def run():
            from global_screener import run_backtest

            for m in modes:
                run_backtest(f"results_{m}.csv", backtest_output=f"backtest_{m}.csv")

        self._start_task("Tier 回測", run)

    def _on_attention(self):
        modes = self._mode_list()

        def run():
            from global_screener import run_backtest_attention_groups

            for m in modes:
                run_backtest_attention_groups(
                    results_csv_path=f"results_{m}.csv",
                    backtest_output=f"backtest_{m}_attention.csv",
                )

        self._start_task("Attention 分組回測", run)


def main():
    root = tk.Tk()
    InvestBotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
