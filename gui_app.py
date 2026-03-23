import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from eye_blink import EyeBlinkDetector
from head_pose import HeadPoseDetector
from shoulder_analysis import ShoulderAnalyzer
from eye_ball import EyeBallTracker
from audio_recorder import AudioRecorder
from audio_analyzer import AudioAnalyzer
import mediapipe as mp
import os
import shutil

# Courtroom AI - Dark Forensic Theme
COLORS = {
    'bg_primary':   '#0a0e14',
    'bg_secondary': '#0d1117',
    'bg_tertiary':  '#111820',
    'bg_input':     '#0d1520',
    'bg_header':    '#0b1320',
    'accent_primary':   '#00e5c0',
    'accent_secondary': '#00b89a',
    'accent_dim':       '#1a3a35',
    'success':  '#00e5c0',
    'warning':  '#ff4444',
    'danger':   '#ff4444',
    'text_primary':   '#c8d8e8',
    'text_secondary': '#4a6a7a',
    'text_dim':       '#2a4a5a',
    'border':   '#1a2a3a',
    'border_accent': '#1a3a35',
    'row_odd':  '#0d1520',
    'row_even': '#0a1018',
}


class LoginWindow:
    """Login interface for the lie detection system"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Lie Detection System - Login")
        
        # Get screen dimensions and calculate adaptive window size (90% of screen)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(False, False)
        self.root.configure(bg=COLORS['bg_primary'])
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup login UI - Courtroom AI style"""
        self.root.configure(bg=COLORS['bg_primary'])

        # ── Full-screen background canvas for subtle grid texture ─────────
        bg_canvas = tk.Canvas(self.root, bg=COLORS['bg_primary'],
                              highlightthickness=0)
        bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # ── Centered title (above card) ───────────────────────────────────
        title_frame = tk.Frame(self.root, bg=COLORS['bg_primary'])
        title_frame.place(relx=0.5, rely=0.28, anchor=tk.CENTER)

        tk.Label(
            title_frame,
            text="COURTROOM AI",
            font=("Courier", 28, "bold"),
            bg=COLORS['bg_primary'],
            fg=COLORS['accent_primary']
        ).pack()

        tk.Label(
            title_frame,
            text="Behavioral Analysis System",
            font=("Courier", 11),
            bg=COLORS['bg_primary'],
            fg=COLORS['text_secondary']
        ).pack(pady=(4, 0))

        # ── Login card ───────────────────────────────────────────────────
        # Outer border frame (1px teal border effect)
        card_border = tk.Frame(self.root, bg=COLORS['accent_secondary'], bd=0)
        card_border.place(relx=0.5, rely=0.56, anchor=tk.CENTER)

        card = tk.Frame(card_border, bg=COLORS['bg_tertiary'], bd=0)
        card.pack(padx=1, pady=1)

        # Card width via inner spacer
        spacer = tk.Frame(card, bg=COLORS['bg_tertiary'], width=380, height=1)
        spacer.pack()

        # SECURE ACCESS heading
        tk.Label(
            card,
            text="SECURE ACCESS",
            font=("Courier", 12, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary']
        ).pack(pady=(26, 20))

        # ── Field helper ─────────────────────────────────────────────────
        def add_field(parent, label_text, show=None):
            # Left-border accent bar + label row
            row = tk.Frame(parent, bg=COLORS['bg_tertiary'])
            row.pack(fill=tk.X, padx=30, pady=(0, 14))

            bar = tk.Frame(row, bg=COLORS['accent_primary'], width=3)
            bar.pack(side=tk.LEFT, fill=tk.Y)

            entry_bg = tk.Frame(row, bg=COLORS['bg_input'])
            entry_bg.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Placeholder label inside entry area
            lbl = tk.Label(
                entry_bg,
                text=label_text,
                font=("Courier", 9),
                bg=COLORS['bg_input'],
                fg=COLORS['text_secondary'],
                anchor=tk.W
            )
            lbl.pack(fill=tk.X, padx=10, pady=(6, 0))

            kwargs = dict(
                font=("Courier", 11),
                bg=COLORS['bg_input'],
                fg=COLORS['text_primary'],
                insertbackground=COLORS['accent_primary'],
                relief=tk.FLAT,
                bd=0,
                highlightthickness=0
            )
            if show:
                kwargs['show'] = show
            e = tk.Entry(entry_bg, **kwargs)
            e.pack(fill=tk.X, padx=10, pady=(2, 8))
            return e

        self.username_entry = add_field(card, "USERNAME")
        self.username_entry.focus()
        self.password_entry = add_field(card, "PASSWORD", show="●")
        self.password_entry.bind("<Return>", lambda e: self.login())

        # ── Button row ───────────────────────────────────────────────────
        btn_row = tk.Frame(card, bg=COLORS['bg_tertiary'])
        btn_row.pack(fill=tk.X, padx=30, pady=(6, 28))

        login_btn = tk.Button(
            btn_row,
            text="LOGIN",
            font=("Courier", 11, "bold"),
            bg=COLORS['accent_primary'],
            fg=COLORS['bg_primary'],
            command=self.login,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=10,
            activebackground=COLORS['accent_secondary'],
            activeforeground=COLORS['bg_primary']
        )
        login_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        exit_btn = tk.Button(
            btn_row,
            text="EXIT",
            font=("Courier", 11),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary'],
            command=self.root.quit,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=10,
            activebackground=COLORS['danger'],
            activeforeground=COLORS['text_primary']
        )
        exit_btn.pack(side=tk.LEFT)
    
    def login(self):
        """Verify login credentials"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if username == "liedet" and password == "liedet":
            self.root.destroy()
            # Launch main application
            root = tk.Tk()
            app = MainApplication(root)
            root.mainloop()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password!\n\nUsername: liedet\nPassword: liedet")
            self.password_entry.delete(0, tk.END)
            self.username_entry.delete(0, tk.END)
            self.username_entry.focus()


class MainApplication:
    """Main application window for lie detection analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Lie Detection System - Analysis")
        
        # Get screen dimensions and calculate adaptive window size (90% of screen)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 1.0)
        window_height = int(screen_height * 1.0)
        
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(False, False)
        self.root.configure(bg=COLORS['bg_primary'])
        
        # Store window dimensions for adaptive UI
        self.window_width = window_width
        self.window_height = window_height
        
        # Application state
        self.is_analyzing = False
        self.analysis_thread = None
        self.cap = None
        self.frame_count = 0
        self.analysis_duration = 40  # 40 seconds
        self.start_time = None
        self.last_update_time = 0  # For throttling UI updates
        
        # File upload tracking
        self.uploaded_video_path = None
        self.uploaded_audio_path = None
        self.video_resize_var = None  # Will be set in setup_ui

        # Recording output
        self.video_writer = None
        self.session_timestamp = None
        self.frame_buffer = []        # stores annotated frames during live analysis
        
        # Detectors
        self.eye_detector = None
        self.head_detector = None
        self.shoulder_analyzer = None
        self.eye_ball_tracker = None
        self.shared_face_mesh = None
        
        # Audio components
        self.audio_recorder = AudioRecorder(sample_rate=16000, duration=40)
        self.audio_analyzer = AudioAnalyzer(model_size="base")
        
        # Results storage
        self.analysis_results = {
            'eye_blinks': 0,
            'eye_flag': 0,
            'head_aversion_events': 0,
            'head_flag': 0,
            'shoulder_fidget_events': 0,
            'shoulder_flag': 0,
            'eyeball_flag': 0,
            'audio_flag': 0,
            'word_count': 0,
            'speech_rate': 0.0,
            'transcript': '',
            'lie_score': 0.0,
            'deviations': []
        }
        
        self.setup_ui()
        
        # Pre-initialize detectors in background to speed up first analysis
        threading.Thread(target=self.pre_initialize_detectors, daemon=True).start()
    
    def setup_ui(self):
        """Setup main application UI - Courtroom AI style"""

        # ── ttk style ────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("CA.TCombobox",
            fieldbackground=COLORS['bg_input'],
            background=COLORS['bg_tertiary'],
            foreground=COLORS['text_primary'],
            selectbackground=COLORS['bg_tertiary'],
            selectforeground=COLORS['accent_primary'],
            bordercolor=COLORS['border'],
            arrowcolor=COLORS['accent_primary'],
            padding=4)
        style.map("CA.TCombobox",
            fieldbackground=[("readonly", COLORS['bg_input'])],
            foreground=[("readonly", COLORS['text_primary'])],
            selectbackground=[("readonly", COLORS['bg_tertiary'])])

        # ── TOP HEADER BAR ────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=COLORS['bg_header'], height=38)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Left: app name
        tk.Label(
            header,
            text="COURTROOM AI",
            font=("Courier", 13, "bold"),
            bg=COLORS['bg_header'],
            fg=COLORS['accent_primary']
        ).pack(side=tk.LEFT, padx=14, pady=8)

        tk.Label(
            header,
            text="/  Behavioral Analysis System",
            font=("Courier", 9),
            bg=COLORS['bg_header'],
            fg=COLORS['text_secondary']
        ).pack(side=tk.LEFT, pady=10)

        # Right: version tag
        tk.Label(
            header,
            text="v2.0  ——  FORENSIC SUITE",
            font=("Courier", 8),
            bg=COLORS['bg_header'],
            fg=COLORS['text_dim']
        ).pack(side=tk.RIGHT, padx=14, pady=10)

        # Thin teal border under header
        tk.Frame(self.root, bg=COLORS['accent_secondary'], height=1).pack(fill=tk.X)

        # ── MAIN BODY ─────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=COLORS['bg_primary'])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # ═══════════════════ LEFT COLUMN ═════════════════════════════════
        left = tk.Frame(body, bg=COLORS['bg_primary'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # "LIVE FEED" label
        tk.Label(
            left,
            text="LIVE FEED",
            font=("Courier", 9),
            bg=COLORS['bg_primary'],
            fg=COLORS['text_secondary']
        ).pack(anchor=tk.W, pady=(0, 4))

        # ── Video canvas ─────────────────────────────────────────────────
        self.canvas_width  = int(self.window_width * 0.60)
        self.canvas_height = int(self.canvas_width * 9 / 16)
        max_h = self.window_height - 280
        if self.canvas_height > max_h:
            self.canvas_height = max_h
            self.canvas_width  = int(self.canvas_height * 16 / 9)

        canvas_border = tk.Frame(left, bg=COLORS['border'], bd=0)
        canvas_border.pack()

        self.canvas = tk.Canvas(
            canvas_border,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=COLORS['bg_secondary'],
            highlightthickness=0
        )
        self.canvas.pack(padx=1, pady=1)

        # ── Primary control row ───────────────────────────────────────────
        ctrl = tk.Frame(left, bg=COLORS['bg_primary'])
        ctrl.pack(fill=tk.X, pady=(8, 4))

        self.start_btn = tk.Button(
            ctrl,
            text="RUN ANALYSIS  (40s)",
            font=("Courier", 10, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['accent_primary'],
            command=self.start_analysis,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=9,
            activebackground=COLORS['accent_dim'],
            activeforeground=COLORS['accent_primary']
        )
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.stop_btn = tk.Button(
            ctrl,
            text="STOP",
            font=("Courier", 10, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['danger'],
            command=self.stop_analysis,
            cursor="hand2",
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=9,
            activebackground=COLORS['bg_tertiary'],
            activeforeground=COLORS['danger'],
            highlightthickness=1,
            highlightbackground=COLORS['danger'],
            highlightcolor=COLORS['danger']
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.calib_btn = tk.Button(
            ctrl,
            text="CALIBRATE",
            font=("Courier", 10),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            command=self.run_calibration,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=9,
            activebackground=COLORS['accent_dim'],
            activeforeground=COLORS['accent_primary']
        )
        self.calib_btn.pack(side=tk.LEFT)

        # ── File row ──────────────────────────────────────────────────────
        file_row = tk.Frame(left, bg=COLORS['bg_primary'])
        file_row.pack(fill=tk.X, pady=(4, 2))

        self.upload_video_btn = tk.Button(
            file_row,
            text="UPLOAD VIDEO",
            font=("Courier", 9),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            command=self.upload_video,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            activebackground=COLORS['accent_dim'],
            activeforeground=COLORS['accent_primary']
        )
        self.upload_video_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.upload_audio_btn = tk.Button(
            file_row,
            text="UPLOAD AUDIO",
            font=("Courier", 9),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            command=self.upload_audio,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            activebackground=COLORS['accent_dim'],
            activeforeground=COLORS['accent_primary']
        )
        self.upload_audio_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.analyze_files_btn = tk.Button(
            file_row,
            text="ANALYZE FILES",
            font=("Courier", 9, "bold"),
            bg=COLORS['accent_primary'],
            fg=COLORS['bg_primary'],
            command=self.start_analysis_from_files,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            activebackground=COLORS['accent_secondary'],
            activeforeground=COLORS['bg_primary']
        )
        self.analyze_files_btn.pack(side=tk.LEFT)

        # File status
        self.file_status_label = tk.Label(
            left,
            text="Files: None uploaded",
            font=("Courier", 8),
            bg=COLORS['bg_primary'],
            fg=COLORS['text_secondary']
        )
        self.file_status_label.pack(anchor=tk.W, pady=(2, 6))

        # ── VIDEO RESIZE QUALITY panel ────────────────────────────────────
        resize_panel = tk.Frame(left, bg=COLORS['bg_tertiary'])
        resize_panel.pack(fill=tk.X, pady=(0, 6))

        resize_top = tk.Frame(resize_panel, bg=COLORS['bg_tertiary'])
        resize_top.pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Label(
            resize_top,
            text="VIDEO RESIZE QUALITY",
            font=("Courier", 8, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_secondary']
        ).pack(side=tk.LEFT)

        resize_row = tk.Frame(resize_panel, bg=COLORS['bg_tertiary'])
        resize_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(
            resize_row,
            text="Resolution:",
            font=("Courier", 8),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.video_resize_var = tk.StringVar(value="100%")
        self.resize_combo = ttk.Combobox(
            resize_row,
            textvariable=self.video_resize_var,
            values=["25% (Faster)", "50% (Balanced)", "75% (Good)", "100% (Full Quality)"],
            state="readonly",
            width=24,
            font=("Courier", 8),
            style="CA.TCombobox"
        )
        self.resize_combo.pack(side=tk.LEFT)

        self.resize_info_label = tk.Label(
            resize_panel,
            text="100% = Full resolution  |  50% = 4x faster processing",
            font=("Courier", 7),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_secondary']
        )
        self.resize_info_label.pack(anchor=tk.W, padx=10, pady=(0, 8))

        # ── AUDIO METRICS panel ───────────────────────────────────────────
        audio_panel = tk.Frame(left, bg=COLORS['bg_tertiary'])
        audio_panel.pack(fill=tk.X, pady=(0, 6))

        tk.Frame(audio_panel, bg=COLORS['border'], height=1).pack(fill=tk.X)

        self.audio_status_label = tk.Label(
            audio_panel,
            text="Audio: Ready",
            font=("Courier", 9, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['success']
        )
        self.audio_status_label.pack(anchor=tk.W, padx=10, pady=(8, 4))

        def audio_row(parent, label, default):
            r = tk.Frame(parent, bg=COLORS['bg_tertiary'])
            r.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(r, text=label, font=("Courier", 8, "bold"),
                     bg=COLORS['bg_tertiary'], fg=COLORS['text_secondary'],
                     width=14, anchor=tk.W).pack(side=tk.LEFT)
            val = tk.Label(r, text=default, font=("Courier", 8),
                           bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'])
            val.pack(side=tk.LEFT, padx=6)
            return val

        self.audio_word_label = audio_row(audio_panel, "Words:", "0 (Threshold: >7)")
        self.audio_rate_label = audio_row(audio_panel, "Speech Rate:", "0.00 wps")

        tk.Label(
            audio_panel, text="Transcript:",
            font=("Courier", 7, "bold"),
            bg=COLORS['bg_tertiary'], fg=COLORS['text_secondary'], anchor=tk.W
        ).pack(anchor=tk.W, padx=10, pady=(6, 2))

        self.audio_transcript_text = tk.Text(
            audio_panel,
            font=("Courier", 8),
            bg=COLORS['bg_input'],
            fg=COLORS['text_primary'],
            height=3,
            relief=tk.FLAT,
            bd=0,
            insertbackground=COLORS['accent_primary']
        )
        self.audio_transcript_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.audio_transcript_text.config(state=tk.DISABLED)

        # ═══════════════════ RIGHT COLUMN ════════════════════════════════
        right_w = int(self.window_width * 0.34)
        right = tk.Frame(body, bg=COLORS['bg_primary'], width=right_w)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)
        right.pack_propagate(False)

        # ── SIGNAL STATUS panel ───────────────────────────────────────────
        sig_panel = tk.Frame(right, bg=COLORS['bg_tertiary'])
        sig_panel.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            sig_panel,
            text="SIGNAL STATUS",
            font=("Courier", 8),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_secondary']
        ).pack(anchor=tk.W, padx=10, pady=(8, 6))

        self.flag_labels = {}
        flag_names = [
            ("eye_flag",      "EYE BLINK FLAG"),
            ("head_flag",     "HEAD AVERSIONS FLAG"),
            ("shoulder_flag", "SHOULDER EVENTS FLAG"),
            ("eyeball_flag",  "EYE GAZE SHIFTS FLAG"),
            ("audio_flag",    "WORDS SPOKEN"),
        ]

        for i, (flag_key, flag_text) in enumerate(flag_names):
            bg = COLORS['row_odd'] if i % 2 == 0 else COLORS['row_even']
            row = tk.Frame(sig_panel, bg=bg)
            row.pack(fill=tk.X)

            tk.Label(
                row,
                text=flag_text,
                font=("Courier", 9, "bold"),
                bg=bg,
                fg=COLORS['text_primary'],
                anchor=tk.W
            ).pack(side=tk.LEFT, padx=10, pady=7, fill=tk.X, expand=True)

            pill = tk.Label(
                row,
                text="✓  CLEAR",
                font=("Courier", 8, "bold"),
                bg=bg,
                fg=COLORS['success'],
                anchor=tk.E
            )
            pill.pack(side=tk.RIGHT, padx=10, pady=7)
            self.flag_labels[flag_key] = pill

        # Speech Rate row (static label, no flag)
        sr_bg = COLORS['row_odd']
        sr_row = tk.Frame(sig_panel, bg=sr_bg)
        sr_row.pack(fill=tk.X)
        tk.Label(sr_row, text="SPEECH RATE", font=("Courier", 9, "bold"),
                 bg=sr_bg, fg=COLORS['text_primary'], anchor=tk.W
                 ).pack(side=tk.LEFT, padx=10, pady=7)
        self.speech_rate_status = tk.Label(
            sr_row, text="✓  CLEAR",
            font=("Courier", 8, "bold"),
            bg=sr_bg, fg=COLORS['success'], anchor=tk.E)
        self.speech_rate_status.pack(side=tk.RIGHT, padx=10, pady=7)

        # Deception warning
        self.warning_label = tk.Label(
            sig_panel,
            text="",
            font=("Courier", 10, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['danger'],
            wraplength=320,
            justify=tk.CENTER
        )
        self.warning_label.pack(pady=(4, 8))

        # ── RESULT BANNER (top of right panel) ───────────────────────────
        self.result_banner = tk.Frame(right, bg=COLORS['bg_tertiary'],
                                      highlightthickness=1,
                                      highlightbackground=COLORS['border_accent'])
        self.result_banner.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            self.result_banner,
            text="RESULT",
            font=("Courier", 8, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_secondary']
        ).pack(pady=(8, 2))

        self.result_main_label = tk.Label(
            self.result_banner,
            text="NO DECEPTION DETECTED",
            font=("Courier", 14, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['success']
        )
        self.result_main_label.pack()

        self.result_sub_label = tk.Label(
            self.result_banner,
            text="0 flags active — all clear",
            font=("Courier", 8, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_secondary']
        )
        self.result_sub_label.pack(pady=(2, 10))

        # ── Scrollable results text ───────────────────────────────────────
        results_wrap = tk.Frame(right, bg=COLORS['bg_primary'])
        results_wrap.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        sb = tk.Scrollbar(results_wrap, bg=COLORS['bg_tertiary'],
                          troughcolor=COLORS['bg_secondary'],
                          activebackground=COLORS['accent_primary'],
                          highlightthickness=0)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = tk.Text(
            results_wrap,
            font=("Courier", 9, "bold"),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            yscrollcommand=sb.set,
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=6,
            insertbackground=COLORS['accent_primary']
        )
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.results_text.yview)

        # ── STATUS BAR ────────────────────────────────────────────────────
        tk.Frame(self.root, bg=COLORS['border'], height=1).pack(fill=tk.X)
        status_bar = tk.Frame(self.root, bg=COLORS['bg_header'], height=28)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        self.status_label = tk.Label(
            status_bar,
            text="Ready for analysis",
            font=("Courier", 8),
            bg=COLORS['bg_header'],
            fg=COLORS['success']
        )
        self.status_label.pack(side=tk.LEFT, padx=14, pady=6)
    
    def update_status(self, message, color=None):
        """Update status bar message"""
        if color is None:
            color = COLORS['success']
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def update_flag_display(self):
        """Update flag status indicators and result banner"""
        flag_configs = {
            'eye_flag': self.analysis_results.get('eye_flag', 0),
            'head_flag': self.analysis_results.get('head_flag', 0),
            'shoulder_flag': self.analysis_results.get('shoulder_flag', 0),
            'eyeball_flag': self.analysis_results.get('eyeball_flag', 0),
            'audio_flag': self.analysis_results.get('audio_flag', 0)
        }
        detected_count = 0
        for flag_key, flag_value in flag_configs.items():
            if flag_value == 1:
                detected_count += 1
                self.flag_labels[flag_key].config(
                    text="✗  ALERT",
                    fg=COLORS['danger']
                )
            else:
                self.flag_labels[flag_key].config(
                    text="✓  CLEAR",
                    fg=COLORS['success']
                )

        # Update speech rate status
        audio_flag = self.analysis_results.get('audio_flag', 0)
        if audio_flag == 1:
            self.speech_rate_status.config(text="✗  ALERT", fg=COLORS['danger'])
        else:
            self.speech_rate_status.config(text="✓  CLEAR", fg=COLORS['success'])

        # ── 3-tier result banner logic ────────────────────────────────────
        if detected_count == 0:
            self.result_banner.config(highlightbackground=COLORS['border_accent'])
            self.result_main_label.config(
                text="NO DECEPTION DETECTED",
                fg=COLORS['success']
            )
            self.result_sub_label.config(
                text="0 flags active — all clear",
                fg=COLORS['text_secondary']
            )
            self.warning_label.config(text="")
        elif detected_count == 1:
            self.result_banner.config(highlightbackground=COLORS['border_accent'])
            self.result_main_label.config(
                text="CLEAR",
                fg=COLORS['success']
            )
            self.result_sub_label.config(
                text="1 flag active — minor concern",
                fg=COLORS['text_secondary']
            )
            self.warning_label.config(text="")
        elif detected_count == 2:
            self.result_banner.config(highlightbackground='#6b3a00')
            self.result_main_label.config(
                text="DECEPTION LIKELY",
                fg=COLORS['warning']
            )
            self.result_sub_label.config(
                text="2 flags active — suspicious behaviour",
                fg=COLORS['warning']
            )
            self.warning_label.config(
                text=f"2 WARNING FLAGS ACTIVE",
                fg=COLORS['warning']
            )
        else:
            self.result_banner.config(highlightbackground=COLORS['danger'])
            self.result_main_label.config(
                text="DECEPTION DETECTED",
                fg=COLORS['danger']
            )
            self.result_sub_label.config(
                text=f"{detected_count} flags active — high confidence",
                fg=COLORS['danger']
            )
            self.warning_label.config(
                text=f"{detected_count} WARNING FLAGS ACTIVE",
                fg=COLORS['danger']
            )
    
    def update_results_display(self, is_final=False):
        """Update the results text area with result at top"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        detected_count = sum([
            self.analysis_results.get('eye_flag', 0),
            self.analysis_results.get('head_flag', 0),
            self.analysis_results.get('shoulder_flag', 0),
            self.analysis_results.get('eyeball_flag', 0),
            self.analysis_results.get('audio_flag', 0)
        ])

        # ── RESULT at the very top ────────────────────────────────────────
        results_text = "=" * 45 + "\n"
        if detected_count == 0 or detected_count == 1:
            results_text += "RESULT: NO DECEPTION DETECTED\n"
            results_text += f"({detected_count} flag active — all clear)\n"
        elif detected_count == 2:
            results_text += "RESULT: DECEPTION LIKELY\n"
            results_text += f"(2 flags active — suspicious behaviour)\n"
        else:
            results_text += "RESULT: DECEPTION DETECTED\n"
            results_text += f"({detected_count} flags active — high confidence)\n"
        results_text += "=" * 45 + "\n\n"

        # ── Header ────────────────────────────────────────────────────────
        if is_final:
            results_text += "FINAL ANALYSIS RESULTS\n\n"
        else:
            results_text += "ANALYSIS IN PROGRESS\n\n"

        # ── Behavioral Analysis ───────────────────────────────────────────
        results_text += "BEHAVIORAL ANALYSIS:\n"
        results_text += f"  - Eye Blinks: {self.analysis_results['eye_blinks']}\n"
        results_text += f"  - Head Movements: {self.analysis_results['head_aversion_events']}\n"
        results_text += f"  - Shoulder Movements: {self.analysis_results['shoulder_fidget_events']}\n"
        results_text += f"  - Words Spoken: {self.analysis_results.get('word_count', 0)}\n"
        results_text += f"  - Speech Rate: {self.analysis_results.get('speech_rate', 0.0):.2f} wps\n\n"

        # ── Status Flags ──────────────────────────────────────────────────
        results_text += "STATUS FLAGS:\n"
        def get_flag_status(flag_value):
            if isinstance(flag_value, bool):
                return "DETECTED" if flag_value else "NORMAL"
            else:
                return "DETECTED" if int(flag_value) == 1 else "NORMAL"
        results_text += f"  - Excessive Blinking: {get_flag_status(self.analysis_results.get('eye_flag', 0))}\n"
        results_text += f"  - Looking Away: {get_flag_status(self.analysis_results.get('head_flag', 0))}\n"
        results_text += f"  - Body Movement: {get_flag_status(self.analysis_results.get('shoulder_flag', 0))}\n"
        results_text += f"  - Eye Gaze Shift: {get_flag_status(self.analysis_results.get('eyeball_flag', 0))}\n"
        results_text += f"  - Speech/Audio: {get_flag_status(self.analysis_results.get('audio_flag', 0))} ({self.analysis_results.get('word_count', 0)} words)\n\n"

        # ── Confidence Score ──────────────────────────────────────────────
        results_text += "CONFIDENCE SCORE:\n"
        results_text += f"  - Score: {self.analysis_results['lie_score']:.2f}\n"
        results_text += f"  - Higher score = More suspicious\n\n"

        # ── Behaviors Detected ────────────────────────────────────────────
        results_text += "BEHAVIORS DETECTED:\n"
        if self.analysis_results['deviations']:
            for i, deviation in enumerate(self.analysis_results['deviations'], 1):
                results_text += f"  {i}. {deviation}\n"
        else:
            results_text += "  - None (Behavior appears normal)\n"

        # ── Transcript ────────────────────────────────────────────────────
        transcript = self.analysis_results.get('transcript', '')
        results_text += "\n" + "=" * 45 + "\n"
        results_text += "TRANSCRIPT:\n"
        results_text += "=" * 45 + "\n"
        if transcript:
            results_text += f"{transcript}\n"
        else:
            results_text += "[No speech detected or transcription pending]\n"

        results_text += "\n" + "=" * 45 + "\n"
        if is_final:
            results_text += "ANALYSIS COMPLETE!\n"
        results_text += "Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"

        self.results_text.insert(1.0, results_text)
        self.results_text.config(state=tk.DISABLED)
    
    def initialize_detectors(self):
        """Initialize all detection modules (only if not already initialized)"""
        try:
            # Skip if already initialized
            if self.shared_face_mesh is not None:
                return True
                
            self.shared_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True
            )
            
            self.eye_detector = EyeBlinkDetector(face_mesh=self.shared_face_mesh)
            self.head_detector = HeadPoseDetector(calibration_file='head_calibration.json')
            self.shoulder_analyzer = ShoulderAnalyzer()
            self.eye_ball_tracker = EyeBallTracker(face_mesh=self.shared_face_mesh, fps=30)
            
            return True
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize detectors:\n{str(e)}")
            return False
    
    def pre_initialize_detectors(self):
        """Pre-initialize detectors in background to speed up first analysis"""
        try:
            print("Pre-initializing detectors in background...")
            self.initialize_detectors()
            print("Detectors pre-initialized successfully!")
        except Exception as e:
            print(f"Pre-initialization warning: {e}")
    
    def show_loading_overlay(self):
        """Show animated STARTING ANALYSIS spinner on the video canvas"""
        self._loading_active = True
        self._spinner_chars = ['|', '/', '-', '\\']
        self._spinner_idx = 0
        self._loading_dots = 0

        def animate():
            if not self._loading_active:
                return
            self.canvas.delete("loading")
            w = self.canvas_width
            h = self.canvas_height
            # Dark background
            self.canvas.create_rectangle(0, 0, w, h,
                fill=COLORS['bg_primary'], outline='', tags="loading")
            # Spinner
            spin = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
            self._spinner_idx += 1
            dots = '.' * (self._loading_dots % 4)
            self._loading_dots += 1
            self.canvas.create_text(w // 2, h // 2 - 28,
                text=spin,
                font=("Courier", 40, "bold"),
                fill=COLORS['accent_primary'],
                tags="loading")
            self.canvas.create_text(w // 2, h // 2 + 20,
                text=f"STARTING ANALYSIS{dots}",
                font=("Courier", 14, "bold"),
                fill=COLORS['text_primary'],
                tags="loading")
            self.canvas.create_text(w // 2, h // 2 + 50,
                text="Initializing camera & AI models",
                font=("Courier", 9),
                fill=COLORS['text_secondary'],
                tags="loading")
            if self._loading_active:
                self.root.after(150, animate)

        animate()

    def hide_loading_overlay(self):
        """Remove loading overlay"""
        self._loading_active = False
        self.canvas.delete("loading")

    def start_analysis(self):
        """Start the 40-second analysis"""
        if self.is_analyzing:
            messagebox.showwarning("Analysis Running", "Analysis is already in progress!")
            return

        # Disable button and show spinner immediately
        self.start_btn.config(state=tk.DISABLED)
        self.update_status("Starting analysis...", COLORS['warning'])
        self.show_loading_overlay()

        def _init_and_start():
            # Open webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.root.after(0, self.hide_loading_overlay)
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: messagebox.showerror(
                    "Webcam Error", "Could not open webcam. Please check your camera."))
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Initialize detectors
            if not self.initialize_detectors():
                self.cap.release()
                self.root.after(0, self.hide_loading_overlay)
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                return

            if not all([self.eye_ball_tracker, self.eye_detector,
                        self.head_detector, self.shoulder_analyzer]):
                self.cap.release()
                self.root.after(0, self.hide_loading_overlay)
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: messagebox.showerror(
                    "Initialization Error", "Detectors not fully initialized. Please try again."))
                return

            # All ready — launch on main thread
            self.root.after(0, self._launch_analysis)

        threading.Thread(target=_init_and_start, daemon=True).start()

    def _get_session_paths(self):
        """Return (video_path, audio_path, result_path) inside the analysis/ folder."""
        ts = self.session_timestamp
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
        os.makedirs(base, exist_ok=True)   # create folder if it doesn't exist
        return (
            os.path.join(base, f"session_{ts}.mp4"),
            os.path.join(base, f"session_{ts}.wav"),
            os.path.join(base, f"session_{ts}_result.txt"),
        )

    def _start_video_writer(self, frame_width, frame_height, fps):
        """Open an mp4 VideoWriter with the true playback fps."""
        video_path, _, _ = self._get_session_paths()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps,
                                            (frame_width, frame_height))
        print(f"🎬 Video writer opened → {video_path}  @ {fps:.2f} fps")

    def _stop_video_writer(self):
        """Flush and close the VideoWriter."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            video_path, _, _ = self._get_session_paths()
            print(f"✅ Video saved → {video_path}")

    def _write_video_from_buffer(self, frames, elapsed_seconds):
        """Write all buffered annotated frames to mp4 using the true fps."""
        if not frames:
            print("⚠️  No frames to save")
            return
        h, w = frames[0].shape[:2]
        # Compute real fps — how many frames were actually captured per second
        true_fps = max(1.0, len(frames) / max(elapsed_seconds, 1.0))
        # Round to a sane value that most players handle well
        true_fps = round(true_fps, 2)
        print(f"🎬 Saving {len(frames)} frames at {true_fps:.2f} fps "
              f"(duration ≈ {elapsed_seconds:.1f}s) …")
        self._start_video_writer(w, h, true_fps)
        for f in frames:
            self.video_writer.write(f)
        self._stop_video_writer()

    def _save_result_txt(self):
        """Write the final analysis results to a .txt file next to the project."""
        _, _, result_path = self._get_session_paths()
        try:
            detected_count = sum([
                self.analysis_results.get('eye_flag', 0),
                self.analysis_results.get('head_flag', 0),
                self.analysis_results.get('shoulder_flag', 0),
                self.analysis_results.get('eyeball_flag', 0),
                self.analysis_results.get('audio_flag', 0),
            ])

            lines = []
            lines.append("=" * 50)
            lines.append("COURTROOM AI — BEHAVIORAL ANALYSIS REPORT")
            lines.append("=" * 50)
            lines.append(f"Session   : {self.session_timestamp}")
            lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")

            if detected_count == 0 or detected_count == 1:
                verdict = "NO DECEPTION DETECTED"
            elif detected_count == 2:
                verdict = "DECEPTION LIKELY"
            else:
                verdict = "DECEPTION DETECTED"
            lines.append(f"VERDICT   : {verdict}  ({detected_count} flags active)")
            lines.append("")

            lines.append("BEHAVIORAL METRICS:")
            lines.append(f"  Eye Blinks        : {self.analysis_results['eye_blinks']}")
            lines.append(f"  Head Movements    : {self.analysis_results['head_aversion_events']}")
            lines.append(f"  Shoulder Movements: {self.analysis_results['shoulder_fidget_events']}")
            lines.append(f"  Words Spoken      : {self.analysis_results.get('word_count', 0)}")
            lines.append(f"  Speech Rate       : {self.analysis_results.get('speech_rate', 0.0):.2f} wps")
            lines.append(f"  Confidence Score  : {self.analysis_results['lie_score']:.2f}")
            lines.append("")

            def fs(v): return "DETECTED" if int(v) == 1 else "NORMAL"
            lines.append("STATUS FLAGS:")
            lines.append(f"  Excessive Blinking: {fs(self.analysis_results.get('eye_flag', 0))}")
            lines.append(f"  Looking Away      : {fs(self.analysis_results.get('head_flag', 0))}")
            lines.append(f"  Body Movement     : {fs(self.analysis_results.get('shoulder_flag', 0))}")
            lines.append(f"  Eye Gaze Shift    : {fs(self.analysis_results.get('eyeball_flag', 0))}")
            lines.append(f"  Speech / Audio    : {fs(self.analysis_results.get('audio_flag', 0))}")
            lines.append("")

            lines.append("BEHAVIORS DETECTED:")
            if self.analysis_results['deviations']:
                for i, d in enumerate(self.analysis_results['deviations'], 1):
                    lines.append(f"  {i}. {d}")
            else:
                lines.append("  None — behaviour appears normal")
            lines.append("")

            lines.append("=" * 50)
            lines.append("TRANSCRIPT:")
            lines.append("=" * 50)
            transcript = self.analysis_results.get('transcript', '')
            lines.append(transcript if transcript else "[No speech detected]")
            lines.append("")
            lines.append("=" * 50)
            lines.append("END OF REPORT")

            with open(result_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"📄 Result saved → {result_path}")
        except Exception as e:
            print(f"❌ Failed to save result TXT: {e}")

    def _save_audio_to_session(self, temp_wav):
        """Copy the temp WAV into the project folder with the session name."""
        _, audio_path, _ = self._get_session_paths()
        try:
            if temp_wav and os.path.exists(temp_wav):
                shutil.copy2(temp_wav, audio_path)
                print(f"🎵 Audio saved → {audio_path}")
        except Exception as e:
            print(f"❌ Failed to save session audio: {e}")

    def _launch_analysis(self):
        """Called on main thread once camera and detectors are ready"""
        self.hide_loading_overlay()

        # New session timestamp
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_buffer = []

        # Reset all detector states
        if self.eye_ball_tracker:
            self.eye_ball_tracker.reset()
        if self.eye_detector:
            self.eye_detector.reset()
        if self.head_detector:
            self.head_detector.reset()
        if self.shoulder_analyzer:
            self.shoulder_analyzer.reset()

        self.analysis_results = {
            'eye_blinks': 0,
            'eye_flag': 0,
            'head_aversion_events': 0,
            'head_flag': 0,
            'shoulder_fidget_events': 0,
            'shoulder_flag': 0,
            'eyeball_flag': 0,
            'audio_flag': 0,
            'word_count': 0,
            'speech_rate': 0.0,
            'transcript': '',
            'lie_score': 0.0,
            'deviations': []
        }
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        for flag_label in self.flag_labels.values():
            flag_label.config(text="WAITING...", fg=COLORS['text_secondary'])
        self.warning_label.config(text="")

        # Update audio status display
        self.audio_status_label.config(text="🎤 Audio: Recording...", fg=COLORS['warning'])
        self.audio_transcript_text.config(state=tk.NORMAL)
        self.audio_transcript_text.delete(1.0, tk.END)
        self.audio_transcript_text.insert(1.0, "[Recording audio...]")
        self.audio_transcript_text.config(state=tk.DISABLED)

        # Reset and start audio recording
        self.audio_recorder.start_recording()
        self.audio_analyzer.reset()

        self.is_analyzing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_status("ANALYZING (40 seconds)...", COLORS['accent_primary'])
        self.analysis_thread = threading.Thread(target=self.analyze_video, daemon=True)
        self.analysis_thread.start()
    
    def analyze_video(self):
        """Main analysis loop (runs for 40 seconds)"""
        try:
            while self.is_analyzing:
                elapsed_time = time.time() - self.start_time
                
                # Stop after 40 seconds
                if elapsed_time >= self.analysis_duration:
                    self.is_analyzing = False
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip for selfie view
                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                
                # Process frame with all detectors
                try:
                    # Check if detectors are initialized
                    if not all([self.eye_ball_tracker, self.eye_detector, self.head_detector, self.shoulder_analyzer]):
                        continue
                    
                    # Eye ball tracking
                    annotated_frame, gaze_text, eyeball_flag = self.eye_ball_tracker.process_frame(frame, annotate=True)
                    frame = annotated_frame
                    
                    # Eye blink detection
                    eye_data = self.eye_detector.process_frame(frame.copy())
                    if eye_data is None or 'image' not in eye_data:
                        continue
                    frame = eye_data['image']
                    
                    # Head pose detection
                    head_data = self.head_detector.process_frame(frame)
                    if head_data is None or 'image' not in head_data:
                        continue
                    frame = head_data['image']
                    
                    # Shoulder analysis
                    shoulder_data = self.shoulder_analyzer.process_frame(frame)
                    if shoulder_data is None or 'image' not in shoulder_data:
                        continue
                    frame = shoulder_data['image']
                    
                    # Update results
                    self.analysis_results['eye_blinks'] = eye_data['blinks']
                    self.analysis_results['eye_flag'] = int(eye_data.get('eye_flag', 0))
                    self.analysis_results['head_aversion_events'] = head_data['aversion_events']
                    self.analysis_results['head_flag'] = int(head_data.get('lie_chance_flag', 0))
                    self.analysis_results['shoulder_fidget_events'] = shoulder_data['fidget_events']
                    self.analysis_results['shoulder_flag'] = int(shoulder_data.get('shoulder_flag', 0))
                    self.analysis_results['eyeball_flag'] = int(eyeball_flag) if eyeball_flag is not None else 0
                    
                    # Calculate composite lie score
                    blink_rate = eye_data['blinks'] / (elapsed_time + 1)
                    self.analysis_results['lie_score'] = (blink_rate * 10) + \
                                                        (head_data['aversion_events'] * 20) + \
                                                        (shoulder_data['fidget_events'] * 15)
                    
                    # Generate deviations
                    deviations = []
                    if eye_data['eye_flag'] == 1:
                        deviations.append("Excessive blinking detected")
                    if head_data.get('lie_chance_flag', 0) == 1:
                        deviations.append("Head aversion detected")
                    if shoulder_data['shoulder_flag'] == 1:
                        deviations.append("Shoulder asymmetry detected")
                    if eyeball_flag == 1:
                        deviations.append("Sustained gaze shift detected")
                    
                    self.analysis_results['deviations'] = deviations
                    
                    # Print Eye metrics every 30 frames to avoid spam
                    if self.frame_count % 30 == 0:
                        ear_threshold = 0.23  # From EyeBlinkDetector
                        blinks = eye_data['blinks']
                        print(f"Frame {self.frame_count} | EAR: {eye_data.get('ear', 0):.3f} | Threshold: {ear_threshold} | Blinks: {blinks}")
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
                
                # Add timer and frame count to display
                remaining_time = max(0, self.analysis_duration - elapsed_time)
                cv2.putText(
                    frame,
                    f"Time: {remaining_time:.1f}s | Frames: {self.frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Add countdown timer at bottom left corner
                countdown = int(self.analysis_duration - elapsed_time)
                cv2.putText(
                    frame,
                    str(countdown),
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    4
                )

                # ── Buffer the annotated frame for post-loop video saving ──
                self.frame_buffer.append(frame.copy())

                # Display frame with webcam source info
                source_info = f"📹 WEBCAM LIVE | Frame: {self.frame_count} | {countdown}s"
                self.display_frame(frame, source_info=source_info)
                
                # Update results display only every 0.5 seconds (not every frame)
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5:
                    self.update_results_display()
                    self.update_flag_display()  # Update flag colors
                    self.last_update_time = current_time
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis:\n{str(e)}")
        finally:
            self.audio_recorder.stop_recording()
            self.cleanup_analysis()
    
    def display_frame(self, frame, source_info=None):
        """Display video frame on canvas (matching live webcam display)"""
        try:
            # Add source info to frame before resizing (just like live webcam)
            if source_info:
                cv2.putText(
                    frame,
                    source_info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 217, 255),  # Cyan color
                    2
                )
            
            # Resize frame to fit canvas (simple and clean - no borders)
            resized_frame = cv2.resize(frame, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def display_black_frame(self):
        """Display a black frame on the video canvas (1280x720)"""
        import numpy as np
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo
    
    def stop_analysis(self):
        """Stop the analysis"""
        self.is_analyzing = False
        self.audio_recorder.stop_recording()
        self.update_status("Analysis stopped by user", COLORS['warning'])
        self.cleanup_analysis()
    
    def cleanup_analysis(self):
        """Cleanup resources after analysis and process audio"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(f"Cleanup camera error: {e}")

        # Finalise video recording from buffer (done in background so UI stays live)
        if self.frame_buffer:
            elapsed = time.time() - self.start_time
            frames_to_save = list(self.frame_buffer)
            self.frame_buffer = []
            threading.Thread(
                target=self._write_video_from_buffer,
                args=(frames_to_save, elapsed),
                daemon=True
            ).start()

        self.is_analyzing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Processing audio transcription...", COLORS['warning'])

        if self.frame_count > 0:
            self.update_results_display(is_final=True)
            self.update_flag_display()
            self.display_black_frame()

        # Process audio in background so UI stays responsive
        def _process_audio():
            try:
                audio_file = self.audio_recorder.save_recording()

                if audio_file:
                    print(f"🎤 Transcribing audio using Whisper...")
                    audio_results = self.audio_analyzer.transcribe_audio(audio_file, duration=40)

                    if audio_results:
                        self.analysis_results['word_count']  = audio_results['word_count']
                        self.analysis_results['speech_rate'] = audio_results['speech_rate']
                        self.analysis_results['audio_flag']  = audio_results['audio_flag']
                        self.analysis_results['transcript']  = audio_results['transcript']

                        def _update_audio_ui():
                            self.audio_status_label.config(
                                text=f"✅ Audio: Complete ({audio_results['word_count']} words)",
                                fg=COLORS['success'])
                            self.audio_word_label.config(
                                text=f"{audio_results['word_count']} (Threshold: >7{'  ✓' if audio_results['audio_flag'] == 0 else '  ✗'})")
                            self.audio_rate_label.config(
                                text=f"{audio_results['speech_rate']:.2f} wps")
                            self.audio_transcript_text.config(state=tk.NORMAL)
                            self.audio_transcript_text.delete(1.0, tk.END)
                            transcript = audio_results['transcript'] if audio_results['transcript'] else "[No speech detected]"
                            self.audio_transcript_text.insert(1.0, transcript)
                            self.audio_transcript_text.config(state=tk.DISABLED)
                            self.update_status("Analysis Complete - Review Results", COLORS['success'])
                            self.update_results_display(is_final=True)
                            self.update_flag_display()
                            # ── Persist session files ──────────────────────────
                            self._save_audio_to_session(audio_file)
                            self._save_result_txt()
                            print(f"📝 Audio complete: {audio_results['word_count']} words, Flag: {audio_results['audio_flag']}")
                        self.root.after(0, _update_audio_ui)
                    else:
                        print("⚠️  Audio transcription failed")
                        self.root.after(0, lambda: self.audio_status_label.config(
                            text="❌ Audio: Transcription failed", fg=COLORS['danger']))
                        self.root.after(0, lambda: self.update_status("Analysis Complete - Review Results", COLORS['success']))
                        self.root.after(0, self._save_result_txt)
                else:
                    print("⚠️  Audio recording save failed")
                    self.root.after(0, lambda: self.audio_status_label.config(
                        text="❌ Audio: Save failed", fg=COLORS['danger']))
                    self.root.after(0, lambda: self.update_status("Analysis Complete - Review Results", COLORS['success']))
                    self.root.after(0, self._save_result_txt)

            except Exception as e:
                print(f"Audio processing error: {e}")
                self.root.after(0, lambda: self.audio_status_label.config(
                    text=f"❌ Audio: Error - {str(e)[:30]}", fg=COLORS['danger']))
                self.root.after(0, lambda: self.update_status("Analysis Complete - Review Results", COLORS['success']))

        threading.Thread(target=_process_audio, daemon=True).start()
    
    def run_calibration(self):
        """Run head pose calibration for 40 seconds (100 frames) and save to head_calibration.json, showing video in canvas."""
        import numpy as np
        import mediapipe as mp
        import json
        import time
        import cv2
        def calibration_task():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Calibration Error", "Could not open webcam for calibration.")
                return
            face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
            yaw_list = []
            TARGET_FRAMES = 100
            frame_num = 0
            start_time = time.time()
            while frame_num < TARGET_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    left_eye = landmarks[33]
                    right_eye = landmarks[362]
                    nose = landmarks[1]
                    eye_center_x = (left_eye.x + right_eye.x) / 2
                    delta_x = nose.x - eye_center_x
                    yaw = np.degrees(np.arctan2(delta_x, 0.18))
                    yaw_list.append(yaw)
                # Show frame in Tkinter canvas (adaptive size)
                if frame.shape[1] == self.canvas_width and frame.shape[0] == self.canvas_height:
                    frame_disp = frame
                else:
                    frame_disp = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.image = photo
                # Show progress in status bar
                elapsed = time.time() - start_time
                self.update_status(f"Calibrating... {frame_num+1}/100 frames ({int(elapsed)}s)", COLORS['accent_primary'])
                self.root.update()
                frame_num += 1
            cap.release()
            face_mesh.close()
            self.display_black_frame()  # Black out video after calibration
            if len(yaw_list) > 70:
                neutral_yaw = float(np.median(yaw_list))
                json.dump({"neutral_yaw": neutral_yaw}, open('head_calibration.json', 'w'), indent=2)
                messagebox.showinfo("Calibration Complete", f"Calibration successful!\nNeutral yaw: {neutral_yaw:.1f}° (saved)")
                self.update_status("Calibration complete. Ready for analysis.", COLORS['success'])
            else:
                messagebox.showerror("Calibration Failed", "Not enough face detections. Please try again.")
                self.update_status("Calibration failed. Try again.", COLORS['danger'])
        # Run calibration in a thread to keep UI responsive
        threading.Thread(target=calibration_task, daemon=True).start()
    
    def upload_video(self):
        """Browse and select a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.uploaded_video_path = file_path
            self.update_file_status_display()
            messagebox.showinfo("Video Uploaded", f"Video loaded:\n{os.path.basename(file_path)}")
    
    def upload_audio(self):
        """Browse and select an audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.aac *.flac *.ogg *.m4a"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.uploaded_audio_path = file_path
            self.update_file_status_display()
            messagebox.showinfo("Audio Uploaded", f"Audio loaded:\n{os.path.basename(file_path)}")
    
    def update_file_status_display(self):
        """Update the file status label"""
        status_parts = []
        if self.uploaded_video_path:
            status_parts.append(f"📽 Video: {os.path.basename(self.uploaded_video_path)}")
        if self.uploaded_audio_path:
            status_parts.append(f"🎵 Audio: {os.path.basename(self.uploaded_audio_path)}")
        
        if status_parts:
            self.file_status_label.config(
                text="📁 " + " | ".join(status_parts),
                fg=COLORS['success']
            )
        else:
            self.file_status_label.config(
                text="📁 Files: None uploaded",
                fg=COLORS['text_secondary']
            )
    
    def start_analysis_from_files(self):
        """Start analysis using uploaded video and/or audio files"""
        if not self.uploaded_video_path and not self.uploaded_audio_path:
            messagebox.showwarning("No Files", "Please upload at least a video or audio file first!")
            return
        
        if self.is_analyzing:
            messagebox.showwarning("Analysis Running", "Analysis is already in progress!")
            return
        
        # Reset results
        self.analysis_results = {
            'eye_blinks': 0,
            'eye_flag': 0,
            'head_aversion_events': 0,
            'head_flag': 0,
            'shoulder_fidget_events': 0,
            'shoulder_flag': 0,
            'eyeball_flag': 0,
            'audio_flag': 0,
            'word_count': 0,
            'speech_rate': 0.0,
            'transcript': '',
            'lie_score': 0.0,
            'deviations': []
        }
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update_time = time.time()

        # New session timestamp for file analysis
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Reset UI
        for flag_label in self.flag_labels.values():
            flag_label.config(text="WAITING", bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'])
        self.warning_label.config(text="")
        
        # Update UI states
        self.update_status("Initializing AI models...", COLORS['warning'])
        self.root.update()
        
        if not self.initialize_detectors():
            return
        
        # Reset detector states
        if self.eye_ball_tracker:
            self.eye_ball_tracker.reset()
        if self.eye_detector:
            self.eye_detector.reset()
        if self.head_detector:
            self.head_detector.reset()
        if self.shoulder_analyzer:
            self.shoulder_analyzer.reset()
        
        # Update audio UI
        self.audio_status_label.config(text="🎤 Audio: Processing...", fg=COLORS['warning'])
        self.audio_transcript_text.config(state=tk.NORMAL)
        self.audio_transcript_text.delete(1.0, tk.END)
        self.audio_transcript_text.insert(1.0, "[Processing audio...]")
        self.audio_transcript_text.config(state=tk.DISABLED)
        
        self.is_analyzing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.upload_video_btn.config(state=tk.DISABLED)
        self.upload_audio_btn.config(state=tk.DISABLED)
        self.analyze_files_btn.config(state=tk.DISABLED)
        
        self.update_status("ANALYZING FILES...", COLORS['accent_primary'])
        self.analysis_thread = threading.Thread(target=self.analyze_from_files, daemon=True)
        self.analysis_thread.start()
    
    def get_resize_scale(self):
        """Get the resize scale based on selected option"""
        if not self.video_resize_var:
            return 1.0
        
        selected = self.video_resize_var.get()
        if "25%" in selected:
            return 0.25
        elif "50%" in selected:
            return 0.50
        elif "75%" in selected:
            return 0.75
        else:  # 100%
            return 1.0
    
    def analyze_from_files(self):
        """Main analysis loop for uploaded files"""
        try:
            # Process video if provided
            if self.uploaded_video_path:
                self.analyze_video_file(self.uploaded_video_path)
            
            # Process audio if provided
            if self.uploaded_audio_path:
                self.process_audio_file(self.uploaded_audio_path)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during file analysis:\n{str(e)}")
        finally:
            self.cleanup_file_analysis()
    
    def analyze_video_file(self, video_path):
        """Analyze a video file"""
        try:
            self.update_status("Processing video...", COLORS['warning'])
            self.root.update()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Video Error", f"Could not open video file:\n{video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 40
            
            # Get resize scale from UI selection
            resize_scale = self.get_resize_scale()
            new_width = int(orig_width * resize_scale)
            new_height = int(orig_height * resize_scale)
            
            # Limit analysis to 40 seconds max
            max_frames = min(total_frames, int(fps * 40) if fps > 0 else 1200)
            frame_count = 0
            
            # Update status with resolution info
            resolution_info = f"({orig_width}×{orig_height} → {new_width}×{new_height})" if resize_scale != 1.0 else f"({orig_width}×{orig_height})"
            self.update_status(f"Analyzing video ({duration:.1f}s) {resolution_info}...", COLORS['accent_primary'])
            
            while self.is_analyzing and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply resize if needed
                if resize_scale != 1.0:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                frame_count += 1
                self.frame_count = frame_count
                
                # Process frame with detectors
                try:
                    if not all([self.eye_ball_tracker, self.eye_detector, self.head_detector, self.shoulder_analyzer]):
                        continue
                    
                    # Eye ball tracking
                    annotated_frame, gaze_text, eyeball_flag = self.eye_ball_tracker.process_frame(frame, annotate=True)
                    frame = annotated_frame
                    
                    # Eye blink detection
                    eye_data = self.eye_detector.process_frame(frame.copy())
                    if eye_data is None or 'image' not in eye_data:
                        continue
                    frame = eye_data['image']
                    
                    # Head pose detection
                    head_data = self.head_detector.process_frame(frame)
                    if head_data is None or 'image' not in head_data:
                        continue
                    frame = head_data['image']
                    
                    # Shoulder analysis
                    shoulder_data = self.shoulder_analyzer.process_frame(frame)
                    if shoulder_data is None or 'image' not in shoulder_data:
                        continue
                    frame = shoulder_data['image']
                    
                    # Update results
                    self.analysis_results['eye_blinks'] = eye_data['blinks']
                    self.analysis_results['eye_flag'] = int(eye_data.get('eye_flag', 0))
                    self.analysis_results['head_aversion_events'] = head_data['aversion_events']
                    self.analysis_results['head_flag'] = int(head_data.get('lie_chance_flag', 0))
                    self.analysis_results['shoulder_fidget_events'] = shoulder_data['fidget_events']
                    self.analysis_results['shoulder_flag'] = int(shoulder_data.get('shoulder_flag', 0))
                    self.analysis_results['eyeball_flag'] = int(eyeball_flag) if eyeball_flag is not None else 0
                    
                    # Calculate lie score
                    elapsed_time = frame_count / (fps if fps > 0 else 30)
                    blink_rate = eye_data['blinks'] / max(elapsed_time, 1)
                    self.analysis_results['lie_score'] = (blink_rate * 10) + \
                                                        (head_data['aversion_events'] * 20) + \
                                                        (shoulder_data['fidget_events'] * 15)
                    
                    # Generate deviations
                    deviations = []
                    if eye_data['eye_flag'] == 1:
                        deviations.append("Excessive blinking detected")
                    if head_data.get('lie_chance_flag', 0) == 1:
                        deviations.append("Head aversion detected")
                    if shoulder_data['shoulder_flag'] == 1:
                        deviations.append("Shoulder asymmetry detected")
                    if eyeball_flag == 1:
                        deviations.append("Sustained gaze shift detected")
                    
                    self.analysis_results['deviations'] = deviations
                    
                except Exception as e:
                    print(f"Video frame processing error: {e}")
                    continue
                
                # Display frame with video info
                video_name = os.path.basename(video_path)
                frame_fps = fps if fps > 0 else 30
                resolution_display = f"{new_width}×{new_height}" if resize_scale != 1.0 else f"{orig_width}×{orig_height}"
                source_info = f"📹 {video_name} | {resolution_display} | Frame: {frame_count}/{max_frames}"
                self.display_frame(frame, source_info=source_info)
                
                # Update results display
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5:
                    progress = (frame_count / max_frames) * 100 if max_frames > 0 else 0
                    elapsed = frame_count / (fps if fps > 0 else 30)
                    resize_info = f" | {new_width}×{new_height}" if resize_scale != 1.0 else ""
                    self.update_status(f"Analyzing: {progress:.0f}% ({elapsed:.1f}s){resize_info}", COLORS['accent_primary'])
                    self.update_results_display()
                    self.update_flag_display()
                    self.last_update_time = current_time
            
            cap.release()
            
        except Exception as e:
            print(f"Video analysis error: {e}")
    
    def process_audio_file(self, audio_path):
        """Process an audio file"""
        try:
            self.update_status("🎤 Transcribing audio...", COLORS['warning'])
            self.root.update()
            
            # Show transcription in progress
            self.audio_transcript_text.config(state=tk.NORMAL)
            self.audio_transcript_text.delete(1.0, tk.END)
            self.audio_transcript_text.insert(1.0, "[Processing... Please wait]")
            self.audio_transcript_text.config(state=tk.DISABLED)
            self.root.update()
            
            # Transcribe using Whisper (duration=None allows auto-detection)
            print(f"🎤 Transcribing audio from file: {os.path.basename(audio_path)}")
            audio_results = self.audio_analyzer.transcribe_audio(audio_path, duration=None)
            
            if audio_results:
                # Update analysis results with audio data
                self.analysis_results['word_count'] = audio_results['word_count']
                self.analysis_results['speech_rate'] = audio_results['speech_rate']
                self.analysis_results['audio_flag'] = audio_results['audio_flag']
                self.analysis_results['transcript'] = audio_results['transcript']
                
                # Update UI with audio metrics
                self.audio_status_label.config(
                    text=f"✅ Audio: Complete ({audio_results['word_count']} words)",
                    fg=COLORS['success']
                )
                self.audio_word_label.config(
                    text=f"{audio_results['word_count']} (Threshold: >7{'  ✓' if audio_results['audio_flag'] == 0 else '  ✗'})"
                )
                self.audio_rate_label.config(
                    text=f"{audio_results['speech_rate']:.2f} wps"
                )
                
                # Update transcript display
                self.audio_transcript_text.config(state=tk.NORMAL)
                self.audio_transcript_text.delete(1.0, tk.END)
                transcript = audio_results['transcript'] if audio_results['transcript'] else "[No speech detected]"
                self.audio_transcript_text.insert(1.0, transcript)
                self.audio_transcript_text.config(state=tk.DISABLED)
                
                print(f"📝 Audio processing complete: {audio_results['word_count']} words, Speech rate: {audio_results['speech_rate']:.2f} wps")
            else:
                print("⚠️  Audio transcription failed")
                self.audio_status_label.config(
                    text="❌ Audio: Transcription failed - check console",
                    fg=COLORS['danger']
                )
                self.audio_transcript_text.config(state=tk.NORMAL)
                self.audio_transcript_text.delete(1.0, tk.END)
                self.audio_transcript_text.insert(1.0, "[Transcription failed]")
                self.audio_transcript_text.config(state=tk.DISABLED)
        
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            self.audio_status_label.config(
                text=f"❌ Audio: Error - {str(e)[:40]}",
                fg=COLORS['danger']
            )
            self.audio_transcript_text.config(state=tk.NORMAL)
            self.audio_transcript_text.delete(1.0, tk.END)
            self.audio_transcript_text.insert(1.0, f"[Error: {str(e)[:100]}]")
            self.audio_transcript_text.config(state=tk.DISABLED)
    
    def cleanup_file_analysis(self):
        """Cleanup after file analysis"""
        self.is_analyzing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.upload_video_btn.config(state=tk.NORMAL)
        self.upload_audio_btn.config(state=tk.NORMAL)
        self.analyze_files_btn.config(state=tk.NORMAL)
        
        self.display_black_frame()
        self.update_status("File Analysis Complete - Review Results", COLORS['success'])
        self.update_results_display(is_final=True)
        self.update_flag_display()
        self._save_result_txt()


def main():
    """Launch the application"""
    root = tk.Tk()
    app = LoginWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
