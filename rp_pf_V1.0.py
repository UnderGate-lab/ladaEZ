#!/usr/bin/env python3
"""
LADA Windows V1.0 - 30FPS最適化版
フルスクリーン進捗バー + 設定機能 + 再生速度制御 + 音声機能追加
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time
import json
from collections import OrderedDict
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QDialog, QSpinBox, QFormLayout, QDialogButtonBox, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QMutex, QMutexLocker, QTimer, QPoint
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QShortcut, QKeySequence, QDragEnterEvent, QDropEvent, QMouseEvent, QCursor
from OpenGL.GL import *

LADA_BASE_PATH = Path(r"./")
PYTHON_PATH = LADA_BASE_PATH / "python" / "Lib" / "site-packages"
sys.path.insert(0, str(PYTHON_PATH))

CONFIG_FILE = Path("lada_config.json")

LADA_AVAILABLE = False
try:
    import torch
    from lada.lib.frame_restorer import load_models
    from lada.lib import video_utils
    LADA_AVAILABLE = True
    print("✓ LADA利用可能")
except ImportError as e:
    print(f"✗ LADA: {e}")

# VLCのインポートを試みる
VLC_AVAILABLE = False
try:
    import vlc
    VLC_AVAILABLE = True
    print("✓ VLC利用可能")
except ImportError as e:
    print(f"✗ VLC: {e} - 音声機能は無効化されます")


class SettingsDialog(QDialog):
    """設定ダイアログ"""
    
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("AI処理設定")
        self.settings = current_settings or {}
        
        layout = QFormLayout(self)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(4, 64)
        self.batch_size_spin.setValue(self.settings.get('batch_size', 16))
        self.batch_size_spin.setSuffix(" frames")
        layout.addRow("バッチサイズ:", self.batch_size_spin)
        
        self.queue_size_spin = QSpinBox()
        self.queue_size_spin.setRange(2048, 32768)
        self.queue_size_spin.setSingleStep(1024)
        self.queue_size_spin.setValue(self.settings.get('queue_size_mb', 12288))
        self.queue_size_spin.setSuffix(" MB")
        layout.addRow("キューサイズ:", self.queue_size_spin)
        
        self.clip_length_spin = QSpinBox()
        self.clip_length_spin.setRange(1, 48)
        self.clip_length_spin.setValue(self.settings.get('max_clip_length', 8))
        self.clip_length_spin.setSuffix(" frames")
        layout.addRow("最大クリップ長:", self.clip_length_spin)
        
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(1024, 16384)
        self.cache_size_spin.setSingleStep(512)
        self.cache_size_spin.setValue(self.settings.get('cache_size_mb', 12288))
        self.cache_size_spin.setSuffix(" MB")
        layout.addRow("キャッシュサイズ:", self.cache_size_spin)
        
        info = QLabel(
            "※設定変更後、処理が完全リセットされます\n"
            "※高い値 = 高速だがメモリ消費大\n"
            "※30FPS達成推奨設定: バッチ16, キュー12GB, クリップ長8"
        )
        info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addRow(info)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_settings(self):
        return {
            'batch_size': self.batch_size_spin.value(),
            'queue_size_mb': self.queue_size_spin.value(),
            'max_clip_length': self.clip_length_spin.value(),
            'cache_size_mb': self.cache_size_spin.value()
        }


class FrameCache:
    def __init__(self, max_size_mb=12288):
        self.cache = OrderedDict()
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0
        self.mutex = QMutex()
    
    def get(self, frame_num):
        with QMutexLocker(self.mutex):
            if frame_num in self.cache:
                self.cache.move_to_end(frame_num)
                return self.cache[frame_num]
            return None
    
    def put(self, frame_num, frame):
        with QMutexLocker(self.mutex):
            if frame is None:
                # キャッシュ無効化用
                if frame_num in self.cache:
                    oldest_frame = self.cache.pop(frame_num)
                    self.current_size_mb -= oldest_frame.nbytes / (1024 * 1024)
                return
                
            frame_size_mb = frame.nbytes / (1024 * 1024)
            
            while self.current_size_mb + frame_size_mb > self.max_size_mb and len(self.cache) > 0:
                oldest_frame_num, oldest_frame = self.cache.popitem(last=False)
                self.current_size_mb -= oldest_frame.nbytes / (1024 * 1024)
            
            if self.current_size_mb + frame_size_mb <= self.max_size_mb:
                self.cache[frame_num] = frame.copy()
                self.current_size_mb += frame_size_mb
    
    def clear(self):
        with QMutexLocker(self.mutex):
            self.cache.clear()
            self.current_size_mb = 0
    
    def get_stats(self):
        with QMutexLocker(self.mutex):
            return {
                'count': len(self.cache),
                'size_mb': self.current_size_mb,
                'max_mb': self.max_size_mb
            }


class VideoGLWidget(QOpenGLWidget):
    playback_toggled = pyqtSignal()
    video_dropped = pyqtSignal(str)
    seek_requested = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = None
        self.is_fullscreen = False
        self.normal_parent_geometry = None
        self.parent_widget = None
        self.setMinimumSize(800, 450)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # D&Dを有効化
        self.setAcceptDrops(True)
        
        # フルスクリーン用UI
        self.fs_progress_bar = QProgressBar(self)
        self.fs_progress_bar.setTextVisible(False)
        self.fs_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(40, 40, 40, 200);
                border: none;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
            }
        """)
        self.fs_progress_bar.hide()
        
        self.fs_time_label = QLabel("00:00:00 / 00:00:00", self)
        self.fs_time_label.setStyleSheet("""
            QLabel {
                background-color: rgba(40, 40, 40, 200);
                color: white;
                padding: 4px 12px;
                font-size: 14px;
                border-radius: 4px;
            }
        """)
        self.fs_time_label.hide()
        
        # UI自動非表示タイマー
        self.ui_hide_timer = QTimer()
        self.ui_hide_timer.timeout.connect(self.hide_fs_ui)
        self.ui_hide_timer.setSingleShot(True)
        
        # 進捗情報
        self.total_frames = 0
        self.current_frame_num = 0
        self.video_fps = 30.0
        
        self.setMouseTracking(True)
        print("[DEBUG] D&D有効化 + フルスクリーンUI初期化")
    
    def set_video_info(self, total_frames, fps):
        """動画情報を設定"""
        self.total_frames = total_frames
        self.video_fps = fps
        self.fs_progress_bar.setMaximum(total_frames)
    
    def update_progress(self, frame_num):
        """進捗更新"""
        self.current_frame_num = frame_num
        self.fs_progress_bar.setValue(frame_num)
        
        # 時間表示更新
        current_sec = frame_num / self.video_fps if self.video_fps > 0 else 0
        total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        
        current_time = self.format_time(current_sec)
        total_time = self.format_time(total_sec)
        self.fs_time_label.setText(f"{current_time} / {total_time}")
    
    def format_time(self, seconds):
        """秒を HH:MM:SS 形式に変換"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def show_fs_ui(self):
        """フルスクリーンUI表示"""
        if self.is_fullscreen:
            # UI位置を強制更新
            self.update_fs_ui_position()
            
            self.fs_progress_bar.show()
            self.fs_time_label.show()
            QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
            self.ui_hide_timer.start(3000)  # 3秒後に自動非表示
    
    def update_fs_ui_position(self):
        """フルスクリーンUI位置更新"""
        if not self.is_fullscreen:
            return
            
        # 進捗バーを画面下部に配置
        bar_height = 8
        bar_margin = 20
        self.fs_progress_bar.setGeometry(
            bar_margin, 
            self.height() - bar_height - bar_margin, 
            self.width() - bar_margin * 2, 
            bar_height
        )
        
        # 時間表示を進捗バーの上に配置
        self.fs_time_label.adjustSize()
        self.fs_time_label.move(
            (self.width() - self.fs_time_label.width()) // 2,
            self.height() - bar_height - bar_margin - self.fs_time_label.height() - 10
        )
    
    def hide_fs_ui(self):
        """フルスクリーンUI非表示"""
        if self.is_fullscreen:
            self.fs_progress_bar.hide()
            self.fs_time_label.hide()
            QApplication.restoreOverrideCursor()
    
    def resizeEvent(self, event):
        """リサイズ時にUI位置調整"""
        super().resizeEvent(event)
        if self.is_fullscreen:
            self.update_fs_ui_position()
    
    def mouseMoveEvent(self, event):
        """マウス移動でUI表示"""
        if self.is_fullscreen:
            self.show_fs_ui()
        super().mouseMoveEvent(event)
    
    def fs_progress_click(self, event: QMouseEvent):
        """フルスクリーン進捗バークリック"""
        if self.total_frames > 0:
            pos = event.pos().x()
            bar_margin = 20
            bar_width = self.width() - bar_margin * 2
            relative_pos = pos - bar_margin
            
            if 0 <= relative_pos <= bar_width:
                target_frame = int((relative_pos / bar_width) * self.total_frames)
                self.seek_requested.emit(target_frame)
                print(f"[DEBUG] フルスクリーンシーク: {target_frame}")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """ドラッグ開始時のイベント"""
        if event.mimeData().hasUrls():
            # 動画ファイルかチェック
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_video_file(file_path):
                    event.acceptProposedAction()
                    print(f"[DEBUG] D&D: 動画ファイル検出 - {file_path}")
    
    def dragMoveEvent(self, event):
        """ドラッグ中のイベント"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_video_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """ドロップ時のイベント"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                print(f"[DEBUG] D&D: 動画ファイルをドロップ - {file_path}")
                self.video_dropped.emit(file_path)
                event.acceptProposedAction()
    
    def is_video_file(self, file_path):
        """動画ファイルかどうかを判定"""
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts']
        file_ext = Path(file_path).suffix.lower()
        return file_ext in video_extensions
    
    def keyPressEvent(self, event):
        if self.is_fullscreen:
            key = event.key()
            if key == Qt.Key.Key_F or key == Qt.Key.Key_Escape:
                self.toggle_fullscreen()
            elif key == Qt.Key.Key_Space or key == Qt.Key.Key_K:
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, QMainWindow) and hasattr(widget, 'toggle_playback'):
                        widget.toggle_playback()
                        break
            elif key == Qt.Key.Key_Right or key == Qt.Key.Key_L:
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, QMainWindow) and hasattr(widget, 'seek_relative'):
                        widget.seek_relative(300)
                        break
            elif key == Qt.Key.Key_Left or key == Qt.Key.Key_J:
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, QMainWindow) and hasattr(widget, 'seek_relative'):
                        widget.seek_relative(-300)
                        break
            elif key == Qt.Key.Key_Semicolon:
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, QMainWindow) and hasattr(widget, 'seek_relative'):
                        widget.seek_relative(30)
                        break
            elif key == Qt.Key.Key_H:
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, QMainWindow) and hasattr(widget, 'seek_relative'):
                        widget.seek_relative(-30)
                        break
        else:
            super().keyPressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """ダブルクリックでフルスクリーン切り替え - 再生状態を保持"""
        # 現在の再生状態を取得
        parent = self.window()
        if hasattr(parent, 'is_paused'):
            current_pause_state = parent.is_paused
        
        # フルスクリーン切り替え
        self.toggle_fullscreen()
        
        # フルスクリーン状態変更後、再生状態を元に戻す
        if hasattr(parent, 'is_paused') and hasattr(parent, 'process_thread'):
            # 少し待ってから状態を復元
            QTimer.singleShot(100, lambda: self.restore_playback_state(parent, current_pause_state))
    
    def restore_playback_state(self, parent, original_pause_state):
        """再生状態を復元"""
        if hasattr(parent, 'process_thread') and parent.process_thread and parent.process_thread.isRunning():
            if original_pause_state:
                # 元々一時停止中だった場合は一時停止を維持
                parent.process_thread.pause()
                parent.is_paused = True
                parent.play_pause_btn.setText("▶ 再開")
                parent.mode_label.setText("📊 モード: ⏸ 一時停止中")
                self.set_progress_bar_color('red')
            else:
                # 元々再生中だった場合は再生を継続
                parent.process_thread.resume()
                parent.is_paused = False
                parent.play_pause_btn.setText("⏸ 一時停止")
                parent.mode_label.setText("📊 モード: 🔄 AI処理中")
                self.set_progress_bar_color('#00ff00')
    
    def mousePressEvent(self, event):
        # フルスクリーン時の進捗バークリック判定
        if self.is_fullscreen and self.fs_progress_bar.isVisible():
            bar_geom = self.fs_progress_bar.geometry()
            if bar_geom.contains(event.pos()):
                self.fs_progress_click(event)
                return
        
        if event.button() == Qt.MouseButton.LeftButton:
            self.playback_toggled.emit()
        super().mousePressEvent(event)
    
    def toggle_fullscreen(self):
        if not self.is_fullscreen:
            self.parent_widget = self.parentWidget()
            parent_window = self.window()
            self.normal_parent_geometry = parent_window.geometry()
            
            self.setParent(None)
            self.setWindowFlags(
                Qt.WindowType.Window | 
                Qt.WindowType.FramelessWindowHint | 
                Qt.WindowType.WindowStaysOnTopHint
            )
            self.showFullScreen()
            self.setFocus(Qt.FocusReason.OtherFocusReason)
            self.activateWindow()
            self.raise_()
            self.is_fullscreen = True
            
            # フルスクリーン移行後にUI位置を更新
            QApplication.processEvents()  # ウィンドウサイズ確定を待つ
            self.update_fs_ui_position()
            
            # フルスクリーンUI表示
            self.show_fs_ui()
        else:
            # フルスクリーンUI非表示
            self.hide_fs_ui()
            self.ui_hide_timer.stop()
            
            if self.parent_widget:
                self.setParent(self.parent_widget)
                self.setWindowFlags(Qt.WindowType.Widget)
                
                parent_window = self.parent_widget.window()
                if hasattr(parent_window, 'video_layout'):
                    parent_window.video_layout.insertWidget(0, self)
                
                self.showNormal()
                
                if self.normal_parent_geometry and parent_window:
                    parent_window.setGeometry(self.normal_parent_geometry)
            
            self.is_fullscreen = False
            self.parent_widget = None
    
    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        if self.current_frame is not None and self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            w, h = self.width(), self.height()
            if h == 0:
                h = 1
            window_aspect = w / h
            
            if self.frame_width > 0 and self.frame_height > 0:
                video_aspect = self.frame_width / self.frame_height
            else:
                video_aspect = 16.0 / 9.0
            
            if window_aspect > video_aspect:
                x_scale = video_aspect / window_aspect
                x1, x2 = -x_scale, x_scale
                y1, y2 = -1.0, 1.0
            else:
                y_scale = window_aspect / video_aspect
                x1, x2 = -1.0, 1.0
                y1, y2 = -y_scale, y_scale
            
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 1.0); glVertex2f(x1, y1)
            glTexCoord2f(1.0, 1.0); glVertex2f(x2, y1)
            glTexCoord2f(1.0, 0.0); glVertex2f(x2, y2)
            glTexCoord2f(0.0, 0.0); glVertex2f(x1, y2)
            glEnd()
    
    def update_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.frame_width != w or self.frame_height != h:
            self.frame_width = w
            self.frame_height = h
            print(f"[DEBUG] フレーム解像度: {w}x{h}")
        
        self.makeCurrent()
        
        if self.texture_id is None:
            self.texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0)
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0)
        
        rgb_contiguous = np.ascontiguousarray(rgb)
        
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_RGB, 
            w, h, 
            0, 
            GL_RGB, 
            GL_UNSIGNED_BYTE, 
            rgb_contiguous.tobytes()
        )
        
        self.current_frame = rgb
        self.update()
    
    def clear_frame(self):
        self.current_frame = None
        self.frame_width = 0
        self.frame_height = 0
        self.update()

    def set_progress_bar_color(self, color):
        """進捗バーの色を設定"""
        self.fs_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(40, 40, 40, 200);
                border: none;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)


class OptimizedFrameRestorer:
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length,
                 mosaic_restoration_model_name, mosaic_detection_model, 
                 mosaic_restoration_model, preferred_pad_mode,
                 batch_size=16, queue_size_mb=12288, mosaic_detection=False):
        from lada.lib.frame_restorer import FrameRestorer
        import queue
        
        self._parent = FrameRestorer(
            device=device, video_file=video_file,
            preserve_relative_scale=preserve_relative_scale,
            max_clip_length=max_clip_length,
            mosaic_restoration_model_name=mosaic_restoration_model_name,
            mosaic_detection_model=mosaic_detection_model,
            mosaic_restoration_model=mosaic_restoration_model,
            preferred_pad_mode=preferred_pad_mode,
            mosaic_detection=mosaic_detection
        )
        
        w = self._parent.video_meta_data.video_width
        h = self._parent.video_meta_data.video_height
        
        # より効率的なキューサイズ計算
        frame_size_bytes = w * h * 3  # RGB
        clip_size_bytes = max_clip_length * 256 * 256 * 4  # 保守的な見積もり
        
        max_frames = max(100, (queue_size_mb * 1024 * 1024) // frame_size_bytes)
        max_clips = max(10, (queue_size_mb * 1024 * 1024) // clip_size_bytes)
        
        # 非同期処理用のキュー改善
        self._parent.frame_restoration_queue = queue.Queue(maxsize=max_frames)
        self._parent.mosaic_clip_queue = queue.Queue(maxsize=max_clips)
        self._parent.restored_clip_queue = queue.Queue(maxsize=max_clips)
        self._parent.mosaic_detector.mosaic_clip_queue = self._parent.mosaic_clip_queue
        
        # バッチ処理の最適化
        self._parent.batch_size = min(batch_size, max_clip_length)
        
        print(f"[OPTIMIZE] Queue: {max_frames}f, {max_clips}c ({queue_size_mb}MB)")
        print(f"[OPTIMIZE] Batch size: {self._parent.batch_size}")
    
    def start(self, start_ns=0):
        return self._parent.start(start_ns)
    
    def stop(self):
        return self._parent.stop()
    
    def __iter__(self):
        return self._parent.__iter__()
    
    def __next__(self):
        return self._parent.__next__()


class ProcessThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int, bool)
    fps_updated = pyqtSignal(float)
    progress_updated = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    
    def __init__(self, video_path, detection_path, restoration_path, frame_cache, start_frame, thread_id, settings, audio_thread=None, video_fps=30.0):
        super().__init__()
        self.video_path = Path(video_path)
        self.detection_path = Path(detection_path)
        self.restoration_path = Path(restoration_path)
        self.frame_cache = frame_cache
        self.start_frame = start_frame
        self.thread_id = thread_id
        
        self.batch_size = settings.get('batch_size', 16)
        self.queue_size_mb = settings.get('queue_size_mb', 12288)
        self.max_clip_length = settings.get('max_clip_length', 8)
        
        self.frame_restorer = None
        self.is_running = False
        self._stop_flag = False
        self.is_paused = False
        self.pause_mutex = QMutex()
        
        self.audio_thread = audio_thread  # 追加: AudioThreadへの参照
        self.video_fps = video_fps        # 追加: 音声同期用
        self.total_frames = 0             # 追加: 音声同期用
    
    def pause(self):
        with QMutexLocker(self.pause_mutex):
            self.is_paused = True
            if self.audio_thread:
                self.audio_thread.pause_audio() # 音声一時停止
    
    def resume(self):
        with QMutexLocker(self.pause_mutex):
            self.is_paused = False
            if self.audio_thread:
                # フレーム数から秒数を計算し、音声再生を再開
                start_sec = self.start_frame / self.video_fps if self.video_fps > 0 else 0
                self.audio_thread.resume_audio(start_sec)
    
    def run(self):
        print(f"[DEBUG] スレッド{self.thread_id}開始:")
        print(f"  batch_size={self.batch_size}")
        print(f"  queue_size_mb={self.queue_size_mb}")
        print(f"  max_clip_length={self.max_clip_length}")
        
        self.is_running = True
        self._stop_flag = False
        
        try:
            if not LADA_AVAILABLE:
                return
            
            video_meta = video_utils.get_video_meta_data(self.video_path)
            self.total_frames = video_meta.frames_count
            self.video_fps = video_meta.video_fps
            
            print(f"[DEBUG] 動画情報: {self.total_frames}フレーム, {self.video_fps}FPS")
            
            if self._stop_flag:
                return
            
            detection_model, restoration_model, pad_mode = load_models(
                device="cuda:0",
                mosaic_restoration_model_name="basicvsrpp-v1.2",
                mosaic_restoration_model_path=str(self.restoration_path),
                mosaic_restoration_config_path=None,
                mosaic_detection_model_path=str(self.detection_path)
            )
            
            if self._stop_flag:
                return
            
            self.frame_restorer = OptimizedFrameRestorer(
                device="cuda:0",
                video_file=self.video_path,
                preserve_relative_scale=True,
                max_clip_length=self.max_clip_length,
                mosaic_restoration_model_name="basicvsrpp-v1.2",
                mosaic_detection_model=detection_model,
                mosaic_restoration_model=restoration_model,
                preferred_pad_mode=pad_mode,
                batch_size=self.batch_size,
                queue_size_mb=self.queue_size_mb,
                mosaic_detection=False
            )
            
            start_ns = int((self.start_frame / self.video_fps) * 1_000_000_000)
            self.frame_restorer.start(start_ns=start_ns)
            
            frame_count = self.start_frame
            start_time = time.time()
            pause_start_time = 0
            total_pause_duration = 0
            frame_interval = 1.0 / self.video_fps
            
            frame_restorer_iter = iter(self.frame_restorer)
            pending_ai_frame = None
            lada_start = time.time()
            lada_time = 0
            last_mode_was_cached = False
            frame_count_at_reset = self.start_frame
            
            # 音声再生開始 (スタートフレームの位置から)
            if self.audio_thread:
                start_sec = self.start_frame / self.video_fps if self.video_fps > 0 else 0
                self.audio_thread.start_playback(str(self.video_path), start_sec)
                
            # 一時停止中のキャッシュ蓄積設定
            cache_frames_during_pause = 1800  # 一時停止中に蓄積するフレーム数(約30秒分@60fps)
            paused_cache_count = 0
            
            # フレーム処理の最適化
            consecutive_cached_frames = 0
            max_consecutive_cached = 30  # 連続キャッシュフレーム数の上限
            
            while self.is_running and not self._stop_flag and frame_count < self.total_frames:
                if self.is_paused and not self._stop_flag:
                    if pause_start_time == 0:
                        pause_start_time = time.time()
                        paused_cache_count = 0
                        print(f"[DEBUG] 一時停止開始 - バックグラウンドキャッシュ蓄積中(目標:{cache_frames_during_pause}フレーム)")
                    
                    # 一時停止中もキャッシュ蓄積を継続(上限まで)
                    if paused_cache_count < cache_frames_during_pause:
                        # キャッシュ未登録のフレームを先行取得
                        if self.frame_cache.get(frame_count + paused_cache_count) is None:
                            try:
                                item = next(frame_restorer_iter)
                                if item is not None:
                                    restored_frame, _ = item
                                    self.frame_cache.put(frame_count + paused_cache_count, restored_frame)
                                    paused_cache_count += 1
                                    
                                    if paused_cache_count % 30 == 0:
                                        print(f"[DEBUG] 一時停止中キャッシュ: {paused_cache_count}/{cache_frames_during_pause}フレーム蓄積")
                            except StopIteration:
                                print(f"[DEBUG] 一時停止中キャッシュ完了: {paused_cache_count}フレーム蓄積済み")
                                break
                        else:
                            paused_cache_count += 1
                    else:
                        # キャッシュ上限到達
                        if paused_cache_count == cache_frames_during_pause:
                            print(f"[DEBUG] 一時停止中キャッシュ完了: {cache_frames_during_pause}フレーム蓄積済み - 待機モード")
                            paused_cache_count += 1  # フラグ用に1増やす
                    
                    time.sleep(0.01)  # 短い待機でレスポンス向上
                    continue
                
                if pause_start_time > 0:
                    pause_duration = time.time() - pause_start_time
                    total_pause_duration += pause_duration
                    print(f"[DEBUG] 一時停止解除 - 停止時間: {pause_duration:.1f}秒, キャッシュ活用モードで再開")
                    pause_start_time = 0
                    paused_cache_count = 0
                
                if self._stop_flag:
                    break
                
                # キャッシュ優先チェック(再開直後は特にキャッシュ活用)
                cached_frame = self.frame_cache.get(frame_count)
                
                if cached_frame is not None:
                    final_frame = cached_frame
                    is_cached = True
                    consecutive_cached_frames += 1
                    
                    # 連続キャッシュが多すぎる場合はAI処理を促進
                    if consecutive_cached_frames > max_consecutive_cached:
                        # 次のフレームをAI処理で強制更新
                        self.frame_cache.put(frame_count, None)  # キャッシュを無効化
                        cached_frame = None
                        consecutive_cached_frames = 0
                    
                    if not last_mode_was_cached:
                        start_time = time.time()
                        total_pause_duration = 0
                        frame_count_at_reset = frame_count
                        print("[DEBUG] キャッシュモード開始 - スムーズ再生")
                    
                else:
                    consecutive_cached_frames = 0
                    
                    if last_mode_was_cached:
                        start_time = time.time()
                        total_pause_duration = 0
                        frame_count_at_reset = frame_count
                        print("[DEBUG] AI処理モード切替")
                    
                    if pending_ai_frame is not None:
                        restored_frame, frame_pts = pending_ai_frame
                        pending_ai_frame = None
                    else:
                        try:
                            item = next(frame_restorer_iter)
                            lada_time = time.time() - lada_start
                            
                            if item is None:
                                break
                            
                            restored_frame, frame_pts = item
                            lada_start = time.time()
                            
                        except StopIteration:
                            break
                    
                    final_frame = restored_frame
                    self.frame_cache.put(frame_count, restored_frame)
                    is_cached = False
                
                last_mode_was_cached = is_cached
                
                frames_since_reset = frame_count - frame_count_at_reset
                target_time = frames_since_reset * frame_interval
                elapsed = time.time() - start_time - total_pause_duration
                wait_time = target_time - elapsed
                
                # 負のwait_time（遅延）が大きすぎる場合はスキップ
                if wait_time < -0.5:  # 500ms以上遅れている場合
                    print(f"[WARNING] フレーム遅延検出: {wait_time:.3f}s, スキップ調整")
                    start_time = time.time() - (frames_since_reset * frame_interval)
                    total_pause_duration = 0
                    wait_time = 0
                
                if wait_time > 0:
                    time.sleep(min(wait_time, 0.1))  # 最大100msまでスリープ
                
                self.frame_ready.emit(final_frame, frame_count, is_cached)
                
                # 音声同期処理: 10秒に一度同期を試みる
                if self.audio_thread and frame_count % (int(self.video_fps) * 10) == 0:
                    current_sec = frame_count / self.video_fps
                    self.audio_thread.seek_to_time(current_sec)
                
                frame_count += 1
                self.progress_updated.emit(frame_count, self.total_frames)
                
                # FPS表示の頻度を調整 (15フレームごとに更新)
                if frame_count % 15 == 0:
                    elapsed = time.time() - start_time - total_pause_duration
                    actual_fps = (frame_count - self.start_frame) / elapsed if elapsed > 0 else 0
                    self.fps_updated.emit(actual_fps)
                    
                    cache_status = "キャッシュ" if is_cached else "AI処理"
                    print(f"[DEBUG] FPS: {actual_fps:.1f}, モード: {cache_status}")
                    if not is_cached and lada_time > 0:
                        print(f"[DEBUG] LADA処理時間: {lada_time:.3f}秒/15フレーム")
            
            if not self._stop_flag:
                self.finished_signal.emit()
            
        except Exception as e:
            print(f"AI処理エラー (thread {self.thread_id}): {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.frame_restorer:
                try:
                    self.frame_restorer.stop()
                except:
                    pass
            self.is_running = False
            if self.audio_thread:
                self.audio_thread.stop_playback() # スレッド終了時に音声停止
    
    def stop(self):
        self._stop_flag = True
        self.is_running = False
        self.resume()
        if self.frame_restorer:
            try:
                self.frame_restorer.stop()
            except:
                pass


class AudioThread(QThread):
    """VLCを使用した音声再生を制御するスレッド"""
    
    def __init__(self, vlc_instance, initial_volume=100, is_muted=False):
        super().__init__()
        self.vlc_instance = vlc_instance
        self.player = self.vlc_instance.media_player_new()
        self._stop_flag = False
        self._is_paused = True
        self.volume = initial_volume
        self.user_muted = is_muted      # ユーザーによるミュート状態
        self.internal_muted = False     # 内部都合によるミュート状態 (新規追加)
        
        self.player.audio_set_volume(self.volume)
        self._update_vlc_mute_state()   # 新しいロジックで初期設定を適用
        
        print(f"[DEBUG] AudioThread初期化: Volume={self.volume}, Mute={self.user_muted} (Internal:{self.internal_muted})")

    def run(self):
        # メインスレッドで制御を行うため、このスレッドは基本的に待機
        while not self._stop_flag:
            time.sleep(0.1)

    def _update_vlc_mute_state(self):
        """ユーザーミュートと内部ミュートの論理和に基づいてVLCのミュート状態を更新する"""
        if not VLC_AVAILABLE:
            return
        # ユーザーミュート OR 内部ミュート のいずれかがTrueならミュート
        should_be_muted = self.user_muted or self.internal_muted
        self.player.audio_set_mute(should_be_muted)

    def set_internal_mute(self, is_muted):
        """内部ミュートの設定"""
        if not VLC_AVAILABLE:
            return
        self.internal_muted = is_muted
        self._update_vlc_mute_state() 
        print(f"[DEBUG] 内部ミュート設定: {is_muted}")
        
    def start_playback(self, video_path, start_sec=0.0):
        if not VLC_AVAILABLE:
            return
            
        media = self.vlc_instance.media_new(video_path)
        self.player.set_media(media)
        
        msec = int(start_sec * 1000)
        
        print(f"[DEBUG] 音声再生開始: {video_path} から {start_sec:.2f}秒")
        
        # 1. 内部ミュートをONにする (音声漏れ防止)
        self.set_internal_mute(True) 
        
        # 2. play()でストリームを初期化 (再生開始)
        self.player.play()
        
        # 3. VLCの起動を待つ
        time.sleep(0.01) 
        
        if start_sec > 0.0:
            # 4. プレイヤーが再生状態になるのを待つ
            for _ in range(10): 
                if self.player.get_state() in (vlc.State.Playing, vlc.State.Paused):
                    break
                time.sleep(0.05)
            
            # 5. 正しい位置にシーク
            if self.player.is_seekable():
                self.player.set_time(msec)
                print(f"[DEBUG] 音声シーク(初期): {msec}ms")

        # 6. 内部ミュートをOFFにする (シーク完了後のタイミング)
        self.set_internal_mute(False) 
        
        self._is_paused = False
        
    def stop_playback(self):
        if not VLC_AVAILABLE:
            return
            
        print("[DEBUG] 音声再生停止")
        self.player.stop()
        self._is_paused = True

    def pause_audio(self):
        if not VLC_AVAILABLE or self._is_paused:
            return
            
        print("[DEBUG] 音声一時停止")
        self.player.pause()
        self._is_paused = True
    
    def resume_audio(self, start_sec):
        if not VLC_AVAILABLE or not self._is_paused:
            return
            
        print(f"[DEBUG] 音声再生再開: {start_sec:.2f}秒へシーク")
        self.seek_to_time(start_sec)
        self.player.play() # play()は一時停止状態から再開する
        self._is_paused = False
        # ユーザーミュート設定を再適用
        self._update_vlc_mute_state() # 内部ミュート状態を考慮して更新

    def seek_to_time(self, seconds):
        if not VLC_AVAILABLE:
            return
            
        msec = int(seconds * 1000)
        
        # 1. 内部ミュートON (シーク中の音声漏れ防止)
        self.set_internal_mute(True)
        
        # 再生可能状態になるまで待つ（重要）
        for _ in range(10): 
            if self.player.get_state() in (vlc.State.Playing, vlc.State.Paused):
                break
            time.sleep(0.1)

        if self.player.is_seekable():
            self.player.set_time(msec)
            print(f"[DEBUG] 音声シーク: {msec}ms")
        
        # 2. 内部ミュートOFF
        self.set_internal_mute(False)

    def set_volume(self, volume):
        if not VLC_AVAILABLE:
            return
        self.volume = max(0, min(100, volume))
        self.player.audio_set_volume(self.volume)
        print(f"[DEBUG] 音量設定: {self.volume}")

    def toggle_mute(self, is_muted):
        """ユーザーによるミュート切り替え"""
        if not VLC_AVAILABLE:
            return
        self.user_muted = is_muted
        self._update_vlc_mute_state() # 内部ミュート状態を考慮して更新
        print(f"[DEBUG] ユーザーミュート設定: {is_muted}")

    def stop(self):
        self._stop_flag = True
        self.stop_playback()

class LadaFinalPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # D&Dを有効化
        self.setAcceptDrops(True)
        
        self.settings = self.load_settings()
        self.frame_cache = FrameCache(max_size_mb=self.settings.get('cache_size_mb', 12288))
        self.process_thread = None
        self.current_video = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 30.0
        self.is_playing = False
        self.is_paused = False
        self.thread_counter = 0
        self._seeking = False
        
        # VLCの初期化
        self.vlc_instance = vlc.Instance('--no-video') if VLC_AVAILABLE else None
        self.audio_thread = None
        if VLC_AVAILABLE:
            initial_volume = self.settings.get('audio_volume', 100)
            initial_mute = self.settings.get('audio_muted', False)
            
            if isinstance(initial_volume, float):
                initial_volume = int(initial_volume * 100)
            initial_volume = max(0, min(100, initial_volume))
            
            self.audio_thread = AudioThread(self.vlc_instance, initial_volume, initial_mute)
            self.audio_thread.start()
        
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("LADA REALTIME PLAYER V1.0")
        self.setGeometry(100, 100, 1200, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.filename_label = QLabel("")
        self.filename_label.setStyleSheet("""
            QLabel {
                background-color: rgba(40, 40, 40, 220);
                color: #00ff00;
                padding: 2px 15px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 3px;
            }
        """)
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.filename_label.setFixedHeight(24)
        self.filename_label.hide()
        layout.addWidget(self.filename_label)
        
        self.video_layout = QVBoxLayout()
        self.video_widget = VideoGLWidget()
        self.video_widget.playback_toggled.connect(self.toggle_playback)
        self.video_widget.video_dropped.connect(self.load_video)
        self.video_widget.seek_requested.connect(self.seek_to_frame)
        self.video_layout.addWidget(self.video_widget)
        layout.addLayout(self.video_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.mousePressEvent = self.seek_click
        layout.addWidget(self.progress_bar)
        
        # 再生時間表示とボリュームコントロールのレイアウト
        time_audio_layout = QHBoxLayout()
        
        # 再生時間表示 (センター寄せ)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("font-size: 12px; color: #aaa;")
        self.time_label.setMaximumHeight(20)
        
        # ボリュームコントロール (右寄せ)
        self.mute_btn = QPushButton("🔇")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(self.settings.get('audio_muted', False))
        self.mute_btn.setFixedWidth(40)
        self.mute_btn.clicked.connect(self.toggle_user_mute)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        # 修正: 設定ファイルの値がfloat (例: 0.7) の場合に 0-100 の整数値に変換してUIに反映
        initial_volume_ui = self.settings.get('audio_volume', 100)
        if isinstance(initial_volume_ui, float):
             initial_volume_ui = int(initial_volume_ui * 100)
        self.volume_slider.setValue(max(0, min(100, initial_volume_ui)))
        
        self.volume_slider.setFixedWidth(150)
        self.volume_slider.valueChanged.connect(self.set_volume_slider)
        self.volume_slider.sliderReleased.connect(self.save_audio_settings)
        
        time_audio_layout.addStretch(1)
        time_audio_layout.addWidget(self.time_label)
        time_audio_layout.addStretch(1)
        
        if VLC_AVAILABLE:
            time_audio_layout.addWidget(self.mute_btn)
            time_audio_layout.addWidget(self.volume_slider)
        
        layout.addLayout(time_audio_layout)
        
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("動画を開く")
        self.open_btn.clicked.connect(self.open_video)
        
        self.play_pause_btn = QPushButton("⏸ 一時停止")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setEnabled(False)
        
        self.settings_btn = QPushButton("⚙️ 設定")
        self.settings_btn.clicked.connect(self.open_settings)
        
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.play_pause_btn)
        btn_layout.addWidget(self.settings_btn)
        layout.addLayout(btn_layout)
        
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("⚡ FPS: --")
        self.mode_label = QLabel("📊 モード: 待機中")
        self.cache_label = QLabel("💾 キャッシュ: 0 MB")
        
        for label in [self.fps_label, self.mode_label, self.cache_label]:
            label.setMaximumHeight(20)
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.mode_label)
        stats_layout.addWidget(self.cache_label)
        layout.addLayout(stats_layout)
        
        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(100)
        info.setText("""
V1.0 - 30FPS最適化: フルスクリーン進捗バー + 設定機能 + 再生速度制御 + 音声機能追加
操作: F=全画面 | ESC=通常 | Space=再生/停止 | ⚙️設定で性能調整 | 進捗バークリックでシーク
最適化: バッチ16, キュー12GB, クリップ長8 で30FPS目標
""")
        layout.addWidget(info)
        
        self.setup_shortcuts()
        print("[INFO] 初期化完了 - 30FPS最適化版")
        
        # UI初期化後に音声設定をAudioThreadに適用
        if self.audio_thread:
            # AudioThreadに渡すのは0-100の整数値
            initial_volume_thread = self.settings.get('audio_volume', 100)
            if isinstance(initial_volume_thread, float):
                 initial_volume_thread = int(initial_volume_thread * 100)
            initial_volume_thread = max(0, min(100, initial_volume_thread))
            
            self.audio_thread.set_volume(initial_volume_thread)
            self.audio_thread.toggle_mute(self.settings.get('audio_muted', False))
            self.mute_btn.setText("🔇" if self.settings.get('audio_muted', False) else "🔊")
    
    def setup_shortcuts(self):
        self.shortcut_fullscreen = QShortcut(QKeySequence('F'), self)
        self.shortcut_fullscreen.activated.connect(self.toggle_fullscreen_shortcut)
        
        self.shortcut_escape = QShortcut(QKeySequence('Esc'), self)
        self.shortcut_escape.activated.connect(self.escape_fullscreen_shortcut)
        
        self.shortcut_space = QShortcut(QKeySequence('Space'), self)
        self.shortcut_space.activated.connect(self.toggle_playback)

        self.shortcut_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.seek_relative(300))
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.seek_relative(-300))
        self.shortcut_semicolon = QShortcut(QKeySequence(Qt.Key.Key_Semicolon), self)
        self.shortcut_semicolon.activated.connect(lambda: self.seek_relative(30))
        self.shortcut_h = QShortcut(QKeySequence('H'), self)
        self.shortcut_h.activated.connect(lambda: self.seek_relative(-30))
        self.shortcut_l = QShortcut(QKeySequence('L'), self)
        self.shortcut_l.activated.connect(lambda: self.seek_relative(300))
        self.shortcut_j = QShortcut(QKeySequence('J'), self)
        self.shortcut_j.activated.connect(lambda: self.seek_relative(-300))
        self.shortcut_k = QShortcut(QKeySequence('K'), self)
        self.shortcut_k.activated.connect(self.toggle_playback)
        
        print("[INFO] ショートカット設定完了")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """メインウィンドウでもD&Dをサポート"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_video_file(file_path):
                    event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_video_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                print(f"[DEBUG] メインウィンドウD&D: {file_path}")
                self.load_video(file_path)
                event.acceptProposedAction()
    
    def is_video_file(self, file_path):
        """動画ファイルかどうかを判定"""
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts']
        file_ext = Path(file_path).suffix.lower()
        return file_ext in video_extensions
    
    def load_settings(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    settings = json.load(f)
                    print(f"[INFO] 設定読み込み: {settings}")
                    return settings
            except:
                pass
        
        # 30FPS達成に向けた最適化設定
        return {
            'batch_size': 16,  # 16フレームに調整
            'queue_size_mb': 12288,  # 12GBに増加
            'max_clip_length': 8,    # クリップ長を短く
            'cache_size_mb': 12288,  # 12GBに増加
            'audio_volume': 100, 
            'audio_muted': False
        }
    
    def save_settings(self):
        # 音声設定も保存
        if self.audio_thread:
            # 0-100 の整数で保存
            self.settings['audio_volume'] = self.audio_thread.volume
            self.settings['audio_muted'] = self.audio_thread.user_muted
            
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"[INFO] 設定保存: {self.settings}")
        except Exception as e:
            print(f"[ERROR] 設定保存失敗: {e}")
            
    def save_audio_settings(self):
        """ボリューム操作完了後に設定を保存"""
        self.save_settings()
    
    def set_volume_slider(self, value):
        """ボリュームスライダー操作時"""
        if self.audio_thread:
            self.audio_thread.set_volume(value)
            # ユーザーミュートがONの場合、スライダーを動かしてもミュート状態を維持
            if self.audio_thread.user_muted:
                self.mute_btn.setText("🔇")
                self.mute_btn.setChecked(True)
            else:
                self.mute_btn.setText("🔊" if value > 0 else "🔇")
                self.mute_btn.setChecked(value == 0) # 音量0はミュート扱いとする
    
    def toggle_user_mute(self, checked):
        """ユーザーによるミュート切り替え"""
        if self.audio_thread:
            self.audio_thread.toggle_mute(checked)
            self.mute_btn.setText("🔇" if checked else "🔊")
            
            # ミュートした場合、ボリュームスライダーを0に
            if checked:
                self.volume_slider.setValue(0)
            else:
                # ミュート解除した場合、ミュート前のボリューム（設定に保存されている値）に戻す
                unmuted_volume = self.settings.get('audio_volume', 100)
                if isinstance(unmuted_volume, float):
                     unmuted_volume = int(unmuted_volume * 100)
                unmuted_volume = max(1, min(100, unmuted_volume)) # 0に戻らないように最低値1
                
                self.volume_slider.setValue(unmuted_volume)
                self.audio_thread.set_volume(unmuted_volume) # VLCに設定
            
            self.save_audio_settings() # 保存
    
    def open_settings(self):
        dialog = SettingsDialog(self, self.settings)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()
            
            # 音声設定はここで保存しない (volume, muteの状態は別の操作で保存するため)
            
            if (new_settings.get('batch_size') != self.settings.get('batch_size') or 
                new_settings.get('queue_size_mb') != self.settings.get('queue_size_mb') or 
                new_settings.get('max_clip_length') != self.settings.get('max_clip_length') or
                new_settings.get('cache_size_mb') != self.settings.get('cache_size_mb')):
                
                self.settings.update(new_settings)
                self.save_settings() # AI処理設定のみを更新し、保存

                print("[INFO] 設定変更 - 完全リセット実行")
                self.full_stop()
                self.frame_cache = FrameCache(max_size_mb=self.settings['cache_size_mb'])
                
                if self.current_video:
                    self.load_video(self.current_video)
                
                print(f"[INFO] 新設定適用完了: {self.settings}")
            else:
                # AI処理設定に変更がない場合でも、念のため保存処理を呼び出し
                self.settings.update(new_settings)
                self.save_settings()
    
    def toggle_fullscreen_shortcut(self):
        self.video_widget.toggle_fullscreen()
    
    def escape_fullscreen_shortcut(self):
        if self.video_widget.is_fullscreen:
            self.video_widget.toggle_fullscreen()
    
    def seek_click(self, event):
        if self.total_frames > 0:
            pos = event.pos().x()
            width = self.progress_bar.width()
            target_frame = int((pos / width) * self.total_frames)
            self.seek_to_frame(target_frame)
    
    def seek_to_frame(self, target_frame):
        if not self.current_video or self._seeking or self.total_frames == 0:
            return
        
        self._seeking = True
        self.full_stop()
        QApplication.processEvents()
        time.sleep(0.1)
        
        self.current_frame = target_frame
        self.progress_bar.setValue(target_frame)
        self.video_widget.update_progress(target_frame)
        
        frame_data = self.frame_cache.get(target_frame)
        if frame_data is not None:
            self.video_widget.update_frame(frame_data)
        
        # 音声シーク
        if self.audio_thread:
            target_sec = target_frame / self.video_fps if self.video_fps > 0 else 0
            self.audio_thread.seek_to_time(target_sec)
        
        self.start_processing_from_frame(target_frame)
        
        # seek_to_frameが一時停止状態で行われた場合、再生を再開しない
        # ProcessThreadのstart_processing_from_frame内で再生が開始される
        
        self._seeking = False
    
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画選択", "", "Videos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)"
        )
        if path:
            self.load_video(path)
    
    def load_video(self, path):
        print(f"[INFO] 動画読み込み: {path}")
        self.full_stop()
        self.frame_cache.clear()
        self.video_widget.clear_frame()
        
        self.current_video = path
        
        fullpath = str(Path(path).resolve())
        max_length = 100
        if len(fullpath) > max_length:
            fullpath = "..." + fullpath[-(max_length-3):]
        self.filename_label.setText(f"🎬 {fullpath}")
        self.filename_label.show()
        
        if LADA_AVAILABLE:
            try:
                video_meta = video_utils.get_video_meta_data(path)
                self.total_frames = video_meta.frames_count
                self.video_fps = video_meta.video_fps
                self.progress_bar.setMaximum(self.total_frames)
                
                # VideoGLWidgetに動画情報を設定
                self.video_widget.set_video_info(self.total_frames, self.video_fps)
            except Exception as e:
                print(f"[ERROR] 動画メタデータ取得失敗: {e}")
                self.total_frames = 0
                self.video_fps = 30.0
                pass
        
        self.start_processing_from_frame(0)
        self.mode_label.setText(f"📊 選択: {Path(path).name}")
    
    def start_processing(self):
        self.start_processing_from_frame(0)
    
    def start_processing_from_frame(self, start_frame):
        if not self.current_video or not LADA_AVAILABLE:
            return
        
        if self.process_thread and self.process_thread.isRunning():
            return
        
        model_dir = LADA_BASE_PATH / "model_weights"
        detection_path = model_dir / "lada_mosaic_detection_model_v3.1_fast.pt"
        restoration_path = model_dir / "lada_mosaic_restoration_model_generic_v1.2.pth"
        
        if not detection_path.exists() or not restoration_path.exists():
            self.mode_label.setText("エラー: モデルなし")
            return
        
        self.thread_counter += 1
        current_id = self.thread_counter
        
        self.process_thread = ProcessThread(
            self.current_video, detection_path, restoration_path,
            self.frame_cache, start_frame, current_id, self.settings,
            audio_thread=self.audio_thread, video_fps=self.video_fps # AudioThreadとFPSを渡す
        )
        
        self.process_thread.frame_ready.connect(
            lambda frame, num, cached: self.on_frame_ready(frame, num, cached, current_id)
        )
        self.process_thread.fps_updated.connect(
            lambda fps: self.fps_label.setText(f"⚡ FPS: {fps:.1f}")
        )
        self.process_thread.progress_updated.connect(
            lambda c, t: self.on_progress_update(c, t)
        )
        self.process_thread.finished_signal.connect(self.on_processing_finished)
        
        self.process_thread.start()
        self.is_playing = True
        self.is_paused = False
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸ 一時停止")
        self.mode_label.setText("📊 モード: 🔄 AI処理中")
        # 再生開始時は緑色
        self.video_widget.set_progress_bar_color('#00ff00')
    
    def on_processing_finished(self):
        """処理完了時の後処理"""
        print("[INFO] AI処理が完了しました")
        self.full_stop()
        self.mode_label.setText("📊 モード: 完了")
    
    def on_progress_update(self, current, total):
        """進捗更新 - 標準画面とフルスクリーン両方"""
        self.current_frame = current
        self.progress_bar.setValue(current)
        self.video_widget.update_progress(current)
    
    def on_frame_ready(self, frame, frame_num, is_cached, thread_id):
        if self.process_thread and thread_id == self.process_thread.thread_id:
            self.current_frame = frame_num
            self.video_widget.update_frame(frame)
            
            current_sec = frame_num / self.video_fps if self.video_fps > 0 else 0
            total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            
            current_time = self.format_time(current_sec)
            total_time = self.format_time(total_sec)
            self.time_label.setText(f"{current_time} / {total_time}")
            
            if is_cached:
                self.mode_label.setText("📊 モード: 💾 キャッシュ再生")
                # キャッシュ再生中は黄色
                if not self.is_paused:
                    self.video_widget.set_progress_bar_color('yellow')
            else:
                self.mode_label.setText("📊 モード: 🔄 AI処理中")
                # AI処理中は緑色
                if not self.is_paused:
                    self.video_widget.set_progress_bar_color('#00ff00')
    
    def format_time(self, seconds):
        """秒を HH:MM:SS 形式に変換"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def toggle_playback(self):
        if not self.process_thread or not self.process_thread.isRunning():
            if self.current_video:
                self.start_processing_from_frame(self.current_frame)
            return
        
        if self.is_paused:
            # 再開
            self.process_thread.resume()
            # AudioThreadのresumeはProcessThread内で呼ばれる
            self.is_paused = False
            self.play_pause_btn.setText("⏸ 一時停止")
            self.mode_label.setText("📊 モード: 🔄 AI処理中")
            # 再生再開時は緑色
            self.video_widget.set_progress_bar_color('#00ff00')
        else:
            # 一時停止
            self.process_thread.pause()
            # AudioThreadのpauseはProcessThread内で呼ばれる
            self.is_paused = True
            self.play_pause_btn.setText("▶ 再開")
            self.mode_label.setText("📊 モード: ⏸ 一時停止中")
            # 一時停止時は赤色
            self.video_widget.set_progress_bar_color('red')
    
    def full_stop(self):
        # AI処理スレッドの停止
        if self.process_thread:
            print("[DEBUG] AI処理スレッド停止中...")
            self.process_thread.stop()
            self.process_thread.wait(10000)
            if self.process_thread and self.process_thread.isRunning():
                print("[ERROR] AI処理スレッド強制終了")
                self.process_thread.terminate()
                self.process_thread.wait(2000)
            
            try:
                self.process_thread.frame_ready.disconnect()
                self.process_thread.fps_updated.disconnect()
                self.process_thread.progress_updated.disconnect()
                self.process_thread.finished_signal.disconnect()
            except:
                pass
            
            self.process_thread = None
        
        # 音声再生の停止
        if self.audio_thread:
            self.audio_thread.stop_playback()
            
        self.is_paused = False
        self.play_pause_btn.setText("▶ 再開")
        self.play_pause_btn.setEnabled(self.current_video is not None)
        QApplication.processEvents()
        time.sleep(0.05)
    
    def update_stats(self):
        stats = self.frame_cache.get_stats()
        self.cache_label.setText(f"💾 キャッシュ: {stats['size_mb']:.1f} MB ({stats['count']} frames)")
    
    def closeEvent(self, event):
        print("=== 終了処理 ===")
        self.full_stop()
        
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait(5000)
            if self.audio_thread.isRunning():
                 self.audio_thread.terminate()
        
        if hasattr(self, 'video_widget') and self.video_widget.texture_id:
            try:
                self.video_widget.makeCurrent()
                glDeleteTextures([self.video_widget.texture_id])
            except:
                pass
        
        self.frame_cache.clear()
        self.save_settings() # 終了時に設定を保存
        event.accept()

    def seek_relative(self, delta):
        if self.total_frames == 0 or not self.current_video:
            return
        target_frame = max(0, min(self.current_frame + delta, self.total_frames - 1))
        
        # シーク前の状態を保持
        was_paused = self.is_paused
        
        self.seek_to_frame(target_frame)
        
        # シーク後に一時停止状態だった場合は、再生を再開しない
        if was_paused:
             if self.process_thread:
                 self.process_thread.pause()
                 self.is_paused = True
                 self.play_pause_btn.setText("▶ 再開")
                 self.mode_label.setText("📊 モード: ⏸ 一時停止中")
                 self.video_widget.set_progress_bar_color('red')


def main():
    app = QApplication(sys.argv)
    player = LadaFinalPlayer()
    player.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()