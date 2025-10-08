#!/usr/bin/env python3
"""
LADA REALTIME PLAYER V1.0 - Smart Cache Edition - デッドロック修正版
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time
import json
import gc
from collections import OrderedDict, deque
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QDialog, QSpinBox, QFormLayout, QDialogButtonBox, QSlider, QSizePolicy, QMessageBox
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
        
        self.chunk_frames_spin = QSpinBox()
        self.chunk_frames_spin.setRange(30, 450)  # 1秒〜15秒 (30fps想定)
        self.chunk_frames_spin.setValue(self.settings.get('chunk_frames', 150))
        self.chunk_frames_spin.setSuffix(" frames")
        self.chunk_frames_spin.setToolTip("チャンクあたりのフレーム数 (推奨: 150 = 5秒@30fps)")
        layout.addRow("チャンクサイズ:", self.chunk_frames_spin)
        
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
            'cache_size_mb': self.cache_size_spin.value(),
            'chunk_frames': self.chunk_frames_spin.value()
        }


class SmartChunkBasedCache:
    """30FPS最適化スマートキャッシュ（デッドロック対策版）"""
    
    def __init__(self, max_size_mb=12288, chunk_frames=150):
        self.chunk_frames = chunk_frames
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0
        
        # チャンク管理
        self.chunks = {}  # chunk_id -> {'frames': dict, 'size_mb': float, 'last_access': float}
        self.access_order = deque()  # LRU順序
        self.mutex = QMutex()  # 通常のミューテックス
        
        # 処理コスト追跡
        self.processing_costs = {}  # chunk_id -> cost_data
        self.cache_policies = {}    # chunk_id -> policy_dict
        
        # パフォーマンス統計
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_frames': 0,
            'total_processing_time': 0.0
        }
        
        # 予測的先読み
        self.prefetch_queue = deque()
        self.prefetch_enabled = True
        
        # デバッグ制御
        self.debug_enabled = False
        self.last_debug_output = 0
        self.debug_interval = 5.0  # 5秒ごとにデバッグ出力
        
        # 非同期クリーンアップ
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._async_cleanup)
        self.cleanup_timer.setSingleShot(True)
        self.pending_cleanup = False
        
        # 再生状態
        self.current_playhead = 0
        
        print(f"[SMART-CACHE] 初期化: {max_size_mb}MB, チャンク={chunk_frames}フレーム, デッドロック対策版")

    def get_chunk_id(self, frame_num):
        """フレーム番号からチャンクIDを計算"""
        return frame_num // self.chunk_frames

    def get_chunk_range(self, chunk_id):
        """チャンクのフレーム範囲を取得"""
        start_frame = chunk_id * self.chunk_frames
        end_frame = start_frame + self.chunk_frames - 1
        return start_frame, end_frame

    def record_frame_processing_time(self, frame_num, processing_time):
        """フレーム処理時間を記録（最小オーバーヘッド）"""
        chunk_id = self.get_chunk_id(frame_num)
        
        # ロック時間を最小限に
        if not self.mutex.tryLock(10):  # 10msタイムアウト
            return
            
        try:
            if chunk_id not in self.processing_costs:
                self.processing_costs[chunk_id] = {
                    'frame_times': [],
                    'total_time': 0.0,
                    'sample_count': 0,
                    'last_sample_time': time.time()
                }
            
            cost_data = self.processing_costs[chunk_id]
            cost_data['frame_times'].append(processing_time)
            cost_data['total_time'] += processing_time
            cost_data['sample_count'] += 1
            cost_data['last_sample_time'] = time.time()
            
            # 統計更新
            self.performance_stats['total_frames'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            
            # 3サンプル以上でポリシー更新（高速判定）
            if cost_data['sample_count'] == 3:
                self._update_chunk_policy(chunk_id)
        finally:
            self.mutex.unlock()

    def _update_chunk_policy(self, chunk_id):
        """高速なキャッシュポリシー決定"""
        cost_data = self.processing_costs[chunk_id]
        avg_ms_per_frame = (cost_data['total_time'] / cost_data['sample_count']) * 1000
        
        # 超高速判定（分岐最小化）
        if avg_ms_per_frame <= 33.3:
            policy, priority = 'no_cache', 0
        elif avg_ms_per_frame <= 50.0:
            policy, priority = 'short_term', 1
        elif avg_ms_per_frame <= 100.0:
            policy, priority = 'standard_cache', 2
        else:
            policy, priority = 'priority_cache', 3
        
        # 簡易平滑化（近傍2チャンクのみチェック）
        smoothed_policy = self._fast_temporal_smoothing(chunk_id, policy, priority)
        
        self.cache_policies[chunk_id] = {
            'policy': smoothed_policy['policy'],
            'priority': smoothed_policy['priority'],
            'avg_ms_per_frame': avg_ms_per_frame,
            'sample_size': cost_data['sample_count'],
            'last_updated': time.time()
        }

    def _fast_temporal_smoothing(self, chunk_id, proposed_policy, proposed_priority):
        """高速な時空間平滑化"""
        # 近傍1チャンクのみチェック（高速化）
        neighbors = []
        for offset in [-1, 1]:
            neighbor_id = chunk_id + offset
            if neighbor_id in self.cache_policies:
                neighbors.append(self.cache_policies[neighbor_id])
        
        if len(neighbors) >= 1:
            # シンプルな多数決
            policy_counts = {}
            for neighbor in neighbors:
                policy = neighbor['policy']
                policy_counts[policy] = policy_counts.get(policy, 0) + 1
            
            most_common = max(policy_counts.items(), key=lambda x: x[1])
            if most_common[1] >= len(neighbors) and most_common[0] != proposed_policy:
                return {'policy': most_common[0], 'priority': proposed_priority}
        
        return {'policy': proposed_policy, 'priority': proposed_priority}

    def should_cache_frame(self, frame_num, frame_data=None):
        """フレームをキャッシュすべきか判定"""
        chunk_id = self.get_chunk_id(frame_num)
        
        if chunk_id not in self.cache_policies:
            return True  # 未知はデフォルトでキャッシュ
        
        policy = self.cache_policies[chunk_id]
        
        # TTLチェック（簡略化）
        if policy['policy'] == 'no_cache':
            return False
        
        return policy['policy'] != 'no_cache'

    def get(self, frame_num):
        """フレーム取得 - デッドロック対策"""
        if not self.mutex.tryLock(10):  # 10msでタイムアウト
            return None
            
        try:
            chunk_id = self.get_chunk_id(frame_num)
            
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                if frame_num in chunk['frames']:
                    # アクセス記録更新
                    chunk['last_access'] = time.time()
                    self._update_access_order(chunk_id)
                    
                    # 統計更新
                    self.performance_stats['cache_hits'] += 1
                    return chunk['frames'][frame_num]
            
            # キャッシュミス
            self.performance_stats['cache_misses'] += 1
            return None
        finally:
            self.mutex.unlock()

    def put(self, frame_num, frame):
        """スマートキャッシュ判定付きのフレーム追加 - デッドロック対策"""
        if not self.mutex.tryLock(10):  # 10msでタイムアウト
            return
            
        try:
            if frame is None:
                self._remove_frame(frame_num)
                return
                
            # スマートキャッシュ判定
            if not self.should_cache_frame(frame_num, frame):
                return
                
            chunk_id = self.get_chunk_id(frame_num)
            
            # チャンクがなければ作成
            if chunk_id not in self.chunks:
                self.chunks[chunk_id] = {
                    'frames': {},
                    'size_mb': 0,
                    'last_access': time.time()
                }
            
            chunk = self.chunks[chunk_id]
            frame_size_mb = frame.nbytes / (1024 * 1024)
            
            # 既存フレームを上書きする場合はサイズ調整
            if frame_num in chunk['frames']:
                old_frame = chunk['frames'][frame_num]
                old_size_mb = old_frame.nbytes / (1024 * 1024)
                chunk['size_mb'] -= old_size_mb
                self.current_size_mb -= old_size_mb
            
            # 新規フレーム追加
            chunk['frames'][frame_num] = frame
            chunk['size_mb'] += frame_size_mb
            chunk['last_access'] = time.time()
            self.current_size_mb += frame_size_mb
            
            # LRU更新
            self._update_access_order(chunk_id)
            
            # 容量超過時は非同期クリーンアップをスケジュール
            if self.current_size_mb > self.max_size_mb:
                self._schedule_async_cleanup()
        finally:
            self.mutex.unlock()

    def _update_access_order(self, chunk_id):
        """LRU順序を更新"""
        if chunk_id in self.access_order:
            self.access_order.remove(chunk_id)
        self.access_order.append(chunk_id)

    def _schedule_async_cleanup(self):
        """非同期クリーンアップをスケジュール"""
        if not self.pending_cleanup and not self.cleanup_timer.isActive():
            self.pending_cleanup = True
            self.cleanup_timer.start(50)  # 50ms後に実行

    def _async_cleanup(self):
        """非同期でチャンク単位のクリーンアップを実行 - デッドロック対策"""
        if not self.pending_cleanup:
            return
            
        start_time = time.time()
        removed_count = 0
        
        if not self.mutex.tryLock(50):  # 50msでタイムアウト
            self.cleanup_timer.start(25)  # 再試行
            return
            
        try:
            if self.current_size_mb <= self.max_size_mb * 0.8:
                self.pending_cleanup = False
                return
            
            # 保護対象のチャンクを計算（優先度考慮）
            protected_chunks = self._get_protected_chunks()
            
            # 優先度の低いチャンクから削除
            chunks_to_remove = []
            for chunk_id in list(self.access_order):
                if (chunk_id not in protected_chunks and 
                    self._get_chunk_cleanup_priority(chunk_id) <= 1):  # 低優先度
                    chunks_to_remove.append(chunk_id)
            
            # それでも足りない場合は標準優先度を対象に
            if self.current_size_mb > self.max_size_mb * 0.8:
                for chunk_id in list(self.access_order):
                    if (chunk_id not in protected_chunks and 
                        chunk_id not in chunks_to_remove and
                        self._get_chunk_cleanup_priority(chunk_id) <= 2):  # 標準優先度以下
                        chunks_to_remove.append(chunk_id)
            
            # 削除実行
            for chunk_id in chunks_to_remove:
                if self._remove_chunk(chunk_id):
                    removed_count += 1
                    if self.current_size_mb <= self.max_size_mb * 0.7:
                        break
                    if removed_count >= 2:  # 一度に削除する数を減らして高速化
                        break
            
            # 必要に応じて継続
            if self.current_size_mb > self.max_size_mb * 0.8:
                self.cleanup_timer.start(25)
            else:
                self.pending_cleanup = False
        finally:
            self.mutex.unlock()

    def _get_chunk_cleanup_priority(self, chunk_id):
        """クリーンアップ時の優先度（低いほど先に削除）"""
        if chunk_id not in self.cache_policies:
            return 0  # 未知は最低優先度
        
        policy = self.cache_policies[chunk_id]
        
        # ポリシーに基づく優先度（数値が小さいほど削除されやすい）
        priority_map = {
            'no_cache': 0,
            'short_term': 1, 
            'standard_cache': 2,
            'priority_cache': 3
        }
        
        return priority_map.get(policy['policy'], 0)

    def _get_protected_chunks(self):
        """保護対象のチャンクを計算"""
        current_chunk = self.get_chunk_id(self.current_playhead)
        protected = set()
        
        # 現在のチャンクと前後1チャンクを保護（範囲を縮小して高速化）
        for offset in range(-1, 2):  # -1, 0, 1
            protected.add(current_chunk + offset)
        
        # 高優先度チャンクを追加保護
        for chunk_id in list(self.chunks.keys()):
            if self._get_chunk_cleanup_priority(chunk_id) >= 3:  # 高優先度
                protected.add(chunk_id)
                
        return protected

    def _remove_chunk(self, chunk_id):
        """チャンク全体を削除"""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            self.current_size_mb -= chunk['size_mb']
            del self.chunks[chunk_id]
            
            if chunk_id in self.access_order:
                self.access_order.remove(chunk_id)
            
            return True
        return False

    def _remove_frame(self, frame_num):
        """単一フレームを削除（特殊ケース用）"""
        chunk_id = self.get_chunk_id(frame_num)
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            if frame_num in chunk['frames']:
                frame = chunk['frames'][frame_num]
                frame_size_mb = frame.nbytes / (1024 * 1024)
                
                del chunk['frames'][frame_num]
                chunk['size_mb'] -= frame_size_mb
                self.current_size_mb -= frame_size_mb
                
                # チャンクが空になったら完全削除
                if not chunk['frames']:
                    self._remove_chunk(chunk_id)

    def update_playhead(self, frame_num):
        """再生位置を更新（保護対象の計算用）"""
        self.current_playhead = frame_num

    def clear(self):
        """キャッシュ全クリア - デッドロック対策"""
        if not self.mutex.tryLock(100):  # 100msでタイムアウト
            print("[WARNING] キャッシュクリア: ミューテックスの取得に失敗")
            return
            
        try:
            self.chunks.clear()
            self.access_order.clear()
            self.current_size_mb = 0
            self.pending_cleanup = False
            self.cleanup_timer.stop()
            
            # スマートキャッシュ関連もクリア
            self.processing_costs.clear()
            self.cache_policies.clear()
            self.prefetch_queue.clear()
            self.performance_stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_frames': 0,
                'total_processing_time': 0.0
            }
        finally:
            self.mutex.unlock()

    def get_stats(self):
        """キャッシュ統計を取得 - デッドロック対策"""
        if not self.mutex.tryLock(10):
            return {
                'chunk_count': 0,
                'total_frames': 0,
                'size_mb': 0,
                'max_mb': self.max_size_mb,
                'chunk_frames': self.chunk_frames,
                'hit_ratio': 0.0,
                'avg_processing_time': 0.0,
                'policy_distribution': {}
            }
            
        try:
            chunk_count = len(self.chunks)
            total_frames = sum(len(chunk['frames']) for chunk in self.chunks.values())
            
            stats = {
                'chunk_count': chunk_count,
                'total_frames': total_frames,
                'size_mb': self.current_size_mb,
                'max_mb': self.max_size_mb,
                'chunk_frames': self.chunk_frames
            }
            
            # スマートキャッシュ統計を追加
            total_requests = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
            if total_requests > 0:
                stats['hit_ratio'] = self.performance_stats['cache_hits'] / total_requests
            else:
                stats['hit_ratio'] = 0.0
                
            if self.performance_stats['total_frames'] > 0:
                stats['avg_processing_time'] = (self.performance_stats['total_processing_time'] / self.performance_stats['total_frames']) * 1000
            else:
                stats['avg_processing_time'] = 0.0
                
            stats['policy_distribution'] = {}
            for policy in self.cache_policies.values():
                policy_name = policy['policy']
                stats['policy_distribution'][policy_name] = stats['policy_distribution'].get(policy_name, 0) + 1
            
            return stats
        finally:
            self.mutex.unlock()


class VideoGLWidget(QOpenGLWidget):
    playback_toggled = pyqtSignal()
    video_dropped = pyqtSignal(str)
    seek_requested = pyqtSignal(int)
    toggle_mute_signal = pyqtSignal()
    toggle_ai_processing_signal = pyqtSignal()
    
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
            self.update_fs_ui_position()
            self.fs_progress_bar.show()
            self.fs_time_label.show()
            QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
            self.ui_hide_timer.start(3000)
    
    def update_fs_ui_position(self):
        """フルスクリーンUI位置更新"""
        if not self.is_fullscreen:
            return
            
        bar_height = 8
        bar_margin = 20
        self.fs_progress_bar.setGeometry(
            bar_margin, 
            self.height() - bar_height - bar_margin, 
            self.width() - bar_margin * 2, 
            bar_height
        )
        
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
        super().resizeEvent(event)
        if self.is_fullscreen:
            self.update_fs_ui_position()
    
    def mouseMoveEvent(self, event):
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
    
    def dragEnterEvent(self, event: QDragEnterEvent):
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
                self.video_dropped.emit(file_path)
                event.acceptProposedAction()
    
    def is_video_file(self, file_path):
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts']
        file_ext = Path(file_path).suffix.lower()
        return file_ext in video_extensions
    
    def get_main_window(self):
        """メインウィンドウを安全に取得"""
        # 親ウィジェットから再帰的にメインウィンドウを探す
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'seek_relative'):
                return parent
            parent = parent.parent()
        
        # トップレベルウィンドウから探す
        for widget in QApplication.topLevelWidgets():
            if hasattr(widget, 'seek_relative'):
                return widget
        
        return None
    
    def keyPressEvent(self, event):
        if self.is_fullscreen:
            key = event.key()
            if key == Qt.Key.Key_F or key == Qt.Key.Key_Escape:
                self.toggle_fullscreen()
            elif key == Qt.Key.Key_Space or key == Qt.Key.Key_K:
                self.playback_toggled.emit()
            elif key == Qt.Key.Key_Right or key == Qt.Key.Key_L:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(300)
            elif key == Qt.Key.Key_Left or key == Qt.Key.Key_J:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(-300)
            elif key == Qt.Key.Key_Semicolon:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(30)
            elif key == Qt.Key.Key_H:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(-30)
            elif key == Qt.Key.Key_M:
                self.toggle_mute_signal.emit()
            elif key == Qt.Key.Key_X:
                self.toggle_ai_processing_signal.emit()
        else:
            key = event.key()
            if key == Qt.Key.Key_M:
                self.toggle_mute_signal.emit()
            elif key == Qt.Key.Key_X:
                self.toggle_ai_processing_signal.emit()
            else:
                super().keyPressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        parent = self.window()
        if hasattr(parent, 'is_paused'):
            current_pause_state = parent.is_paused
        
        self.toggle_fullscreen()
        
        if hasattr(parent, 'is_paused') and hasattr(parent, 'process_thread'):
            QTimer.singleShot(100, lambda: self.restore_playback_state(parent, current_pause_state))
    
    def restore_playback_state(self, parent, original_pause_state):
        if hasattr(parent, 'process_thread') and parent.process_thread and parent.process_thread.isRunning():
            if original_pause_state:
                parent.process_thread.pause()
                parent.is_paused = True
                parent.play_pause_btn.setText("▶ 再開")
                parent.mode_label.setText("📊 モード: ⏸ 一時停止中")
                self.set_progress_bar_color('red')
            else:
                parent.process_thread.resume()
                parent.is_paused = False
                parent.play_pause_btn.setText("⏸ 一時停止")
                parent.mode_label.setText("📊 モード: 🔄 AI処理中")
                self.set_progress_bar_color('#00ff00')
    
    def mousePressEvent(self, event):
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
            
            QApplication.processEvents()
            self.update_fs_ui_position()
            self.show_fs_ui()
        else:
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
        
        frame_size_bytes = w * h * 3
        clip_size_bytes = max_clip_length * 256 * 256 * 4
        
        max_frames = max(100, (queue_size_mb * 1024 * 1024) // frame_size_bytes)
        max_clips = max(10, (queue_size_mb * 1024 * 1024) // clip_size_bytes)
        
        self._parent.frame_restoration_queue = queue.Queue(maxsize=max_frames)
        self._parent.mosaic_clip_queue = queue.Queue(maxsize=max_clips)
        self._parent.restored_clip_queue = queue.Queue(maxsize=max_clips)
        self._parent.mosaic_detector.mosaic_clip_queue = self._parent.mosaic_clip_queue
        
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
        self.pause_mutex = QMutex()  # 通常のミューテックス
        
        self.audio_thread = audio_thread
        self.video_fps = video_fps
        self.total_frames = 0
        
        # 高速シーク用の変数
        self._seek_requested = False
        self._seek_target = 0
        self._seek_mutex = QMutex()  # 通常のミューテックス
        
        # デッドロック防止用
        self._safe_stop = False
        
        print(f"[THREAD-{thread_id}] プロセススレッド初期化完了")

    def request_seek(self, target_frame):
        """高速シークリクエスト - デッドロック対策"""
        if not self._seek_mutex.tryLock(10):
            print(f"[THREAD-{self.thread_id}] シークリクエスト: ミューテックス取得失敗")
            return False
            
        try:
            self._seek_requested = True
            self._seek_target = target_frame
            print(f"[THREAD-{self.thread_id}] シークリクエスト受信: フレーム{target_frame}")
            return True
        finally:
            self._seek_mutex.unlock()

    def pause(self):
        """高速一時停止 - デッドロック対策"""
        if not self.pause_mutex.tryLock(10):
            print(f"[THREAD-{self.thread_id}] 一時停止: ミューテックス取得失敗")
            return
            
        try:
            self.is_paused = True
            if self.audio_thread:
                self.audio_thread.pause_audio()
            print(f"[THREAD-{self.thread_id}] 一時停止完了")
        finally:
            self.pause_mutex.unlock()

    def resume(self):
        """高速再開 - デッドロック対策"""
        if not self.pause_mutex.tryLock(10):
            print(f"[THREAD-{self.thread_id}] 再開: ミューテックス取得失敗")
            return
            
        try:
            self.is_paused = False
            if self.audio_thread:
                start_sec = self.start_frame / self.video_fps if self.video_fps > 0 else 0
                self.audio_thread.resume_audio(start_sec)
            print(f"[THREAD-{self.thread_id}] 再開完了")
        finally:
            self.pause_mutex.unlock()

    def safe_stop(self):
        """安全な停止 - デッドロック対策"""
        print(f"[THREAD-{self.thread_id}] 安全停止開始")
        self._safe_stop = True
        self._stop_flag = True
        self.is_running = False
        self.is_paused = False
        
        # フレームレストーラーの停止（例外を無視）
        if self.frame_restorer:
            try:
                self.frame_restorer.stop()
            except Exception as e:
                print(f"[THREAD-{self.thread_id}] フレームレストーラー停止中の例外: {e}")
        
        # スレッド終了待機（タイムアウト付き）
        if not self.wait(1000):  # 1秒待機
            print(f"[THREAD-{self.thread_id}] スレッド終了待機タイムアウト、強制終了")
            self.terminate()
            self.wait(500)
        
        print(f"[THREAD-{self.thread_id}] 安全停止完了")

    def run(self):
        print(f"[THREAD-{self.thread_id}] スレッド開始")
        
        self.is_running = True
        self._stop_flag = False
        self._safe_stop = False
        
        try:
            if not LADA_AVAILABLE:
                print(f"[THREAD-{self.thread_id}] LADA利用不可")
                return
            
            video_meta = video_utils.get_video_meta_data(self.video_path)
            self.total_frames = video_meta.frames_count
            self.video_fps = video_meta.video_fps
            
            print(f"[THREAD-{self.thread_id}] 動画情報: {self.total_frames}フレーム, {self.video_fps}FPS")
            
            if self._stop_flag or self._safe_stop:
                return
            
            detection_model, restoration_model, pad_mode = load_models(
                device="cuda:0",
                mosaic_restoration_model_name="basicvsrpp-v1.2",
                mosaic_restoration_model_path=str(self.restoration_path),
                mosaic_restoration_config_path=None,
                mosaic_detection_model_path=str(self.detection_path)
            )
            
            if self._stop_flag or self._safe_stop:
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
            
            # キャッシュに再生位置を通知
            self.frame_cache.update_playhead(frame_count)
            
            if self.audio_thread and not self._safe_stop:
                start_sec = self.start_frame / self.video_fps if self.video_fps > 0 else 0
                self.audio_thread.start_playback(str(self.video_path), start_sec)
                
            cache_frames_during_pause = 1800
            paused_cache_count = 0
            
            consecutive_cached_frames = 0
            max_consecutive_cached = 30
            
            while self.is_running and not self._stop_flag and not self._safe_stop and frame_count < self.total_frames:
                # 安全停止チェック
                if self._safe_stop:
                    break
                    
                # シークリクエストチェック（高速）
                seek_processed = False
                if self._seek_mutex.tryLock(1):  # 1msでタイムアウト
                    try:
                        if self._seek_requested:
                            frame_count = self._seek_target
                            self.start_frame = frame_count
                            start_ns = int((frame_count / self.video_fps) * 1_000_000_000)
                            
                            # フレームレストーラーを再起動
                            try:
                                self.frame_restorer.stop()
                            except:
                                pass
                            
                            self.frame_restorer.start(start_ns=start_ns)
                            frame_restorer_iter = iter(self.frame_restorer)
                            pending_ai_frame = None
                            
                            # 状態リセット
                            start_time = time.time()
                            total_pause_duration = 0
                            frame_count_at_reset = frame_count
                            last_mode_was_cached = False
                            paused_cache_count = 0
                            pause_start_time = 0
                            
                            # キャッシュに再生位置を通知
                            self.frame_cache.update_playhead(frame_count)
                            
                            # 音声シーク
                            if self.audio_thread and not self._safe_stop:
                                target_sec = frame_count / self.video_fps
                                self.audio_thread.seek_to_time(target_sec)
                            
                            self._seek_requested = False
                            seek_processed = True
                            print(f"[THREAD-{self.thread_id}] シーク完了: フレーム{frame_count}")
                    finally:
                        self._seek_mutex.unlock()
                
                if seek_processed:
                    continue
                
                # フレーム処理開始時間
                frame_start_time = time.time()
                
                # キャッシュに再生位置を定期的に通知
                if frame_count % 30 == 0:
                    self.frame_cache.update_playhead(frame_count)
                
                # 一時停止チェック
                pause_check_start = time.time()
                is_paused_check = False
                if self.pause_mutex.tryLock(1):  # 1msでタイムアウト
                    try:
                        is_paused_check = self.is_paused
                    finally:
                        self.pause_mutex.unlock()
                
                if is_paused_check and not self._stop_flag and not self._safe_stop:
                    if pause_start_time == 0:
                        pause_start_time = time.time()
                        paused_cache_count = 0
                        print(f"[THREAD-{self.thread_id}] 一時停止開始")
                    
                    if paused_cache_count < cache_frames_during_pause:
                        if self.frame_cache.get(frame_count + paused_cache_count) is None:
                            try:
                                item = next(frame_restorer_iter)
                                if item is not None:
                                    restored_frame, _ = item
                                    self.frame_cache.put(frame_count + paused_cache_count, restored_frame)
                                    paused_cache_count += 1
                            except StopIteration:
                                break
                        else:
                            paused_cache_count += 1
                    
                    time.sleep(0.01)
                    continue
                
                if pause_start_time > 0:
                    pause_duration = time.time() - pause_start_time
                    total_pause_duration += pause_duration
                    pause_start_time = 0
                    paused_cache_count = 0
                
                if self._stop_flag or self._safe_stop:
                    break
                
                cached_frame = self.frame_cache.get(frame_count)
                
                if cached_frame is not None:
                    final_frame = cached_frame
                    is_cached = True
                    consecutive_cached_frames += 1
                    processing_time = 0.0
                    
                    if consecutive_cached_frames > max_consecutive_cached:
                        self.frame_cache.put(frame_count, None)
                        cached_frame = None
                        consecutive_cached_frames = 0
                    
                    if not last_mode_was_cached:
                        start_time = time.time()
                        total_pause_duration = 0
                        frame_count_at_reset = frame_count
                    
                else:
                    consecutive_cached_frames = 0
                    
                    if last_mode_was_cached:
                        start_time = time.time()
                        total_pause_duration = 0
                        frame_count_at_reset = frame_count
                    
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
                    is_cached = False
                    
                    # 処理時間計測
                    processing_time = time.time() - frame_start_time
                    
                    # スマートキャッシュに処理時間を記録
                    if hasattr(self.frame_cache, 'record_frame_processing_time'):
                        self.frame_cache.record_frame_processing_time(frame_count, processing_time)
                    
                    # 条件付きでキャッシュに保存
                    if hasattr(self.frame_cache, 'should_cache_frame'):
                        if self.frame_cache.should_cache_frame(frame_count, final_frame):
                            self.frame_cache.put(frame_count, final_frame)
                    else:
                        self.frame_cache.put(frame_count, final_frame)
                
                last_mode_was_cached = is_cached
                
                frames_since_reset = frame_count - frame_count_at_reset
                target_time = frames_since_reset * frame_interval
                elapsed = time.time() - start_time - total_pause_duration
                wait_time = target_time - elapsed
                
                if wait_time < -0.5:
                    start_time = time.time() - (frames_since_reset * frame_interval)
                    total_pause_duration = 0
                    wait_time = 0
                
                if wait_time > 0:
                    time.sleep(min(wait_time, 0.1))
                
                # フレーム準備シグナル発行（デッドロック防止）
                if not self._safe_stop:
                    self.frame_ready.emit(final_frame, frame_count, is_cached)
                
                if self.audio_thread and frame_count % (int(self.video_fps) * 10) == 0 and not self._safe_stop:
                    current_sec = frame_count / self.video_fps
                    self.audio_thread.seek_to_time(current_sec)
                
                frame_count += 1
                if not self._safe_stop:
                    self.progress_updated.emit(frame_count, self.total_frames)
                
                if frame_count % 15 == 0:
                    elapsed = time.time() - start_time - total_pause_duration
                    actual_fps = (frame_count - self.start_frame) / elapsed if elapsed > 0 else 0
                    if not self._safe_stop:
                        self.fps_updated.emit(actual_fps)
            
            if not self._stop_flag and not self._safe_stop:
                self.finished_signal.emit()
            
        except Exception as e:
            print(f"[THREAD-{self.thread_id}] AI処理エラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[THREAD-{self.thread_id}] スレッド終了処理開始")
            # フレームレストーラーの安全な停止
            if self.frame_restorer and not self._safe_stop:
                try:
                    self.frame_restorer.stop()
                except Exception as e:
                    print(f"[THREAD-{self.thread_id}] フレームレストーラー停止中の例外: {e}")
            
            self.is_running = False
            if self.audio_thread and not self._safe_stop:
                self.audio_thread.stop_playback()
            print(f"[THREAD-{self.thread_id}] スレッド終了処理完了")


class AudioThread(QThread):
    def __init__(self, vlc_instance, initial_volume=100, is_muted=False):
        super().__init__()
        self.vlc_instance = vlc_instance
        self.player = self.vlc_instance.media_player_new()
        self._stop_flag = False
        self._is_paused = True
        self.volume = initial_volume
        self.user_muted = is_muted
        self.internal_muted = False
        
        self.player.audio_set_volume(self.volume)
        self._update_vlc_mute_state()
        
        print(f"[AUDIO] AudioThread初期化: Volume={self.volume}, Mute={self.user_muted}")

    def run(self):
        while not self._stop_flag:
            time.sleep(0.1)

    def _update_vlc_mute_state(self):
        if not VLC_AVAILABLE:
            return
        should_be_muted = self.user_muted or self.internal_muted
        self.player.audio_set_mute(should_be_muted)

    def set_internal_mute(self, is_muted):
        if not VLC_AVAILABLE:
            return
        self.internal_muted = is_muted
        self._update_vlc_mute_state()

    def start_playback(self, video_path, start_sec=0.0):
        if not VLC_AVAILABLE or self._stop_flag:
            return
            
        try:
            media = self.vlc_instance.media_new(video_path)
            self.player.set_media(media)
            
            msec = int(start_sec * 1000)
            
            self.set_internal_mute(True)
            self.player.play()
            time.sleep(0.01)
            
            if start_sec > 0.0:
                for _ in range(10):
                    if self.player.get_state() in (vlc.State.Playing, vlc.State.Paused):
                        break
                    time.sleep(0.05)
                
                if self.player.is_seekable():
                    self.player.set_time(msec)

            self.set_internal_mute(False)
            self._is_paused = False
        except Exception as e:
            print(f"[AUDIO] 再生開始エラー: {e}")
        
    def stop_playback(self):
        if not VLC_AVAILABLE:
            return
            
        try:
            self.player.stop()
            self._is_paused = True
        except Exception as e:
            print(f"[AUDIO] 再生停止エラー: {e}")

    def pause_audio(self):
        if not VLC_AVAILABLE or self._is_paused or self._stop_flag:
            return
            
        try:
            self.player.pause()
            self._is_paused = True
        except Exception as e:
            print(f"[AUDIO] 一時停止エラー: {e}")

    def resume_audio(self, start_sec):
        if not VLC_AVAILABLE or not self._is_paused or self._stop_flag:
            return
            
        try:
            self.seek_to_time(start_sec)
            self.player.play()
            self._is_paused = False
            self._update_vlc_mute_state()
        except Exception as e:
            print(f"[AUDIO] 再生再開エラー: {e}")

    def seek_to_time(self, seconds):
        if not VLC_AVAILABLE or self._stop_flag:
            return
            
        try:
            msec = int(seconds * 1000)
            
            self.set_internal_mute(True)
            
            for _ in range(10):
                if self.player.get_state() in (vlc.State.Playing, vlc.State.Paused):
                    break
                time.sleep(0.1)

            if self.player.is_seekable():
                self.player.set_time(msec)
            
            self.set_internal_mute(False)
        except Exception as e:
            print(f"[AUDIO] シークエラー: {e}")

    def set_volume(self, volume):
        if not VLC_AVAILABLE:
            return
        self.volume = max(0, min(100, volume))
        self.player.audio_set_volume(self.volume)

    def toggle_mute(self, is_muted):
        if not VLC_AVAILABLE:
            return
        self.user_muted = is_muted
        self._update_vlc_mute_state()

    def safe_stop(self):
        """安全な停止"""
        print("[AUDIO] 安全停止開始")
        self._stop_flag = True
        self.stop_playback()
        if not self.wait(1000):  # 1秒待機
            self.terminate()
            self.wait(500)
        print("[AUDIO] 安全停止完了")


class LadaFinalPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setAcceptDrops(True)
        
        self.settings = self.load_settings()
        
        # スマートキャッシュで初期化
        chunk_frames = self.settings.get('chunk_frames', 150)
        cache_size_mb = self.settings.get('cache_size_mb', 12288)
        self.frame_cache = SmartChunkBasedCache(
            max_size_mb=cache_size_mb, 
            chunk_frames=chunk_frames
        )
        
        self.current_video = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 30.0
        self.is_playing = False
        self.is_paused = False
        self.thread_counter = 0
        self._seeking = False
        self.ai_processing_enabled = True
        
        # process_threadをNoneで明示的に初期化
        self.process_thread = None
        
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
        print("[MAIN] プレイヤー初期化完了 - デッドロック対策版")

    def load_settings(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    settings = json.load(f)
                    print(f"[MAIN] 設定読み込み: 音量={settings.get('audio_volume')}, ミュート={settings.get('audio_muted')}")
                    return settings
            except:
                pass
        
        return {
            'batch_size': 16,
            'queue_size_mb': 12288,
            'max_clip_length': 8,
            'cache_size_mb': 12288,
            'chunk_frames': 150,
            'audio_volume': 100, 
            'audio_muted': False
        }

    def save_settings(self):
        if self.audio_thread:
            if not self.audio_thread.user_muted:
                self.settings['audio_volume'] = self.audio_thread.volume
            self.settings['audio_muted'] = self.audio_thread.user_muted
            
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"[MAIN] 設定保存失敗: {e}")

    def init_ui(self):
        self.setWindowTitle("LADA REALTIME PLAYER V1.0 - Smart Cache - デッドロック対策版")
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
        self.video_widget.toggle_mute_signal.connect(self.toggle_mute_shortcut)
        self.video_widget.toggle_ai_processing_signal.connect(self.toggle_ai_processing)
        self.video_layout.addWidget(self.video_widget)
        layout.addLayout(self.video_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.mousePressEvent = self.seek_click
        layout.addWidget(self.progress_bar)
        
        time_audio_layout = QHBoxLayout()
        
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #aaa;
                background-color: rgba(40, 40, 40, 150);
                padding: 2px 8px;
                border-radius: 4px;
                min-width: 150px;
            }
        """)
        self.time_label.setMaximumHeight(20)
        
        self.mute_btn = QPushButton("🔇")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(self.settings.get('audio_muted', False))
        self.mute_btn.setFixedWidth(40)
        self.mute_btn.clicked.connect(self.toggle_user_mute)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        initial_volume_ui = self.settings.get('audio_volume', 100)
        if isinstance(initial_volume_ui, float):
             initial_volume_ui = int(initial_volume_ui * 100)
        self.volume_slider.setValue(max(0, min(100, initial_volume_ui)))
        
        self.volume_slider.setFixedWidth(150)
        self.volume_slider.valueChanged.connect(self.set_volume_slider)
        self.volume_slider.sliderReleased.connect(self.save_audio_settings)
        
        time_audio_layout.addStretch(1)
        time_audio_layout.addWidget(self.time_label)
        
        if VLC_AVAILABLE:
            time_audio_layout.addStretch(1)
            time_audio_layout.addWidget(self.mute_btn)
            time_audio_layout.addWidget(self.volume_slider)
        else:
            time_audio_layout.addStretch(1)
        
        layout.addLayout(time_audio_layout)
        
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("動画を開く")
        self.open_btn.clicked.connect(self.open_video)
        
        self.play_pause_btn = QPushButton("⏸ 一時停止")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setEnabled(False)
        
        self.settings_btn = QPushButton("⚙️ 設定")
        self.settings_btn.clicked.connect(self.open_settings)
        
        self.ai_toggle_btn = QPushButton("🤖 AI: ON")
        self.ai_toggle_btn.setCheckable(True)
        self.ai_toggle_btn.setChecked(True)
        self.ai_toggle_btn.clicked.connect(self.toggle_ai_processing)
        
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.play_pause_btn)
        btn_layout.addWidget(self.settings_btn)
        btn_layout.addWidget(self.ai_toggle_btn)
        layout.addLayout(btn_layout)
        
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("⚡ FPS: --")
        self.mode_label = QLabel("📊 モード: 待機中")
        self.cache_label = QLabel("💾 キャッシュ: 0 MB")
        self.smart_cache_label = QLabel("🤖 スマート: --")
        
        for label in [self.fps_label, self.mode_label, self.cache_label, self.smart_cache_label]:
            label.setMaximumHeight(20)
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.mode_label)
        stats_layout.addWidget(self.cache_label)
        stats_layout.addWidget(self.smart_cache_label)
        layout.addLayout(stats_layout)
        
        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(100)
        info.setText("""
V1.0 Smart Cache Edition - デッドロック対策版 : 
操作: F=フルスクリーントグル | Space=再生/停止 | M=ミュートトグル | X=AI処理トグル | 進捗バークリックでシーク
デッドロック対策: タイムアウト付きミューテックス、安全なスレッド停止
""")
        layout.addWidget(info)
        
        self.setup_shortcuts()
        
        if self.audio_thread:
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
        
        self.shortcut_mute = QShortcut(QKeySequence('M'), self)
        self.shortcut_mute.activated.connect(self.toggle_mute_shortcut)
        
        self.shortcut_ai_toggle = QShortcut(QKeySequence('X'), self)
        self.shortcut_ai_toggle.activated.connect(self.toggle_ai_processing)

    def toggle_mute_shortcut(self):
        if self.audio_thread:
            new_mute_state = not self.audio_thread.user_muted
            self.audio_thread.toggle_mute(new_mute_state)
            self.mute_btn.setChecked(new_mute_state)
            self.mute_btn.setText("🔇" if new_mute_state else "🔊")
            
            if new_mute_state:
                self.settings['last_volume'] = self.audio_thread.volume
                self.volume_slider.setValue(0)
            else:
                unmuted_volume = self.settings.get('last_volume', self.settings.get('audio_volume', 100))
                if isinstance(unmuted_volume, float):
                    unmuted_volume = int(unmuted_volume * 100)
                unmuted_volume = max(1, min(100, unmuted_volume))
                
                self.volume_slider.setValue(unmuted_volume)
                self.audio_thread.set_volume(unmuted_volume)
            
            self.save_audio_settings()

    def toggle_user_mute(self, checked):
        if self.audio_thread:
            self.audio_thread.toggle_mute(checked)
            self.mute_btn.setText("🔇" if checked else "🔊")
            
            if checked:
                self.settings['last_volume'] = self.audio_thread.volume
                self.volume_slider.setValue(0)
            else:
                unmuted_volume = self.settings.get('last_volume', self.settings.get('audio_volume', 100))
                if isinstance(unmuted_volume, float):
                    unmuted_volume = int(unmuted_volume * 100)
                unmuted_volume = max(1, min(100, unmuted_volume))
                
                self.volume_slider.setValue(unmuted_volume)
                self.audio_thread.set_volume(unmuted_volume)
            
            self.save_audio_settings()

    def set_volume_slider(self, value):
        if self.audio_thread:
            self.audio_thread.set_volume(value)
            
            if value > 0 and self.audio_thread.user_muted:
                self.audio_thread.toggle_mute(False)
                self.mute_btn.setChecked(False)
                self.mute_btn.setText("🔊")
            
            self.settings['audio_volume'] = value
            self.save_audio_settings()

    def toggle_ai_processing(self):
        self.ai_processing_enabled = not self.ai_processing_enabled
        
        if self.ai_processing_enabled:
            self.ai_toggle_btn.setText("🤖 AI: ON")
            self.ai_toggle_btn.setChecked(True)
            self.mode_label.setText("📊 モード: 🔄 AI処理有効")
        else:
            self.ai_toggle_btn.setText("🎥 原画: ON")
            self.ai_toggle_btn.setChecked(False)
            self.mode_label.setText("📊 モード: 🎥 原画再生")
        
        if self.current_video:
            current_frame = self.current_frame
            self.safe_restart_playback(current_frame)

    def safe_restart_playback(self, start_frame):
        """安全な再生再開"""
        print(f"[MAIN] 安全な再生再開: フレーム{start_frame}")
        
        # 安全な停止
        self.safe_stop()
        
        # 即時再開
        self.start_processing_from_frame(start_frame)

    def dragEnterEvent(self, event: QDragEnterEvent):
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
                self.load_video(file_path)
                event.acceptProposedAction()

    def is_video_file(self, file_path):
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts']
        file_ext = Path(file_path).suffix.lower()
        return file_ext in video_extensions

    def update_stats(self):
        stats = self.frame_cache.get_stats()
        self.cache_label.setText(f"💾 キャッシュ: {stats['size_mb']:.1f}MB ({stats['total_frames']}f)")
        
        # スマートキャッシュ統計
        if 'hit_ratio' in stats and 'policy_distribution' in stats:
            hit_ratio = stats['hit_ratio'] * 100
            
            policy_summary = ""
            total_chunks = sum(stats['policy_distribution'].values())
            for policy, count in stats['policy_distribution'].items():
                percentage = (count / total_chunks) * 100 if total_chunks > 0 else 0
                if percentage >= 5.0:
                    policy_summary += f"{policy[:2]}:{percentage:.0f}% "
            
            self.smart_cache_label.setText(f"🤖 Hit:{hit_ratio:.0f}% {policy_summary.strip()}")

    def format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def on_frame_ready(self, frame, frame_num, is_cached, thread_id):
        if self.process_thread and thread_id == self.process_thread.thread_id:
            self.current_frame = frame_num
            self.video_widget.update_frame(frame)
            
            # キャッシュに再生位置を通知
            self.frame_cache.update_playhead(frame_num)
            
            current_sec = frame_num / self.video_fps if self.video_fps > 0 else 0
            total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            
            current_time = self.format_time(current_sec)
            total_time = self.format_time(total_sec)
            self.time_label.setText(f"{current_time} / {total_time}")
            
            if is_cached:
                self.mode_label.setText("📊 モード: 💾 キャッシュ再生")
                if not self.is_paused:
                    self.video_widget.set_progress_bar_color('yellow')
            else:
                self.mode_label.setText("📊 モード: 🔄 AI処理中")
                if not self.is_paused:
                    self.video_widget.set_progress_bar_color('#00ff00')

    def on_progress_update(self, current, total):
        self.current_frame = current
        self.progress_bar.setValue(current)
        self.video_widget.update_progress(current)

    def on_processing_finished(self):
        print("[MAIN] AI処理が完了しました")
        self.safe_stop()
        self.mode_label.setText("📊 モード: 完了")

    def seek_relative(self, delta):
        """高速相対シーク"""
        if self.total_frames == 0 or not self.current_video:
            return
        
        target_frame = max(0, min(self.current_frame + delta, self.total_frames - 1))
        
        # 即時UI更新
        self.current_frame = target_frame
        self.progress_bar.setValue(target_frame)
        self.video_widget.update_progress(target_frame)
        
        # キャッシュチェック
        cached_frame = self.frame_cache.get(target_frame)
        if cached_frame is not None:
            self.video_widget.update_frame(cached_frame)
        
        # 非同期シーク処理
        self.fast_seek_to_frame(target_frame)

    def fast_seek_to_frame(self, target_frame):
        """高速シーク処理"""
        if not self.current_video or self._seeking:
            return
        
        self._seeking = True
        
        # 音声シーク（非ブロッキング）
        if self.audio_thread:
            target_sec = target_frame / self.video_fps if self.video_fps > 0 else 0
            self.audio_thread.seek_to_time(target_sec)
        
        # スレッドが動作中の場合はシークリクエストを送信
        if self.process_thread and self.process_thread.isRunning():
            success = self.process_thread.request_seek(target_frame)
            if not success:
                print("[MAIN] シークリクエスト送信失敗")
        else:
            # スレッドがなければ新規開始
            self.start_processing_from_frame(target_frame)
        
        self._seeking = False

    def seek_to_frame(self, target_frame):
        """互換性のためのシーク処理"""
        self.fast_seek_to_frame(target_frame)

    def closeEvent(self, event):
        print("=== 安全な終了処理 ===")
        self.safe_stop()
        
        # 音声スレッドの安全な停止
        if self.audio_thread:
            self.audio_thread.safe_stop()
        
        # OpenGLリソース解放
        if hasattr(self, 'video_widget') and self.video_widget.texture_id:
            try:
                self.video_widget.makeCurrent()
                glDeleteTextures([self.video_widget.texture_id])
            except:
                pass
        
        self.frame_cache.clear()
        self.save_settings()
        print("=== 終了処理完了 ===")
        event.accept()

    def seek_click(self, event):
        if self.total_frames > 0:
            pos = event.pos().x()
            width = self.progress_bar.width()
            target_frame = int((pos / width) * self.total_frames)
            self.fast_seek_to_frame(target_frame)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画選択", "", "Videos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)"
        )
        if path:
            self.load_video(path)

    def open_settings(self):
        dialog = SettingsDialog(self, self.settings)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()
            
            needs_restart = False
            needs_cache_rebuild = False
            
            cache_related_settings = [
                'batch_size', 'queue_size_mb', 'max_clip_length',
                'cache_size_mb', 'chunk_frames'
            ]
            
            for key in cache_related_settings:
                if new_settings.get(key) != self.settings.get(key):
                    needs_restart = True
                    if key == 'chunk_frames':
                        needs_cache_rebuild = True
                    break
            
            if needs_restart:
                self.settings.update(new_settings)
                self.save_settings()

                print("[MAIN] 設定変更 - 安全なリセット実行")
                self.safe_stop()
                
                if needs_cache_rebuild:
                    self.frame_cache = SmartChunkBasedCache(
                        max_size_mb=self.settings['cache_size_mb'],
                        chunk_frames=self.settings['chunk_frames']
                    )
                else:
                    self.frame_cache = SmartChunkBasedCache(
                        max_size_mb=self.settings['cache_size_mb'],
                        chunk_frames=self.settings.get('chunk_frames', 150)
                    )
                
                if self.current_video:
                    self.load_video(self.current_video)
                
                msg = QMessageBox(self)
                msg.setWindowTitle("設定変更")
                if needs_cache_rebuild:
                    msg.setText("キャッシュ設定を変更しました。\nキャッシュを再構築します。")
                else:
                    msg.setText("処理設定を変更しました。\n再生を再開します。")
                msg.setIcon(QMessageBox.Icon.Information)
                msg.exec()
            else:
                self.settings.update(new_settings)
                self.save_settings()

    def toggle_fullscreen_shortcut(self):
        self.video_widget.toggle_fullscreen()

    def escape_fullscreen_shortcut(self):
        if self.video_widget.is_fullscreen:
            self.video_widget.toggle_fullscreen()

    def save_audio_settings(self):
        self.save_settings()

    def load_video(self, path):
        print(f"[MAIN] 動画読み込み: {path}")
        self.safe_stop()
        self.frame_cache.clear()
        self.video_widget.clear_frame()
        
        self.current_video = path
        
        fullpath = str(Path(path).resolve())
        max_length = 100
        if len(fullpath) > max_length:
            fullpath = "..." + fullpath[-(max_length-3):]
        self.filename_label.setText(f"🎬 {fullpath}")
        self.filename_label.show()
        
        self.original_capture = None
        if not self.ai_processing_enabled:
            try:
                self.original_capture = cv2.VideoCapture(str(path))
                if not self.original_capture.isOpened():
                    print("[MAIN] 元動画の読み込みに失敗")
                    self.original_capture = None
            except Exception as e:
                print(f"[MAIN] 元動画キャプチャ作成失敗: {e}")
                self.original_capture = None
        
        try:
            if self.ai_processing_enabled and LADA_AVAILABLE:
                video_meta = video_utils.get_video_meta_data(path)
                self.total_frames = video_meta.frames_count
                self.video_fps = video_meta.video_fps
            else:
                temp_capture = cv2.VideoCapture(str(path))
                if temp_capture.isOpened():
                    self.total_frames = int(temp_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.video_fps = temp_capture.get(cv2.CAP_PROP_FPS)
                    temp_capture.release()
                else:
                    self.total_frames = 0
                    self.video_fps = 30.0
            
            self.progress_bar.setMaximum(self.total_frames)
            self.video_widget.set_video_info(self.total_frames, self.video_fps)
            
        except Exception as e:
            print(f"[MAIN] 動画メタデータ取得失敗: {e}")
            self.total_frames = 0
            self.video_fps = 30.0
        
        self.start_processing_from_frame(0)
        mode_text = "🎥 原画" if not self.ai_processing_enabled else "🤖 AI"
        self.mode_label.setText(f"📊 選択: {Path(path).name} ({mode_text})")

    def start_processing_from_frame(self, start_frame):
        if not self.current_video:
            return
        
        print(f"[MAIN] フレーム{start_frame}から再生開始 (AI処理: {self.ai_processing_enabled})")
        
        # 既存のスレッド/タイマーが残っていないか確認
        if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.isRunning():
            print("[MAIN] 既存のAIスレッドが動作中です。安全停止します。")
            self.process_thread.safe_stop()
        
        if hasattr(self, 'original_timer') and self.original_timer and self.original_timer.isActive():
            print("[MAIN] 既存の原画タイマーが動作中です。停止します。")
            self.original_timer.stop()
        
        # AI処理無効時はOpenCVで直接再生
        if not self.ai_processing_enabled:
            self.start_original_playback(start_frame)
            return
        
        # AI処理有効時
        if not LADA_AVAILABLE:
            self.mode_label.setText("エラー: LADA利用不可")
            return
        
        # スレッドが既に動作していないか再確認
        if self.process_thread and self.process_thread.isRunning():
            print("[MAIN] スレッドがまだ動作しています。処理を中止します。")
            return
        
        model_dir = LADA_BASE_PATH / "model_weights"
        detection_path = model_dir / "lada_mosaic_detection_model_v3.1_fast.pt"
        restoration_path = model_dir / "lada_mosaic_restoration_model_generic_v1.2.pth"
        
        if not detection_path.exists() or not restoration_path.exists():
            self.mode_label.setText("エラー: モデルなし")
            return
        
        self.thread_counter += 1
        current_id = self.thread_counter
        
        # 新しいスレッドを作成
        self.process_thread = ProcessThread(
            self.current_video, detection_path, restoration_path,
            self.frame_cache, start_frame, current_id, self.settings,
            audio_thread=self.audio_thread, video_fps=self.video_fps
        )
        
        # シグナル接続
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
        
        # スレッド開始
        self.process_thread.start()
        self.is_playing = True
        self.is_paused = False
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸ 一時停止")
        self.mode_label.setText("📊 モード: 🔄 AI処理中")
        self.video_widget.set_progress_bar_color('#00ff00')
        
        print(f"[MAIN] AI処理スレッド開始完了: ID{current_id}")

    def start_original_playback(self, start_frame):
        """AI処理無効時の元動画再生"""
        print(f"[MAIN] 原画再生開始: フレーム{start_frame}")
        
        # 既存のキャプチャとタイマーを確実にクリーンアップ
        if hasattr(self, 'original_capture') and self.original_capture:
            self.original_capture.release()
            self.original_capture = None
        
        if hasattr(self, 'original_timer') and self.original_timer:
            self.original_timer.stop()
            self.original_timer = None
        
        # 新しいキャプチャを作成
        try:
            self.original_capture = cv2.VideoCapture(str(self.current_video))
            if not self.original_capture.isOpened():
                print("[MAIN] 元動画の読み込みに失敗")
                self.mode_label.setText("エラー: 動画読み込み失敗")
                return
        except Exception as e:
            print(f"[MAIN] 元動画キャプチャ作成失敗: {e}")
            self.mode_label.setText("エラー: 動画読み込み失敗")
            return
        
        # フレーム位置を設定
        self.original_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.current_frame = start_frame
        
        # 新しいタイマーを作成
        self.original_timer = QTimer()
        self.original_timer.timeout.connect(self.update_original_frame)
        frame_interval = int(1000 / self.video_fps) if self.video_fps > 0 else 33
        self.original_timer.start(frame_interval)
        
        # 状態設定
        self.is_playing = True
        self.is_paused = False
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸ 一時停止")
        self.mode_label.setText("📊 モード: 🎥 原画再生")
        self.video_widget.set_progress_bar_color('#00ff00')
        
        # キャッシュに再生位置を通知
        self.frame_cache.update_playhead(start_frame)
        
        # 音声再生開始
        if self.audio_thread:
            start_sec = start_frame / self.video_fps if self.video_fps > 0 else 0
            self.audio_thread.start_playback(str(self.current_video), start_sec)
        
        print(f"[MAIN] 原画再生開始完了: フレーム{start_frame}, 間隔{frame_interval}ms")

    def update_original_frame(self):
        if not hasattr(self, 'original_capture') or not self.original_capture or not self.is_playing or self.is_paused:
            return
        
        ret, frame = self.original_capture.read()
        if ret:
            self.video_widget.update_frame(frame)
            self.current_frame += 1
            self.progress_bar.setValue(self.current_frame)
            self.video_widget.update_progress(self.current_frame)
            
            # キャッシュに再生位置を通知
            if self.current_frame % 30 == 0:
                self.frame_cache.update_playhead(self.current_frame)
            
            current_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
            total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            current_time = self.format_time(current_sec)
            total_time = self.format_time(total_sec)
            self.time_label.setText(f"{current_time} / {total_time}")
            
            if self.current_frame >= self.total_frames:
                self.original_timer.stop()
                self.is_playing = False
                self.play_pause_btn.setText("▶ 再生")
                self.mode_label.setText("📊 モード: 🎥 再生完了")
        else:
            self.original_timer.stop()
            self.is_playing = False
            self.play_pause_btn.setText("▶ 再生")
            self.mode_label.setText("📊 モード: 🎥 再生完了")

    def toggle_playback(self):
        """安全な再生/一時停止トグル"""
        if not self.ai_processing_enabled and hasattr(self, 'original_timer'):
            if self.is_paused:
                self.original_timer.start()
                self.is_paused = False
                self.play_pause_btn.setText("⏸ 一時停止")
                self.mode_label.setText("📊 モード: 🎥 原画再生")
                self.video_widget.set_progress_bar_color('#00ff00')
                
                if self.audio_thread:
                    start_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
                    self.audio_thread.resume_audio(start_sec)
            else:
                self.original_timer.stop()
                self.is_paused = True
                self.play_pause_btn.setText("▶ 再開")
                self.mode_label.setText("📊 モード: 🎥 一時停止中")
                self.video_widget.set_progress_bar_color('red')
                
                if self.audio_thread:
                    self.audio_thread.pause_audio()
            return
        
        if not self.process_thread or not self.process_thread.isRunning():
            if self.current_video:
                self.start_processing_from_frame(self.current_frame)
            return
        
        if self.is_paused:
            self.process_thread.resume()
            self.is_paused = False
            self.play_pause_btn.setText("⏸ 一時停止")
            self.mode_label.setText("📊 モード: 🔄 AI処理中")
            self.video_widget.set_progress_bar_color('#00ff00')
        else:
            self.process_thread.pause()
            self.is_paused = True
            self.play_pause_btn.setText("▶ 再開")
            self.mode_label.setText("📊 モード: ⏸ 一時停止中")
            self.video_widget.set_progress_bar_color('red')

    def safe_stop(self):
        """安全な停止 - デッドロック防止"""
        print("[MAIN] 安全停止開始")
        
        # 状態フラグのみ設定
        self.is_playing = False
        self.is_paused = False
        
        # 原画処理の停止
        if hasattr(self, 'original_timer') and self.original_timer:
            self.original_timer.stop()
        
        if hasattr(self, 'original_capture') and self.original_capture:
            self.original_capture.release()
            self.original_capture = None
        
        # AI処理スレッドの安全な停止
        if hasattr(self, 'process_thread') and self.process_thread:
            self.process_thread.safe_stop()
            self.process_thread = None
        
        # UI状態リセット
        self.play_pause_btn.setText("▶ 再生")
        self.play_pause_btn.setEnabled(self.current_video is not None)
        
        print("[MAIN] 安全停止完了")


def main():
    app = QApplication(sys.argv)
    player = LadaFinalPlayer()
    player.show()
    
    # アプリケーション終了時の安全確保
    def safe_quit():
        player.close()
    
    app.aboutToQuit.connect(safe_quit)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()