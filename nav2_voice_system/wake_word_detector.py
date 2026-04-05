"""
Wake Word 감지 모듈

"""

import time
import threading
from typing import Optional


class WakeWordDetector:
    """
    Wake Word 감지기

    enabled=True  → wake word 들은 후에만 명령 처리
    enabled=False → 모든 발화 바로 처리 (기존 동작)
    """

    def __init__(self, config: dict):
        ww_cfg = config.get('wake_word', {})
        self.enabled = ww_cfg.get('enabled', False)
        self.keyword = ww_cfg.get('keyword', '안내야')
        self.sensitivity = ww_cfg.get('sensitivity', 0.8)
        self.timeout_sec = ww_cfg.get('timeout_sec', 10)
        self.confirm_sound = ww_cfg.get('confirm_sound', False)

        # 활성화 상태 (wake word 들은 후 timeout 동안 True)
        self._active = False
        self._active_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # A/B 테스트용 통계
        self._stats = {
            'total_utterances': 0,
            'wake_word_detected': 0,
            'commands_processed': 0,
            'commands_ignored': 0,
            'false_activations': 0,
        }

        if self.enabled:
            print(f"[WakeWord] 활성화 | 키워드: '{self.keyword}' | 타임아웃: {self.timeout_sec}초")
        else:
            print("[WakeWord] 비활성화 (모든 발화 처리)")

    # ── 핵심 메서드 ───────────────────────────────────────────

    def should_process(self, text: str) -> bool:
        """
        이 발화를 명령으로 처리할지 여부 반환

        Returns:
            True  → 명령으로 처리
            False → 무시 (wake word 대기 중)
        """
        self._stats['total_utterances'] += 1

        # wake word 비활성화 모드 → 항상 처리
        if not self.enabled:
            self._stats['commands_processed'] += 1
            return True

        text_lower = text.lower().strip()

        # wake word 포함 여부 확인
        if self._contains_wake_word(text_lower):
            self._stats['wake_word_detected'] += 1
            self._activate()

            # wake word만 있고 명령이 없는 경우 → 처리하지 않음 (활성화만)
            # 예: "안내야" 만 말한 경우
            remaining = self._strip_wake_word(text_lower)
            if len(remaining.strip()) < 2:
                print(f"[WakeWord] 감지됨 → {self.timeout_sec}초 활성화")
                return False

            # wake word + 명령을 한 번에 말한 경우
            # 예: "안내야 화장실 가줘"
            print(f"[WakeWord] 감지 + 명령 포함 → 처리")
            self._stats['commands_processed'] += 1
            return True

        # 현재 활성화 상태인지 확인
        with self._lock:
            is_active = self._active

        if is_active:
            self._stats['commands_processed'] += 1
            print(f"[WakeWord] 활성 상태 → 명령 처리")
            return True
        else:
            self._stats['commands_ignored'] += 1
            print(f"[WakeWord] 비활성 상태 → 무시: '{text}'")
            return False

    def get_clean_text(self, text: str) -> str:
        """
        발화에서 wake word 부분 제거 후 반환
        예: "안내야 화장실 가줘" → "화장실 가줘"
        """
        if not self.enabled:
            return text
        return self._strip_wake_word(text.lower().strip())

    def is_active(self) -> bool:
        """현재 wake word 활성화 상태"""
        with self._lock:
            return self._active

    def force_activate(self, duration: Optional[float] = None):
        """외부에서 강제 활성화 (예: 도착 후 안내 시작 시)"""
        self._activate(duration)

    def deactivate(self):
        """강제 비활성화"""
        with self._lock:
            self._active = False
            if self._active_timer:
                self._active_timer.cancel()
                self._active_timer = None

    # ── 내부 메서드 ───────────────────────────────────────────

    def _contains_wake_word(self, text_lower: str) -> bool:
        """wake word 포함 여부 (유사어 허용)"""
        # 정확한 매칭
        if self.keyword.lower() in text_lower:
            return True

        # 유사 발음 허용 (STT 오인식 대응)
        # 예: "안내야" → "안내야", "안내아", "한내야"
        similar_words = self._get_similar_words(self.keyword)
        return any(w in text_lower for w in similar_words)

    def _get_similar_words(self, keyword: str) -> list:
        """키워드 유사 발음 목록 (필요시 확장)"""
        similar_map = {
            '안내야': ['안내야', '안내아', '한내야', '안내이', '안내'],
            '캔고야': ['캔고야', '캔고아', '캔고', 'cango야', 'cango', '캔 고야'],
            '네비야': ['네비야', '네비아', '내비야'],
            '로봇아': ['로봇아', '로봇아', '로봇이'],
        }
        return similar_map.get(keyword, [keyword])

    def _strip_wake_word(self, text: str) -> str:
        """발화에서 wake word 제거"""
        for w in self._get_similar_words(self.keyword):
            text = text.replace(w, '').strip()
        return text

    def _activate(self, duration: Optional[float] = None):
        """wake word 활성화 타이머 시작"""
        timeout = duration or self.timeout_sec

        with self._lock:
            self._active = True
            # 기존 타이머 취소
            if self._active_timer:
                self._active_timer.cancel()

            # 타임아웃 후 자동 비활성화
            self._active_timer = threading.Timer(timeout, self._deactivate_callback)
            self._active_timer.daemon = True
            self._active_timer.start()

        if self.confirm_sound:
            self._play_confirm_sound()

    def _deactivate_callback(self):
        with self._lock:
            self._active = False
            self._active_timer = None
        print(f"[WakeWord] 타임아웃 → 비활성화")

    def _play_confirm_sound(self):
        """wake word 감지 확인음 재생 (짧은 삑 소리)"""
        try:
            import pygame
            import numpy as np
            sample_rate = 22050
            duration = 0.15
            freq = 880
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
            stereo = np.column_stack([wave, wave])
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
        except Exception:
            pass

    # ── 통계 ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """A/B 테스트용 통계 반환"""
        stats = self._stats.copy()
        total = stats['total_utterances']
        if total > 0:
            stats['process_rate'] = f"{stats['commands_processed']/total*100:.1f}%"
            stats['ignore_rate'] = f"{stats['commands_ignored']/total*100:.1f}%"
        return stats

    def print_stats(self):
        print(f"\n[WakeWord 통계] (enabled={self.enabled})")
        for k, v in self.get_stats().items():
            print(f"  {k}: {v}")
