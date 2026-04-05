"""
노트북 A (또는 NUC) - 음성 입출력 담당

"""

import os
import sys
import yaml
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cango_msgs.msg import SoundRequest

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(__file__))
from wake_word_detector import WakeWordDetector


@contextmanager
def suppress_stderr():
    devnull = open(os.devnull, 'w')
    old = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old
        devnull.close()


class VoiceClientNode(Node):
    def __init__(self):
        super().__init__('voice_client')

        # ── 설정 로드 ─────────────────────────────────────
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'system_config.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        topics = self.cfg['topics']
        speech_cfg = self.cfg.get('speech', {})
        self.emergency_keywords = self.cfg.get('emergency_keywords', ['정지', '멈춰'])

        # ── Wake Word 초기화 ──────────────────────────────
        self.wake_word = WakeWordDetector(self.cfg)

        # ── 오디오 초기화 (실패해도 계속 동작) ───────────
        self.audio_available = False
        self.stt_available = False

        try:
            import speech_recognition as sr
            self.sr = sr
            with suppress_stderr():
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self.recognizer.energy_threshold = speech_cfg.get('energy_threshold', 3000)
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = speech_cfg.get('pause_threshold', 1.2)
            self.stt_available = True
            self.get_logger().info("[STT] 마이크 초기화 완료")
        except Exception as e:
            self.get_logger().warn(f"[STT] 마이크 초기화 실패 (텍스트 모드로 동작): {e}")

        try:
            import pygame
            self.pygame = pygame
            with suppress_stderr():
                pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)
            self.audio_available = True
            self.get_logger().info("[TTS] 스피커 초기화 완료")
        except Exception as e:
            self.get_logger().warn(f"[TTS] 스피커 초기화 실패 (텍스트 출력으로 동작): {e}")

        self.language = speech_cfg.get('language', 'ko-KR')

        # ── 상태 변수 ─────────────────────────────────────
        self.is_speaking = False
        self.interrupt_flag = False
        self.listening = True
        self.speech_queue = queue.Queue()
        self.tts_queue = queue.Queue()

        # ── ROS2 구독/발행 ────────────────────────────────
        # tts_input: LLM→A 방향 (node_b_server의 pub_sound 토픽과 일치해야 함)
        tts_topic = topics.get('tts_input', '/cango_master/llm2sound')
        self.create_subscription(
            SoundRequest, tts_topic, self.on_tts_input, 10
        )
        # tts_stop: node_b에서 STT 수신 시 TTS 중단 신호
        tts_stop_topic = topics.get('tts_stop', '/cango_master/tts_stop')
        self.create_subscription(
            String, tts_stop_topic, self.on_tts_stop, 10
        )
        self.pub_stt = self.create_publisher(String, topics['stt_result'], 10)

        # ── 백그라운드 스레드 시작 ────────────────────────
        if self.stt_available:
            self.get_logger().info("배경 소음 측정 중...")
            try:
                with suppress_stderr():
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.get_logger().info("배경 소음 측정 완료")
            except Exception as e:
                self.get_logger().warn(f"배경 소음 측정 실패: {e}")
                self.stt_available = False

            if self.stt_available:
                threading.Thread(target=self._always_listen, daemon=True).start()
                threading.Thread(target=self._process_speech, daemon=True).start()

        threading.Thread(target=self._process_tts, daemon=True).start()

        mode = "Wake Word 모드" if self.wake_word.enabled else "항상 듣기 모드"
        audio_mode = "음성" if self.audio_available else "텍스트(스피커 없음)"
        stt_mode = "마이크" if self.stt_available else "토픽 수신만"
        self.get_logger().info(f"Voice Client 준비 완료 | {mode} | TTS:{audio_mode} | STT:{stt_mode}")

    # ── TTS 수신 ──────────────────────────────────────────

    def on_tts_input(self, msg: SoundRequest):
        """LLM→A 방향 TTS 수신. ordered_num=4(커스텀 텍스트)만 처리"""
        if not msg.request:
            return
        if msg.ordered_num != 4:
            return
        text = msg.text.strip()
        if text:
            self.tts_queue.put(text)

    def on_tts_stop(self, msg: String):
        """node_b에서 STT 수신 시 TTS 즉시 중단"""
        if self.is_speaking:
            self.get_logger().info("[TTS 중단 신호 수신]")
            self.interrupt_flag = True
            # tts_queue 비우기
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                except:
                    break

    # ── TTS 처리 스레드 ───────────────────────────────────

    def _process_tts(self):
        while self.listening:
            try:
                text = self.tts_queue.get(timeout=0.5)
                self._speak(text)
            except queue.Empty:
                continue

    def _speak(self, text: str):
        # 마이크/스피커 없어도 항상 터미널에 출력
        self.get_logger().info(f"[음성 출력] {text}")
        print(f"\n▶ TTS: {text}\n")

        if not self.audio_available:
            return

        self.is_speaking = True
        self.interrupt_flag = False

        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='ko', slow=False)
            tts.save("/tmp/tts_out.mp3")

            with suppress_stderr():
                self.pygame.mixer.music.load("/tmp/tts_out.mp3")
                self.pygame.mixer.music.play()

                while self.pygame.mixer.music.get_busy():
                    self.pygame.time.Clock().tick(10)
                    if self.interrupt_flag:
                        self.pygame.mixer.music.stop()
                        self.get_logger().info("[TTS 중단됨]")
                        break

            time.sleep(0.2)
        except Exception as e:
            self.get_logger().error(f"TTS 재생 오류: {e}")
        finally:
            self.is_speaking = False

    # ── 항상 듣기 스레드 ──────────────────────────────────

    def _always_listen(self):
        self.get_logger().info("[마이크] 항상 듣는 중...")
        with suppress_stderr():
            with self.microphone as source:
                while self.listening:
                    try:
                        audio = self.recognizer.listen(
                            source, timeout=1, phrase_time_limit=6
                        )
                        threading.Thread(
                            target=self._recognize,
                            args=(audio,),
                            daemon=True
                        ).start()
                    except self.sr.WaitTimeoutError:
                        continue
                    except Exception:
                        time.sleep(0.1)

    def _recognize(self, audio):
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            text = text.strip()
            if text:
                self.speech_queue.put(text)
        except Exception:
            pass

    # ── STT 처리 스레드 ───────────────────────────────────

    def _process_speech(self):
        while self.listening:
            try:
                text = self.speech_queue.get(timeout=0.1)
                self.get_logger().info(f"[STT 인식] '{text}'")

                is_emergency = any(
                    w in text.lower() for w in self.emergency_keywords
                )

                if is_emergency:
                    if self.is_speaking:
                        self.get_logger().info("[긴급 중단]")
                        self.interrupt_flag = True
                        while not self.tts_queue.empty():
                            try:
                                self.tts_queue.get_nowait()
                            except queue.Empty:
                                break
                        time.sleep(0.2)
                    self._publish_stt(text)
                    continue

                if not self.wake_word.should_process(text):
                    continue

                clean_text = self.wake_word.get_clean_text(text)
                if clean_text:
                    self._publish_stt(clean_text)

            except queue.Empty:
                continue

    def _publish_stt(self, text: str):
        msg = String()
        msg.data = text
        self.pub_stt.publish(msg)
        self.get_logger().info(f"[STT→B] '{text}'")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.wake_word.print_stats()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
