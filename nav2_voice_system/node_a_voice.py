"""
NUC용 통합 노드 - 음성 입출력 + 마스터 중계

역할:
1. 음성 입출력 (STT/TTS)
2. 마스터 ↔ 노트북B 토픽 중계 (ROS2 DDS ↔ rosbridge WebSocket)

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
from cango_msgs.msg import LlmRequest, SoundRequest

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from contextlib import contextmanager
import roslibpy

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


class NucNode(Node):
    def __init__(self):
        super().__init__('nuc_node')

        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'system_config.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        topics = self.cfg['topics']
        speech_cfg = self.cfg.get('speech', {})
        self.emergency_keywords = self.cfg.get('emergency_keywords', ['정지', '멈춰'])
        self.language = speech_cfg.get('language', 'ko-KR')

        self.wake_word = WakeWordDetector(self.cfg)

        # ── 상태 변수 ─────────────────────────────────────
        self.is_speaking = False
        self.interrupt_flag = False
        self.listening = True
        self.speech_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.initialized = False  # 마스터에서 첫 위치 받았는지

        # ── 오디오 초기화 ─────────────────────────────────
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
            self.get_logger().warn(f"[STT] 마이크 초기화 실패: {e}")

        try:
            import pygame
            self.pygame = pygame
            with suppress_stderr():
                pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)
            self.audio_available = True
            self.get_logger().info("[TTS] 스피커 초기화 완료")
        except Exception as e:
            self.get_logger().warn(f"[TTS] 스피커 초기화 실패: {e}")

        # ── rosbridge 연결 ────────────────────────────────
        ros_host = self.cfg.get('rosbridge', {}).get('host', '100.95.77.76')
        ros_port = self.cfg.get('rosbridge', {}).get('port', 9090)

        self.get_logger().info(f"[rosbridge] {ros_host}:{ros_port} 연결 중...")
        self.ros_client = roslibpy.Ros(host=ros_host, port=ros_port)
        self.ros_client.run()

        if self.ros_client.is_connected:
            self.get_logger().info("[rosbridge] 연결 완료!")
        else:
            self.get_logger().error("[rosbridge] 연결 실패!")
            return

        # ── rosbridge 토픽 (노트북B 방향) ─────────────────
        # NUC → 노트북B
        self.rb_pub_stt = roslibpy.Topic(
            self.ros_client, topics['stt_result'], 'std_msgs/String'
        )
        self.rb_pub_master2llm = roslibpy.Topic(
            self.ros_client, topics['master2llm'], 'cango_msgs/LlmRequest'
        )
        self.rb_pub_sound_trigger = roslibpy.Topic(
            self.ros_client, topics['sound_trigger'], 'cango_msgs/SoundRequest'
        )

        # 노트북B → NUC
        rb_sub_llm2master = roslibpy.Topic(
            self.ros_client, topics['llm2master'], 'cango_msgs/LlmRequest'
        )
        rb_sub_llm2master.subscribe(self.on_rb_llm2master)

        rb_sub_tts = roslibpy.Topic(
            self.ros_client, topics['tts_input'], 'cango_msgs/SoundRequest'
        )
        rb_sub_tts.subscribe(self.on_rb_tts)

        rb_sub_stop = roslibpy.Topic(
            self.ros_client, topics['tts_stop'], 'std_msgs/String'
        )
        rb_sub_stop.subscribe(self.on_rb_tts_stop)

        rb_sub_sound2ui = roslibpy.Topic(
            self.ros_client, '/cango/sound2ui', 'cango_msgs/SoundRequest'
        )
        rb_sub_sound2ui.subscribe(self.on_rb_sound2ui)

        # ── ROS2 토픽 (마스터 방향) ───────────────────────
        # 마스터 → NUC
        self.create_subscription(
            LlmRequest, topics['master2llm'], self.on_master2llm, 10
        )
        self.create_subscription(
            SoundRequest, topics['sound_trigger'], self.on_master_sound, 10
        )
        self.create_subscription(
            String, '/cango/llm_ui_text', self.on_ui_text, 10
        )

        # NUC → 마스터
        self.pub_llm2master = self.create_publisher(
            LlmRequest, topics['llm2master'], 10
        )
        self.pub_tts_stop = self.create_publisher(
            String, topics['tts_stop'], 10
        )
        self.pub_sound2ui = self.create_publisher(
            SoundRequest, '/cango/sound2ui', 10
        )

        self.get_logger().info("[rosbridge] 토픽 설정 완료")

        # ── STT 스레드 시작 ───────────────────────────────
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
        self.get_logger().info(f"NUC 노드 준비 완료 | {mode}")

    # ── 마스터 → rosbridge(노트북B) 중계 ─────────────────

    def on_master2llm(self, msg: LlmRequest):
        """마스터에서 받은 LlmRequest → rosbridge로 노트북B에 전달"""
        # 첫 위치 수신 시 초기화 완료 표시
        if not self.initialized and msg.local_candi1:
            self.get_logger().info(f"[초기 위치] 마스터에서 수신: {msg.local_candi1}")
            self.initialized = True

        self.rb_pub_master2llm.publish(roslibpy.Message({
            'request': msg.request,
            'local_candi1': msg.local_candi1,
            'local_candi2': msg.local_candi2,
            'goalpoint': msg.goalpoint,
            'waypoints': list(msg.waypoints),
            'user_start': msg.user_start,
            'user_interrupt': msg.user_interrupt,
            'user_finish': msg.user_finish,
            'map_search': msg.map_search,
            'stand': msg.stand if hasattr(msg, 'stand') else False,
        }))

    def on_master_sound(self, msg: SoundRequest):
        """마스터에서 받은 SoundRequest(ordered_num=1,2,3) → rosbridge로 노트북B 전달"""
        if msg.ordered_num == 4:
            return  # TTS 텍스트는 노트북B→NUC 방향
        self.rb_pub_sound_trigger.publish(roslibpy.Message({
            'request': msg.request,
            'ordered_num': msg.ordered_num,
            'text': msg.text,
            'user': msg.user if hasattr(msg, 'user') else '',
            'llm': msg.llm if hasattr(msg, 'llm') else '',
        }))

    # ── rosbridge(노트북B) → 마스터 중계 ─────────────────

    def on_rb_llm2master(self, msg):
        """노트북B에서 받은 LlmRequest → ROS2로 마스터에 전달"""
        ros_msg = LlmRequest()
        ros_msg.request = msg.get('request', False)
        ros_msg.local_candi1 = msg.get('local_candi1', '')
        ros_msg.local_candi2 = msg.get('local_candi2', '')
        ros_msg.goalpoint = msg.get('goalpoint', '')
        ros_msg.waypoints = msg.get('waypoints', [])
        ros_msg.user_start = msg.get('user_start', False)
        ros_msg.user_interrupt = msg.get('user_interrupt', False)
        ros_msg.user_finish = msg.get('user_finish', False)
        ros_msg.map_search = msg.get('map_search', 0)
        self.pub_llm2master.publish(ros_msg)

    def on_rb_tts(self, msg):
        """노트북B에서 받은 TTS(ordered_num=4) → 재생"""
        if not msg.get('request', False):
            return
        if msg.get('ordered_num', 0) != 4:
            return
        text = msg.get('text', '').strip()
        if text:
            self.get_logger().info(f"[음성 출력] {text}")
            self.tts_queue.put(text)

    def on_rb_tts_stop(self, msg):
        """노트북B에서 TTS 중단 신호 수신"""
        if self.is_speaking:
            self.get_logger().info("[TTS 중단 신호 수신]")
            self.interrupt_flag = True
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                except:
                    break

    # ── TTS 처리 ──────────────────────────────────────────

    def _process_tts(self):
        while self.listening:
            try:
                text = self.tts_queue.get(timeout=0.5)
                self._speak(text)
            except queue.Empty:
                continue

    def _speak(self, text: str):
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

    # ── STT ───────────────────────────────────────────────

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

    def _process_speech(self):
        while self.listening:
            try:
                text = self.speech_queue.get(timeout=0.1)
                self.get_logger().info(f"[STT 인식] '{text}'")

                is_emergency = any(w in text.lower() for w in self.emergency_keywords)

                if is_emergency:
                    if self.is_speaking:
                        self.interrupt_flag = True
                        while not self.tts_queue.empty():
                            try:
                                self.tts_queue.get_nowait()
                            except queue.Empty:
                                break
                        time.sleep(0.2)
                        # TTS 중단 신호를 마스터에도 전달
                        stop_msg = String()
                        stop_msg.data = 'stop'
                        self.pub_tts_stop.publish(stop_msg)
                    self._publish_stt(text)
                    continue

                if not self.wake_word.should_process(text):
                    continue

                clean_text = self.wake_word.get_clean_text(text)
                if clean_text:
                    self._publish_stt(clean_text)

            except queue.Empty:
                continue

    def on_rb_sound2ui(self, msg):
        """노트북B에서 받은 sound2ui → ROS2로 UI에 relay"""
        self.get_logger().info(f"[SOUND2UI RAW] {msg}")
        if not msg.get('request', False):
            return
        ros_msg = SoundRequest()
        ros_msg.request = True
        ros_msg.ordered_num = int(msg.get('ordered_num', 4))
        ros_msg.text = str(msg.get('text', '') or '')
        ros_msg.user = str(msg.get('user', '') or '')
        ros_msg.llm = str(msg.get('llm', '') or '')
        self.get_logger().info(f"[SOUND2UI→UI] user='{ros_msg.user[:20]}' llm='{ros_msg.llm[:20]}'")
        self.pub_sound2ui.publish(ros_msg)

    def on_ui_text(self, msg: String):
        """UI 텍스트 입력 → STT와 동일하게 처리"""
        text = msg.data.strip()
        if not text:
            return
        self.get_logger().info(f"[UI 입력] '{text}'")
        is_emergency = any(w in text.lower() for w in self.emergency_keywords)
        if is_emergency and self.is_speaking:
            self.interrupt_flag = True
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                except queue.Empty:
                    break
        self._publish_stt(text)

    def _publish_stt(self, text: str):
        self.rb_pub_stt.publish(roslibpy.Message({'data': text}))
        self.get_logger().info(f"[STT→B] '{text}'")


def main(args=None):
    rclpy.init(args=args)
    node = NucNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.wake_word.print_stats()
        node.ros_client.terminate()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
