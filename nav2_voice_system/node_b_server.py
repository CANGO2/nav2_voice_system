"""
노트북 B (LLM 노드) - cango_master 연동 버전

[변경사항]
- Navigation.msg 직접 구독으로 실시간 좌표 수신
- candi1/candi2 기반 위치 추정으로 방향 계산 (fallback)
- ordered_num=3 도착 시 벡터 외적으로 방향 계산 후 SoundRequest.text 채움
- goalpoint feature 이름 → node_id 자동 변환
- waypoints 직접 계산해서 마스터로 전송
"""

import os
import sys
import time
import yaml
import math
import rclpy
from rclpy.node import Node

sys.path.insert(0, os.path.dirname(__file__))

from map_manager import MapManager
from llm_processor import LLMProcessor
from wake_word_detector import WakeWordDetector

from cango_msgs.msg import LlmRequest, SoundRequest, Navigation
from std_msgs.msg import String

STATE_WAITING_DESTINATION = 'waiting_destination'
STATE_WAITING_CONFIRM     = 'waiting_confirm'
STATE_NAVIGATING          = 'navigating'
STATE_WAITING_ARRIVAL     = 'waiting_arrival'

STATE_KR = {
    STATE_WAITING_DESTINATION: '목적지 대기',
    STATE_WAITING_CONFIRM:     '출발 확인 대기',
    STATE_NAVIGATING:          '이동 중',
    STATE_WAITING_ARRIVAL:     '도착 후 대기',
}


class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'system_config.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        topics = self.cfg['topics']

        map_file = self.cfg['navigation'].get('map_file') or None
        self.map = MapManager(map_file)
        self.llm = LLMProcessor(self.map, self.cfg)
        self.wake_word = WakeWordDetector(self.cfg)

        # ── 상태 변수 ─────────────────────────────────────
        self.state = STATE_WAITING_DESTINATION
        self.candi1 = self.cfg['navigation'].get('start_position', 'hall_203')
        self.candi2 = self.cfg['navigation'].get('start_position', 'hall_203')
        self.prev_candi1 = self.cfg['navigation'].get('start_position', 'hall_203')
        self.goalpoint = ''
        self.waypoints = []

        # 실시간 로봇 좌표 (Navigation.msg에서 업데이트)
        # fallback: candi 기반 추정값 사용
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.has_real_coords = False  # 실제 좌표 수신 여부

        # 전체 경로 저장 (방향 안내용)
        self.full_path = []
        self.current_path_idx = 0

        # ── ROS2 구독 ─────────────────────────────────────
        self.create_subscription(LlmRequest, topics['master2llm'], self.on_master_msg, 10)
        self.create_subscription(String, topics['stt_result'], self.on_stt, 10)
        self.create_subscription(SoundRequest, topics['sound_trigger'], self.on_sound_trigger, 10)

        # SLAM 실시간 좌표 구독
        navi_topic = topics.get('navi2master', '/cango_master/navi2master')
        self.create_subscription(Navigation, navi_topic, self.on_navigation, 10)

        # ── ROS2 발행 ─────────────────────────────────────
        self.pub_llm = self.create_publisher(LlmRequest, topics['llm2master'], 10)
        tts_stop_topic = topics.get('tts_stop', '/cango_master/tts_stop')
        self.pub_tts_stop = self.create_publisher(String, tts_stop_topic, 10)
        self.pub_sound = self.create_publisher(SoundRequest, topics['tts_input'], 10)

        self.create_timer(1.0, self.publish_status)

        self.get_logger().info("LLM Node 준비 완료")
        time.sleep(2.0)
        self.speak_custom("안녕하세요. 어디로 가실 건가요?")

    # ── Navigation 수신 (실시간 좌표) ─────────────────────

    def on_navigation(self, msg: Navigation):
        """SLAM 실시간 좌표 업데이트"""
        self.robot_x = msg.current_location.x
        self.robot_y = msg.current_location.y
        self.robot_yaw = self.map.get_robot_yaw(
            msg.current_location.x,
            msg.current_location.y,
            msg.current_location.w
        )
        self.has_real_coords = True

    # ── master 메시지 수신 ────────────────────────────────

    def on_master_msg(self, msg: LlmRequest):
        if msg.local_candi1:
            self.candi1 = msg.local_candi1
        if msg.local_candi2:
            self.candi2 = msg.local_candi2
        if msg.goalpoint:
            self.goalpoint = msg.goalpoint

        if msg.map_search == 2:
            self.get_logger().info("[경로 확인] 가능 → 출발 확인 요청")
            dest_name = self.map.get_display_name(self.goalpoint)
            self.speak_custom(f"{dest_name}로 안내하겠습니다. 출발할까요?")
            self.state = STATE_WAITING_CONFIRM

        elif msg.map_search == 3:
            self.get_logger().info("[경로 확인] 불가능")
            self.speak_custom("죄송합니다. 해당 경로로는 이동할 수 없습니다. 다른 목적지를 말씀해주세요")
            self.state = STATE_WAITING_DESTINATION

    # ── STT 수신 ──────────────────────────────────────────

    def on_stt(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        is_emergency = any(w in text.lower() for w in self.cfg.get('emergency_keywords', []))
        if not is_emergency and not self.wake_word.should_process(text):
            return

        clean_text = self.wake_word.get_clean_text(text)
        if not clean_text:
            return

        # STT 수신 즉시 TTS 중단 신호 발행
        stop_msg = String()
        stop_msg.data = 'stop'
        self.pub_tts_stop.publish(stop_msg)

        state_kr = STATE_KR.get(self.state, self.state)
        self.get_logger().info(f"[STT] '{clean_text}' | 상태: {state_kr}")

        if self.state == STATE_NAVIGATING:
            self._handle_navigating(clean_text)
        else:
            self._handle_conversation(clean_text)

    # ── SLAM 트리거 수신 ──────────────────────────────────

    def on_sound_trigger(self, msg: SoundRequest):
        if not msg.request:
            return
        if msg.ordered_num == 4:  # 자기가 발행한 TTS는 무시
            return
        if msg.ordered_num == 3 and self.state == STATE_WAITING_ARRIVAL:
            return  # 도착 처리 후 중복 트리거 무시
        if self.state not in (STATE_NAVIGATING,):
            if msg.ordered_num in (1, 2):
                return  # 이동 중이 아닐 때 위치 안내 무시

        if msg.ordered_num == 1:
            # 위치 안내: 실제 좌표 또는 candi 기반 추정
            x, y, yaw = self._get_current_coords()

            # 다음 waypoint 찾기
            next_node = self._get_next_waypoint()

            # full_path를 llm에 전달해서 경로 기반 방향 계산 가능하게
            self.llm._current_full_path = self.full_path
            guide_text = self.llm.generate_location_guide(
                self.candi1, x, y, yaw, next_node
            )
            self.speak_custom(guide_text)

            # 경로 인덱스 진행
            if self.current_path_idx < len(self.full_path) - 1:
                self.current_path_idx += 1

        elif msg.ordered_num == 2:
            dest_name = self.map.get_display_name(self.goalpoint)
            self.speak_custom(f"{dest_name} 부근입니다")

        elif msg.ordered_num == 3:
            # 도착: 벡터 외적으로 방향 계산 후 SoundRequest에 채워서 발행
            dest_name = self.map.get_display_name(self.goalpoint)
            direction = self._calc_destination_direction()
            self.get_logger().info(f"[도착] {dest_name} | 방향: {direction}")

            self.get_logger().info(f"[TTS] 목적지에 도착했습니다. {direction}에 {dest_name}이 있습니다")
            self.speak_custom(f"목적지에 도착했습니다. {direction}에 {dest_name}이 있습니다")

            self.state = STATE_WAITING_ARRIVAL
            self.speak_custom("더 필요한 것이 있으신가요?")

    # ── 대화 처리 (1단계 목적지 대화) ────────────────────

    def _handle_conversation(self, text: str):
        result = self.llm.analyze_destination(text, self.candi1, self.candi2, self.state)
        intent = result.get('intent', 'unknown')
        goalpoint = result.get('goalpoint')
        response = result.get('response', '')
        unavailable = result.get('unavailable', False)
        map_search = result.get('map_search', 0)

        self.get_logger().info(f"[의도] {intent} | goalpoint={goalpoint}")

        if unavailable:
            reason = result.get('reason', '')
            organized = self.llm.get_organized_destinations(
                text, unavailable=True, unavailable_dest=reason)
            self.speak_custom(organized)
            return

        if intent == 'destination_list':
            organized = self.llm.get_organized_destinations(text)
            self.speak_custom(organized)
            return

        if intent == 'where_am_i':
            loc_desc = self.map.get_location_description(self.candi1, self.candi2)
            self.speak_custom(f"현재 {loc_desc} 입니다")
            return

        if intent == 'end_navigation':
            self.speak_custom("안내를 종료합니다. 이용해 주셔서 감사합니다")
            self.state = STATE_WAITING_DESTINATION
            self.goalpoint = ''
            self.waypoints = []
            self._publish_llm(user_finish=True)
            return

        if intent in ('set_destination', 'change_destination') and goalpoint:
            goalpoint = self._resolve_goalpoint(goalpoint)
            self.goalpoint = goalpoint
            start_node = self.candi1 or self.cfg['navigation'].get('start_position', 'hall_203')
            self.waypoints = self._calc_waypoints(start_node, goalpoint)

            if map_search == 1:
                self._publish_llm(goalpoint=goalpoint, waypoints=self.waypoints, map_search=True)
            else:
                dest_name = self.map.get_display_name(goalpoint)
                self.speak_custom(f"{dest_name}로 안내하겠습니다. 출발할까요?")
                self.state = STATE_WAITING_CONFIRM
            return

        if self.state == STATE_WAITING_CONFIRM:
            if intent == 'confirm_yes':
                self._start_navigation()
            elif intent == 'confirm_no':
                self.speak_custom("다시 목적지를 말씀해주세요")
                self.state = STATE_WAITING_DESTINATION
            else:
                self.speak_custom("출발할까요?")
            return

        if self.state == STATE_WAITING_ARRIVAL:
            if intent in ('arrival_response_no', 'confirm_no'):
                self.speak_custom("안내를 종료합니다. 안전한 하루 되세요")
                self.state = STATE_WAITING_DESTINATION
                self._publish_llm(user_finish=True)
            elif intent in ('arrival_response_yes', 'confirm_yes'):
                self.speak_custom("어디로 가실 건가요?")
                self.state = STATE_WAITING_DESTINATION
            else:
                self.speak_custom("더 필요한 것이 있으신가요?")
            return

        if response:
            self.speak_custom(response)
        elif intent == 'unknown':
            self.speak_custom("목적지를 다시 말씀해주세요")

    # ── 이동 중 사용자 말걸기 처리 (3단계) ───────────────

    def _handle_navigating(self, text: str):
        result = self.llm.analyze_intent(text, state=self.state)
        intent = result.get('intent', 'unknown')
        user_interrupt = result.get('user_interrupt', False)
        user_finish = result.get('user_finish', False)
        response = result.get('response', '')
        new_dest = result.get('new_destination')

        self.get_logger().info(f"[이동중 의도] {intent}")

        if intent == 'stop' or user_interrupt:
            self.speak_custom(response or "잠시 멈추겠습니다")
            self._publish_llm(user_interrupt=True)
            return

        if intent == 'finish' or user_finish:
            self.speak_custom(response or "안내를 종료합니다")
            self.state = STATE_WAITING_DESTINATION
            self._publish_llm(user_finish=True)
            return

        if intent == 'resume':
            self.speak_custom(response or "다시 출발합니다")
            self._publish_llm(user_start=True)
            return

        if intent == 'where_am_i':
            loc_desc = self.map.get_location_description(self.candi1, self.candi2)
            self.speak_custom(f"현재 {loc_desc} 입니다")
            return

        if intent == 'change_destination' and new_dest:
            node_id = self._resolve_goalpoint(new_dest)
            if node_id:
                self.goalpoint = node_id
                self.waypoints = self._calc_waypoints(self.candi1, node_id)
                dest_name = self.map.get_display_name(node_id)
                self.speak_custom(f"목적지를 {dest_name}으로 변경합니다")
                self._publish_llm(goalpoint=node_id, waypoints=self.waypoints, map_search=True)
            return

        if response:
            self.speak_custom(response)

    # ── 출발 확정 ─────────────────────────────────────────

    def _start_navigation(self):
        dest_name = self.map.get_display_name(self.goalpoint)
        self.speak_custom("안내를 시작합니다")
        self.state = STATE_NAVIGATING
        self.current_path_idx = 0
        self._publish_llm(user_start=True)
        self.get_logger().info(f"[안내 시작] {dest_name} ({self.goalpoint})")
        self.get_logger().info(f"[전체 waypoints] {self.waypoints}")

        # 전체 경로 요약 안내 (LLM으로 생성)
        if self.full_path and len(self.full_path) > 2:
            route_summary = self.llm.generate_route_summary(self.full_path)
            if route_summary:
                self.speak_custom(route_summary)

    # ── 헬퍼 함수들 ───────────────────────────────────────

    def _get_next_waypoint(self) -> str:
        """현재 경로에서 다음 waypoint 반환"""
        # full_path에서 candi1과 가장 가까운 위치 찾기
        if not self.full_path:
            return None

        # candi1이 full_path에 있으면 그 다음 노드
        if self.candi1 in self.full_path:
            idx = self.full_path.index(self.candi1)
            if idx + 1 < len(self.full_path):
                return self.full_path[idx + 1]

        # current_path_idx 기반
        if self.current_path_idx + 1 < len(self.full_path):
            return self.full_path[self.current_path_idx + 1]

        return None

    def _get_current_coords(self):
        """
        현재 좌표 반환
        실제 좌표가 있으면 그거 쓰고, 없으면 candi 기반 추정
        """
        if self.has_real_coords:
            return self.robot_x, self.robot_y, self.robot_yaw
        # fallback: candi1/candi2 노드 좌표로 추정
        x, y, yaw = self.map.estimate_position_from_candis(self.candi1, self.candi2, prev_candi=self.prev_candi1, full_path=self.full_path)
        self.get_logger().info(f"[좌표 추정] candi 기반 x={x:.1f} y={y:.1f} yaw={yaw:.2f}")
        return x, y, yaw

    def _calc_destination_direction(self) -> str:
        """
        목적지 feature의 방향을 벡터 외적으로 계산
        도착 시점: full_path의 마지막 이전 노드 → 마지막 노드 방향으로 yaw 계산
        """
        if not self.goalpoint:
            return "앞"

        # 실제 좌표 있으면 그거 사용
        if self.has_real_coords:
            x, y, yaw = self.robot_x, self.robot_y, self.robot_yaw
        else:
            # 도착 시점: full_path의 마지막 두 노드로 yaw 계산
            x, y = self.map.get_node_coords(self.goalpoint) or (0.0, 0.0)
            yaw = math.pi / 2  # 기본값
            if self.full_path and len(self.full_path) >= 2:
                prev_node = self.full_path[-2]
                curr_node = self.full_path[-1]
                prev_c = self.map.get_node_coords(prev_node)
                curr_c = self.map.get_node_coords(curr_node)
                if prev_c and curr_c:
                    dx = curr_c[0] - prev_c[0]
                    dy = curr_c[1] - prev_c[1]
                    if math.hypot(dx, dy) > 0.001:
                        yaw = math.atan2(dy, dx)
            self.get_logger().info(f"[좌표 추정] 도착 기반 x={x:.1f} y={y:.1f} yaw={yaw:.2f}")

        for feat in self.map.features:
            if feat.get('connected_node') == self.goalpoint:
                fc = feat.get('feature_coords')
                if fc:
                    direction = self.map.calc_feature_direction(x, y, yaw, fc)
                    return direction
        return "앞"

    def _resolve_goalpoint(self, goalpoint: str) -> str:
        """goalpoint가 feature 이름이면 node_id로 변환"""
        if not goalpoint:
            return goalpoint
        if goalpoint in self.map.nodes:
            return goalpoint
        node_id = self.map.find_node_by_name(goalpoint)
        if node_id:
            self.get_logger().info(f"[goalpoint 변환] '{goalpoint}' → '{node_id}'")
            return node_id
        return goalpoint

    def _calc_waypoints(self, start_node: str, goal_node: str) -> list:
        """경로 계산 후 waypoints 반환 + full_path 저장"""
        path = self.map.find_path(start_node, goal_node)
        if path:
            self.full_path = path
            self.current_path_idx = 0
            waypoints = self.map.path_to_waypoints(path)
            self.get_logger().info(f"[경로 계산] {start_node} → {goal_node}")
            self.get_logger().info(f"[전체 경로] {path}")
            self.get_logger().info(f"[Waypoints] {waypoints}")
            return waypoints
        else:
            self.full_path = [goal_node]
            self.current_path_idx = 0
            self.get_logger().warn(f"[경로 없음] {start_node} → {goal_node}")
            return [goal_node]

    def speak_custom(self, text: str):
        self.get_logger().info(f"[TTS] {text}")
        msg = SoundRequest()
        msg.request = True
        msg.ordered_num = 4
        msg.text = text
        self.pub_sound.publish(msg)

    def _publish_llm(self, goalpoint=None, waypoints=None, map_search=False,
                     user_start=False, user_interrupt=False, user_finish=False):
        msg = LlmRequest()
        msg.request = (self.state in [STATE_WAITING_DESTINATION, STATE_WAITING_CONFIRM])
        msg.local_candi1 = self.candi1
        msg.local_candi2 = self.candi2
        msg.goalpoint = goalpoint or self.goalpoint
        msg.waypoints = waypoints or self.waypoints
        msg.map_search = 1 if map_search else 0
        msg.user_start = user_start
        msg.user_interrupt = user_interrupt
        msg.user_finish = user_finish
        self.pub_llm.publish(msg)
        self.get_logger().info(
            f"[LLM→Master] start={user_start} interrupt={user_interrupt} "
            f"finish={user_finish} map_search={msg.map_search} "
            f"goalpoint={msg.goalpoint} waypoints={msg.waypoints}"
        )

    def publish_status(self):
        msg = LlmRequest()
        msg.request = (self.state in [STATE_WAITING_DESTINATION, STATE_WAITING_CONFIRM])
        msg.local_candi1 = self.candi1
        msg.local_candi2 = self.candi2
        msg.goalpoint = self.goalpoint
        msg.waypoints = self.waypoints
        msg.map_search = 0
        msg.user_start = False
        msg.user_interrupt = False
        msg.user_finish = False
        self.pub_llm.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.llm.print_stats()
        node.wake_word.print_stats()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
