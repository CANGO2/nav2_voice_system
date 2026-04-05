"""
노트북 B (LLM 노드) - cango_master 연동 버전

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
        start_pos = self.cfg['navigation'].get('start_position', 'hall_203')
        self.candi1 = start_pos
        self.candi2 = start_pos
        # 직전 위치 추적 (방향 계산용)
        self.prev_candi1 = start_pos
        self.goalpoint = ''
        self.waypoints = []

        # 실시간 로봇 좌표
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.has_real_coords = False

        # 전체 경로 저장 (방향 안내용)
        self.full_path = []
        self.current_path_idx = 0

        # ── ROS2 구독 ─────────────────────────────────────
        self.create_subscription(LlmRequest, topics['master2llm'], self.on_master_msg, 10)
        self.create_subscription(String, topics['stt_result'], self.on_stt, 10)

        # sound_trigger: 마스터→LLM 방향 트리거 (tts_input과 다른 토픽)
        sound_trigger_topic = topics.get('sound_trigger', '/cango_master/master2llm_sound')
        self.create_subscription(SoundRequest, sound_trigger_topic, self.on_sound_trigger, 10)

        # SLAM 실시간 좌표
        navi_topic = topics.get('navi2master', '/cango_master/navi2master')
        self.create_subscription(Navigation, navi_topic, self.on_navigation, 10)

        # ── ROS2 발행 ─────────────────────────────────────
        self.pub_llm = self.create_publisher(LlmRequest, topics['llm2master'], 10)
        # tts_input: LLM→노드A 방향만 (마스터 수신 안 함)
        tts_topic = topics.get('tts_input', '/cango_master/llm2sound')
        self.pub_sound = self.create_publisher(SoundRequest, tts_topic, 10)
        # tts_stop: TTS 중단 신호 (node_b → node_a)
        tts_stop_topic = topics.get('tts_stop', '/cango_master/tts_stop')
        self.pub_tts_stop = self.create_publisher(String, tts_stop_topic, 10)

        self.create_timer(1.0, self.publish_status)

        self.get_logger().info("LLM Node 준비 완료")
        time.sleep(2.0)
        self.speak_custom("안녕하세요. 어디로 가실 건가요?")

    # ── Navigation 수신 (실시간 좌표) ─────────────────────

    def on_navigation(self, msg: Navigation):
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
        # 위치 업데이트
        if msg.local_candi1:
            # candi1이 바뀌면 이전 candi1을 prev로 저장
            if msg.local_candi1 != self.candi1:
                self.prev_candi1 = self.candi1
            self.candi1 = msg.local_candi1
        if msg.local_candi2:
            # candi2는 마스터가 보내는 두 번째 후보 노드
            # 시뮬레이터: candi2=이전노드 / 실제 마스터: candi2=인접 노드
            # prev_candi1이 없을 때 fallback으로 사용
            self.candi2 = msg.local_candi2
            if self.prev_candi1 == self.candi1 and msg.local_candi2 != msg.local_candi1:
                self.prev_candi1 = msg.local_candi2
        if msg.goalpoint:
            self.goalpoint = msg.goalpoint

        # map_search=2,3은 waiting_destination 상태일 때만 처리
        # 마스터가 1Hz로 계속 보내기 때문에 중복 처리 방지
        if msg.map_search == 2 and self.state == STATE_WAITING_DESTINATION:
            self.get_logger().info("[경로 확인] 가능 → 출발 확인 요청")
            dest_name = self.map.get_display_name(self.goalpoint)
            self.speak_custom(f"{dest_name}로 안내하겠습니다. 출발할까요?")
            self.state = STATE_WAITING_CONFIRM

        elif msg.map_search == 3 and self.state == STATE_WAITING_DESTINATION:
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

        # STT 수신 즉시 TTS 중단 신호 발행 (node_a_voice가 받아서 재생 중단)
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
        """마스터→LLM 방향 트리거만 처리"""
        if not msg.request:
            return
        # 자기가 발행한 ordered_num=4는 무시
        if msg.ordered_num == 4:
            return

        if msg.ordered_num == 1:
            # 이동 중 상태일 때만 처리
            if self.state != STATE_NAVIGATING:
                return

            import math

            curr_node = self.candi1
            next_node = self._get_next_waypoint()

            # 진입 방향: 이전 노드 → 현재 노드 (주변 시설 방향 계산용)
            # prev_candi1이 있으면 그걸 쓰고, 없으면 full_path에서 찾기
            arrival_yaw = None
            prev_node = self.prev_candi1 if self.prev_candi1 != curr_node else None
            if not prev_node and curr_node in self.full_path:
                idx = self.full_path.index(curr_node)
                if idx > 0:
                    prev_node = self.full_path[idx - 1]

            if prev_node:
                pc = self.map.get_node_coords(prev_node)
                cc = self.map.get_node_coords(curr_node)
                if pc and cc:
                    dx = cc[0] - pc[0]
                    dy = cc[1] - pc[1]
                    if math.hypot(dx, dy) > 0.001:
                        arrival_yaw = math.atan2(dy, dx)

            # 다음 방향: 현재 노드 → 다음 노드 (꺾을 방향 계산용)
            turn_yaw = None
            turn_str = "정보 없음"
            if next_node:
                cc = self.map.get_node_coords(curr_node)
                nc = self.map.get_node_coords(next_node)
                if cc and nc:
                    dx = nc[0] - cc[0]
                    dy = nc[1] - cc[1]
                    if math.hypot(dx, dy) > 0.001:
                        turn_yaw = math.atan2(dy, dx)
                # 꺾을 방향은 진입 yaw 기준 외적으로 계산
                if arrival_yaw is not None and turn_yaw is not None:
                    diff = turn_yaw - arrival_yaw
                    # -π ~ π 정규화
                    while diff > math.pi: diff -= 2*math.pi
                    while diff < -math.pi: diff += 2*math.pi
                    if abs(diff) < 0.35:   # 20도 이내
                        turn_str = "직진"
                    elif diff > 0:
                        turn_str = "왼쪽"
                    else:
                        turn_str = "오른쪽"
                elif turn_yaw is not None:
                    turn_str = "앞으로 이동"

            # 위치 안내는 진입 방향 yaw 기준으로 주변 시설 계산
            x, y = self.map.get_node_coords(curr_node) or (0.0, 0.0)
            yaw_for_guide = arrival_yaw if arrival_yaw is not None else (turn_yaw or math.pi/2)

            next_label = self.map.get_display_name(next_node) if next_node else ""
            next_direction_str = f"{turn_str}으로 이동 ({next_label} 방향)" if next_node else "정보 없음"

            self.get_logger().info(
                f"[방향] 진입={math.degrees(yaw_for_guide):.0f}° "
                f"다음={turn_str}({next_node})"
            )

            guide_text = self.llm.generate_location_guide(
                curr_node, x, y, yaw_for_guide, next_node,
                turn_str=turn_str
            )
            self.speak_custom(guide_text)

            if self.current_path_idx < len(self.full_path) - 1:
                self.current_path_idx += 1

        elif msg.ordered_num == 2:
            # 이동 중 상태일 때만 처리
            if self.state != STATE_NAVIGATING:
                return
            dest_name = self.map.get_display_name(self.goalpoint)
            self.speak_custom(f"{dest_name} 부근입니다")

        elif msg.ordered_num == 3:
            # 도착: STATE_NAVIGATING일 때만 1회 처리
            if self.state != STATE_NAVIGATING:
                return
            dest_name = self.map.get_display_name(self.goalpoint)
            direction = self._calc_destination_direction()
            self.get_logger().info(f"[도착] {dest_name} | 방향: {direction}")

            # 상태를 먼저 변경해서 중복 처리 차단
            self.state = STATE_WAITING_ARRIVAL

            # 마스터에 방향 정보 포함 SoundRequest 발행
            arrive_msg = SoundRequest()
            arrive_msg.request = True
            arrive_msg.ordered_num = 3
            arrive_msg.text = f"{direction}에 {dest_name}"
            self.pub_sound.publish(arrive_msg)
            self.get_logger().info(f"[TTS] 목적지에 도착했습니다. {direction}에 {dest_name}이 있습니다")

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
            # 검증: goalpoint가 실제 맵에 있는 노드인지 확인
            if goalpoint not in self.map.nodes:
                self.get_logger().warn(f"[goalpoint 검증 실패] '{goalpoint}' 맵에 없음")
                self.speak_custom("목적지를 찾을 수 없습니다. 다시 말씀해주세요")
                return

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
            if node_id and node_id in self.map.nodes:
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

        if self.full_path and len(self.full_path) > 2:
            route_summary = self.llm.generate_route_summary(self.full_path)
            if route_summary:
                self.speak_custom(route_summary)

    # ── 헬퍼 함수들 ───────────────────────────────────────

    def _get_next_waypoint(self) -> str:
        if not self.full_path:
            return None
        if self.candi1 in self.full_path:
            idx = self.full_path.index(self.candi1)
            if idx + 1 < len(self.full_path):
                return self.full_path[idx + 1]
        if self.current_path_idx + 1 < len(self.full_path):
            return self.full_path[self.current_path_idx + 1]
        return None

    def _get_current_coords(self):
        if self.has_real_coords:
            return self.robot_x, self.robot_y, self.robot_yaw
        x, y, yaw = self.map.estimate_position_from_candis(
            self.candi1, self.candi2,
            prev_candi=self.prev_candi1,
            full_path=self.full_path
        )
        self.get_logger().info(f"[좌표 추정] x={x:.1f} y={y:.1f} yaw={yaw:.2f}rad ({math.degrees(yaw):.0f}°)")
        return x, y, yaw

    def _calc_destination_direction(self) -> str:
        """
        도착 시 목적지 feature 방향을 계산
        진입 방향: prev_candi → candi1 벡터 사용
        """
        if not self.goalpoint:
            return "앞"

        if self.has_real_coords:
            x, y, yaw = self.robot_x, self.robot_y, self.robot_yaw
        else:
            # 도착 시: full_path 마지막 두 노드로 진입 방향 계산
            x, y = self.map.get_node_coords(self.goalpoint) or (0.0, 0.0)
            yaw = math.pi / 2
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
            self.get_logger().info(f"[도착 방향 추정] x={x:.1f} y={y:.1f} yaw={math.degrees(yaw):.0f}°")

        # goalpoint에 연결된 feature 중 우선순위 높은 것 찾기
        priority_types = ['화장실', '엘리베이터', '편의점', '출입구']
        target_feat = None
        for feat in self.map.features:
            if feat.get('connected_node') == self.goalpoint:
                if feat.get('type') in priority_types:
                    target_feat = feat
                    break
        if not target_feat:
            for feat in self.map.features:
                if feat.get('connected_node') == self.goalpoint:
                    target_feat = feat
                    break

        if target_feat:
            fc = target_feat.get('feature_coords')
            if fc:
                direction = self.map.calc_feature_direction(x, y, yaw, fc)
                return direction
        return "앞"

    def _resolve_goalpoint(self, goalpoint: str) -> str:
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
        """1Hz 상태 발행 - llm2master 토픽으로만 (tts/sound 토픽 건드리지 않음)"""
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
