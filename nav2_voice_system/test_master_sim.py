"""
테스트용 마스터 시뮬레이터
노트북 B에서 마이크로 테스트할 때 마스터 역할

사용법:
    cd ~/capstone
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 nav2_voice_system/nav2_voice_system/test_master_sim.py

키 조작:
    숫자 1~9  : 경로 노드 순서대로 위치 이동
    t         : ordered_num=1 트리거 (waypoint 도달)
    2         : ordered_num=2 트리거 (목적지 근처)
    3         : ordered_num=3 트리거 (목적지 도착)
    m         : map_search=2 전송 (경로 가능 응답)
    x         : map_search=3 전송 (경로 불가능 응답)
    p         : 현재 상태 출력
    q         : 종료
"""

import os
import sys
import rclpy
from rclpy.node import Node
from cango_msgs.msg import LlmRequest, SoundRequest
import threading
import time

# ── 테스트 시나리오 정의 ──────────────────────────────────────
# 원하는 경로로 바꿔서 쓰면 됨
SCENARIOS = {
    '1': {
        'name': '중앙홀 → 회의실1',
        'path': [
            'central_hall',
            'path_slope',
            'hall_205_mid',
            'hall_205',
            'hall_206',
            'hall_207',
            'hall_208',
            'hall_210',
            'hall_210_mid',
        ],
        'goal': 'hall_210_mid',
    },
    '2': {
        'name': 'CU편의점 → 화장실',
        'path': [
            'hall_203',
            'hall_204',
            'hall_205',
            'hall_206',
            'hall_207',
            'hall_208',
            'hall_toilet',
        ],
        'goal': 'hall_toilet',
    },
    '3': {
        'name': '화장실 → 엘리베이터',
        'path': [
            'hall_toilet',
            'hall_208',
            'hall_207',
            'hall_206',
            'hall_205',
            'hall_205_mid',
        ],
        'goal': 'hall_205_mid',
    },
}


class MasterSimulator(Node):
    def __init__(self):
        super().__init__('master_sim')

        self.pub_llm = self.create_publisher(
            LlmRequest, '/cango_master/master2llm', 10
        )
        self.pub_sound = self.create_publisher(
            SoundRequest, '/cango_master/master2sound', 10
        )

        # 현재 시나리오
        self.scenario = None
        self.path = []
        self.path_idx = 0
        self.goal = ''

        self.get_logger().info("마스터 시뮬레이터 준비 완료")

    def set_scenario(self, key: str):
        if key not in SCENARIOS:
            print(f"없는 시나리오: {key}")
            return
        s = SCENARIOS[key]
        self.scenario = s
        self.path = s['path']
        self.path_idx = 0
        self.goal = s['goal']
        print(f"\n[시나리오] {s['name']}")
        print(f"  경로: {' → '.join(self.path)}")
        self._send_position(self.path[0], self.path[0])

    def _send_position(self, candi1: str, candi2: str, map_search: int = 0, goalpoint: str = ''):
        msg = LlmRequest()
        msg.request = False
        msg.local_candi1 = candi1
        msg.local_candi2 = candi2
        msg.goalpoint = goalpoint or self.goal
        msg.waypoints = []
        msg.user_start = False
        msg.user_interrupt = False
        msg.user_finish = False
        msg.map_search = map_search
        self.pub_llm.publish(msg)
        print(f"  [위치] {candi1} / {candi2} | map_search={map_search}")

    def _send_trigger(self, ordered_num: int):
        msg = SoundRequest()
        msg.request = True
        msg.ordered_num = ordered_num
        msg.text = ''
        self.pub_sound.publish(msg)
        label = {1: 'waypoint 도달', 2: '목적지 근처', 3: '목적지 도착'}.get(ordered_num, '?')
        print(f"  [트리거] ordered_num={ordered_num} ({label})")

    def next_position(self):
        """경로를 한 칸 앞으로"""
        if not self.path:
            print("  시나리오가 설정 : s키로 선택하세요")
            return
        if self.path_idx < len(self.path) - 1:
            self.path_idx += 1
        curr = self.path[self.path_idx]
        prev = self.path[self.path_idx - 1] if self.path_idx > 0 else curr
        self._send_position(curr, prev)
        remaining = len(self.path) - 1 - self.path_idx
        print(f"  현재: {curr} | 이전: {prev} (남은: {remaining}개)")

    def send_map_available(self, available: bool = True):
        curr = self.path[self.path_idx] if self.path else 'hall_203'
        self._send_position(curr, curr, map_search=2 if available else 3)

    def print_status(self):
        if not self.path:
            print("  시나리오 미설정")
            return
        curr = self.path[self.path_idx] if self.path_idx < len(self.path) else '끝'
        print(f"\n  시나리오: {self.scenario['name']}")
        print(f"  현재 위치: {curr} ({self.path_idx+1}/{len(self.path)})")
        print(f"  전체 경로: {' → '.join(self.path)}")


def print_help():
    print("""
┌─────────────────────────────────────────┐
│         마스터 시뮬레이터 조작키          │
├─────────────────────────────────────────┤
│  s      : 시나리오 선택                  │
│  n      : 다음 위치로 이동               │
│  t      : ordered_num=1 (waypoint 도달)  │
│  nt     : 다음 위치 이동 + 트리거 한번에  │
│  2      : ordered_num=2 (목적지 근처)    │
│  3      : ordered_num=3 (목적지 도착)    │
│  m      : map_search=2 (경로 가능)       │
│  x      : map_search=3 (경로 불가능)     │
│  p      : 현재 상태 출력                 │
│  h      : 도움말                         │
│  q      : 종료                           │
└─────────────────────────────────────────┘
""")


def input_loop(sim: MasterSimulator):
    print_help()
    print("node_b_server가 실행 중이어야 함\n")

    while True:
        try:
            cmd = input(">> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'q':
            break
        elif cmd == 's':
            print("시나리오 선택:")
            for k, v in SCENARIOS.items():
                print(f"  {k}: {v['name']}")
            key = input("번호: ").strip()
            sim.set_scenario(key)
        elif cmd == 'n':
            sim.next_position()
        elif cmd == 't':
            sim._send_trigger(1)
        elif cmd == 'nt':
            sim.next_position()
            time.sleep(0.3)
            sim._send_trigger(1)
        elif cmd == '2':
            sim._send_trigger(2)
        elif cmd == '3':
            sim.next_position()
            time.sleep(0.3)
            sim._send_trigger(3)
        elif cmd == 'm':
            sim.send_map_available(True)
        elif cmd == 'x':
            sim.send_map_available(False)
        elif cmd == 'p':
            sim.print_status()
        elif cmd == 'h':
            print_help()
        elif cmd == '':
            pass
        else:
            print(f"  모르는 명령어: '{cmd}' (h: 도움말)")

    print("종료합니다")
    rclpy.shutdown()
    sys.exit(0)


def main():
    rclpy.init()
    sim = MasterSimulator()

    # ROS2 spin은 백그라운드 스레드
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim,), daemon=True)
    spin_thread.start()

    time.sleep(0.5)
    input_loop(sim)


if __name__ == '__main__':
    main()
