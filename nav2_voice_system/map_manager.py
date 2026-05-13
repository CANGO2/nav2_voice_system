"""
맵 매니저 - nuri.json 기반 (새 구조 지원)

[변경사항]
- edges: from/to → start/end (둘 다 지원)
- features 없음: facility/room 타입 노드에서 시설 정보 추출
- label 없음: node id에서 자동으로 이름 생성
- range 필드 예외 처리 (open_space 타입)
"""

import os
import json
import math
import heapq
import re
from typing import Optional, Dict, List, Tuple


class MapManager:
    def __init__(self, map_file: str = None):
        if map_file is None:
            map_file = os.path.join(
                os.path.dirname(__file__), 'maps', 'nuri.json'
            )
        self.map_file = map_file
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []
        self.features: List[Dict] = []
        self.adjacency: Dict[str, List[str]] = {}
        self.feature_name_index: Dict[str, str] = {}
        self.feature_type_index: Dict[str, List[str]] = {}
        self._position_history: List[Tuple[float, float]] = []
        self._load()

    def _id_to_label(self, node_id: str) -> str:
        """node id → 시각장애인 안내에 적합한 자연스러운 한국어"""
        label_map = {
            'hall_left_top':        '왼쪽 복도 위쪽',
            'hall_left_mid':        '왼쪽 복도 중간',
            'hall_left_bottom':     '왼쪽 복도 아래',
            'hall_right_top':       '오른쪽 복도 위쪽',
            'hall_right_mid':       '오른쪽 복도 중간',
            'hall_right_bottom':    '오른쪽 복도 아래',
            'hall_left_top_end':    '왼쪽 위 끝',
            'hall_left_bottom_end': '왼쪽 아래 끝',
            'hall_right_top_end':   '오른쪽 위 끝',
            'hall_공터_center':     '공터 중앙',
            'hall_317': '317호 앞',
            'hall_318': '318호 앞',
            'hall_319': '319호 앞',
            'hall_320': '320호 앞',
            'hall_321': '321호 앞',
            'hall_322': '322호 앞',
            'hall_화장실_좌':     '왼쪽 화장실',
            'hall_화장실_우':     '오른쪽 화장실',
            'hall_엘리베이터_좌': '왼쪽 엘리베이터',
            'hall_엘리베이터_우': '오른쪽 엘리베이터',
            'hall_계단_좌':       '왼쪽 계단',
            'hall_계단_우':       '오른쪽 계단',
            'hall_산학협력단':    '산학협력단',
        }
        if node_id in label_map:
            return label_map[node_id]
        name = node_id
        if name.startswith('hall_'):
            name = name[5:]
        name = name.replace('_', ' ')
        return name

    def _load(self):
        with open(self.map_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for node in data.get('nodes', []):
            node_id = node['id']
            node_type = node.get('type', 'corridor')

            # coords = [x, y] 그대로 사용
            if 'range' in node:
                r = node['range']
                cx = (r[0][0] + r[1][0]) / 2
                cy = (r[0][1] + r[1][1]) / 2
                coords = [cx, cy]
            else:
                coords = list(node.get('coords', [0, 0]))

            label = node.get('label', self._id_to_label(node_id))

            self.nodes[node_id] = {
                'type': node_type,
                'coords': coords,
                'label': label,
            }

            # facility/room 타입은 features로도 등록
            if node_type in ('facility', 'room'):
                feat = {
                    'name': label,
                    'type': node_type,
                    'feature_coords': coords,
                    'connected_node': node_id,
                }
                self.features.append(feat)
                self.feature_name_index[label.lower()] = node_id
                self.feature_name_index[node_id.lower()] = node_id
                self.feature_type_index.setdefault(node_type, []).append(node_id)

                # 키워드 인덱싱 (부분 매칭용)
                for kw in label.split():
                    if len(kw) > 1:
                        self.feature_name_index.setdefault(kw.lower(), node_id)

        # edges 로드 (start/end 또는 from/to 둘 다 지원)
        for edge in data.get('edges', []):
            src = edge.get('start') or edge.get('from', '')
            dst = edge.get('end') or edge.get('to', '')
            if src and dst:
                self.edges.append({'from': src, 'to': dst})
                self.adjacency.setdefault(src, []).append(dst)
                self.adjacency.setdefault(dst, []).append(src)

        print(f"[MapManager] 로드 완료: 노드 {len(self.nodes)}개 / "
              f"엣지 {len(self.edges)}개 / 시설 {len(self.features)}개")

    def reload(self):
        self.nodes.clear()
        self.edges.clear()
        self.features.clear()
        self.adjacency.clear()
        self.feature_name_index.clear()
        self.feature_type_index.clear()
        self._load()

    # ── 기본 조회 ─────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self.nodes.get(node_id)

    def get_node_coords(self, node_id: str) -> Optional[Tuple[float, float]]:
        node = self.nodes.get(node_id)
        if node:
            c = node['coords']
            return (float(c[0]), float(c[1]))
        return None

    def get_display_name(self, node_id: str) -> str:
        node = self.nodes.get(node_id)
        if not node:
            return node_id
        return node.get('label', self._id_to_label(node_id))

    def get_all_destinations(self) -> List[str]:
        """목적지로 사용 가능한 시설/방 이름 목록"""
        excluded_keywords = ['접근금지', '금지', 'end']
        names = []
        for node_id, node in self.nodes.items():
            if node.get('type') in ('facility', 'room'):
                label = node.get('label', self._id_to_label(node_id))
                if any(k in label for k in excluded_keywords):
                    continue
                if label:
                    names.append(label)
        return names

    def find_node_by_name(self, query: str) -> Optional[str]:
        query_lower = query.lower().strip()

        # 정확 매칭
        if query_lower in self.feature_name_index:
            return self.feature_name_index[query_lower]

        # 포함 매칭
        for name, node_id in self.feature_name_index.items():
            if query_lower in name or name in query_lower:
                return node_id

        # 숫자 매칭
        nums = re.findall(r'\d+', query)
        for num in nums:
            for name, node_id in self.feature_name_index.items():
                if num in name:
                    return node_id

        # 키워드 매칭
        type_keywords = {
            '화장실': ['화장실 좌', '화장실 우'],
            '엘리베이터': ['엘리베이터 좌', '엘리베이터 우'],
            '계단': ['계단 좌', '계단 우'],
            '산학': ['산학협력단'],
        }
        for kw, labels in type_keywords.items():
            if kw in query_lower:
                for label in labels:
                    if label.lower() in self.feature_name_index:
                        return self.feature_name_index[label.lower()]

        return None

    # ── 위치 설명 ─────────────────────────────────────────────

    def get_location_description(self, candi1: str, candi2: str) -> str:
        if not candi1 or candi1 == 'Unknown':
            return "현재 위치를 파악 중입니다"
        name1 = self.get_display_name(candi1)
        if candi1 == candi2 or not candi2 or candi2 == 'Unknown':
            return f"{name1} 근처"
        name2 = self.get_display_name(candi2)
        if name1 == name2:
            return f"{name1} 근처"
        return f"{name1}과 {name2} 사이"

    # ── 주변 시설 조회 ────────────────────────────────────────

    def get_nearby_features(self, node_id: str) -> List[Dict]:
        """현재 노드와 인접 노드의 facility/room 조회"""
        nearby = []
        node = self.nodes.get(node_id)
        if node and node.get('type') in ('facility', 'room'):
            nearby.append({
                'name': node.get('label', self._id_to_label(node_id)),
                'type': node.get('type'),
                'feature_coords': node['coords'],
                'direction': None,
            })

        for nb_id in self.adjacency.get(node_id, []):
            nb_node = self.nodes.get(nb_id)
            if nb_node and nb_node.get('type') in ('facility', 'room'):
                nearby.append({
                    'name': nb_node.get('label', self._id_to_label(nb_id)),
                    'type': nb_node.get('type'),
                    'feature_coords': nb_node['coords'],
                    'direction': None,
                })
        return nearby

    def get_location_guide_context(
        self,
        current_node_id: str,
        robot_x: float, robot_y: float, yaw: float,
        prev_node_id: str = None
    ) -> Dict:
        node = self.nodes.get(current_node_id)
        if not node:
            return {}

        node_label = node.get('label', self._id_to_label(current_node_id))
        nearby = self.get_nearby_features(current_node_id)

        # 이동 방향 계산: prev → current 벡터로 yaw 대신 사용
        effective_yaw = yaw
        if prev_node_id and prev_node_id != current_node_id:
            pc = self.get_node_coords(prev_node_id)
            cc = self.get_node_coords(current_node_id)
            if pc and cc:
                dx = cc[0] - pc[0]
                dy = cc[1] - pc[1]
                if math.hypot(dx, dy) > 0.001:
                    effective_yaw = math.atan2(dy, dx)

        for feat in nearby:
            fc = feat.get('feature_coords')
            if fc:
                feat['direction'] = self.calc_feature_direction(
                    robot_x, robot_y, effective_yaw, fc
                )

        return {
            'current_label': node_label,
            'nearby_features': nearby,
        }

    # ── 벡터 외적 방향 판별 ───────────────────────────────────

    def calc_feature_direction(
        self,
        robot_x: float, robot_y: float, yaw: float,
        feat_coords
    ) -> str:
        if feat_coords and isinstance(feat_coords[0], list):
            fx = sum(c[0] for c in feat_coords) / len(feat_coords)
            fy = sum(c[1] for c in feat_coords) / len(feat_coords)
        else:
            fx, fy = float(feat_coords[0]), float(feat_coords[1])

        to_feat_x = fx - robot_x
        to_feat_y = fy - robot_y
        dist = math.hypot(to_feat_x, to_feat_y)
        if dist < 0.001:
            return "앞"

        dir_x = math.cos(yaw)
        dir_y = math.sin(yaw)
        cross = dir_x * to_feat_y - dir_y * to_feat_x
        dot = dir_x * to_feat_x + dir_y * to_feat_y

        threshold = 0.2 * dist
        if abs(cross) < threshold:
            return "앞" if dot > 0 else "뒤"
        elif cross > 0:
            return "왼쪽"
        else:
            return "오른쪽"

    # ── yaw 추론 ──────────────────────────────────────────────

    def estimate_position_from_candis(
        self,
        candi1: str,
        candi2: str,
        prev_candi: str = None,
        full_path: list = None
    ) -> Tuple[float, float, float]:
        c1 = self.get_node_coords(candi1)
        if not c1:
            return 0.0, 0.0, math.pi / 2

        c2 = self.get_node_coords(candi2) if candi2 and candi2 != candi1 else None
        if c2:
            x = (c1[0] + c2[0]) / 2
            y = (c1[1] + c2[1]) / 2
        else:
            x, y = c1[0], c1[1]

        if prev_candi and prev_candi != candi1:
            pc = self.get_node_coords(prev_candi)
            if pc:
                dx = c1[0] - pc[0]
                dy = c1[1] - pc[1]
                if math.hypot(dx, dy) > 0.001:
                    return x, y, math.atan2(dy, dx)

        if c2:
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            if math.hypot(dx, dy) > 0.001:
                return x, y, math.atan2(dy, dx)

        yaw = self._estimate_yaw_from_path(candi1, full_path)
        return x, y, yaw

    def _estimate_yaw_from_path(self, node_id: str, full_path: list = None) -> float:
        if full_path and node_id in full_path:
            idx = full_path.index(node_id)
            if idx > 0:
                prev_c = self.get_node_coords(full_path[idx - 1])
                curr_c = self.get_node_coords(node_id)
                if prev_c and curr_c:
                    dx = curr_c[0] - prev_c[0]
                    dy = curr_c[1] - prev_c[1]
                    if math.hypot(dx, dy) > 0.001:
                        return math.atan2(dy, dx)
            if idx < len(full_path) - 1:
                curr_c = self.get_node_coords(node_id)
                next_c = self.get_node_coords(full_path[idx + 1])
                if curr_c and next_c:
                    dx = next_c[0] - curr_c[0]
                    dy = next_c[1] - curr_c[1]
                    if math.hypot(dx, dy) > 0.001:
                        return math.atan2(dy, dx)

        neighbors = self.adjacency.get(node_id, [])
        if neighbors:
            curr_c = self.get_node_coords(node_id)
            nb_c = self.get_node_coords(neighbors[0])
            if curr_c and nb_c:
                dx = nb_c[0] - curr_c[0]
                dy = nb_c[1] - curr_c[1]
                if math.hypot(dx, dy) > 0.001:
                    return math.atan2(dy, dx)
        return math.pi / 2

    def get_robot_yaw(self, x: float, y: float, w: float) -> float:
        if w != 0.0:
            return float(w)
        return self._estimate_yaw_from_history(x, y)

    def _estimate_yaw_from_history(self, x: float, y: float) -> float:
        self._position_history.append((x, y))
        if len(self._position_history) > 10:
            self._position_history.pop(0)
        if len(self._position_history) < 2:
            return 0.0
        prev_x, prev_y = self._position_history[-2]
        dx = x - prev_x
        dy = y - prev_y
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            if len(self._position_history) >= 3:
                px, py = self._position_history[-3]
                dx = prev_x - px
                dy = prev_y - py
            else:
                return 0.0
        return math.atan2(dy, dx)

    # ── 경로 탐색 (BFS) ───────────────────────────────────────

    def find_path(self, start_node: str, goal_node: str,
                  exclude_types: List[str] = None) -> Optional[List[str]]:
        if start_node == goal_node:
            return [start_node]
        if start_node not in self.nodes or goal_node not in self.nodes:
            return None

        exclude_types = exclude_types or []
        queue = [(0, start_node, [start_node])]
        visited = set()
        distances = {start_node: 0}

        while queue:
            dist, current, path = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == goal_node:
                return path
            for neighbor in self.adjacency.get(current, []):
                if neighbor in visited:
                    continue
                if self.nodes.get(neighbor, {}).get('type', '') in exclude_types:
                    continue
                new_dist = dist + 1
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor, path + [neighbor]))
        return None

    def find_path_from_feature(self, goal_feature_name: str, start_node: str,
                                exclude_types: List[str] = None) -> Optional[List[str]]:
        goal_node = self.find_node_by_name(goal_feature_name)
        if not goal_node:
            return None
        return self.find_path(start_node, goal_node, exclude_types)

    def get_path_directions(self, path: List[str]) -> List[Dict]:
        if not path or len(path) < 2:
            return []

        directions = []
        for i, node_id in enumerate(path):
            label = self.get_display_name(node_id)
            nearby = self.get_nearby_features(node_id)
            nearby_names = [f["name"] for f in nearby]

            if i == 0:
                turn = "출발"
            elif i == len(path) - 1:
                turn = "도착"
            else:
                # 출발점이 endpoint/open_space이면 첫 꺾음은 "직진"으로 처리
                # (출발점의 진입 방향이 없어서 방향 계산 불가)
                prev_type = self.nodes.get(path[i-1], {}).get('type', '')
                if prev_type in ('endpoint', 'open_space'):
                    turn = "직진"
                else:
                    turn = self._calc_turn_at_node(path[i-1], node_id, path[i+1])

            directions.append({
                "node": node_id,
                "label": label,
                "direction": turn,
                "nearby": nearby_names,
            })
        return directions

    def _calc_turn_at_node(self, prev_node: str, curr_node: str, next_node: str) -> str:
        p = self.get_node_coords(prev_node)
        c = self.get_node_coords(curr_node)
        n = self.get_node_coords(next_node)

        if not p or not c or not n:
            return "직진"

        v1x, v1y = c[0] - p[0], c[1] - p[1]
        v2x, v2y = n[0] - c[0], n[1] - c[1]
        d1 = math.hypot(v1x, v1y)
        d2 = math.hypot(v2x, v2y)

        if d1 < 0.001 or d2 < 0.001:
            return "직진"

        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y
        threshold = 0.50 * d1 * d2  # sin(30°)=0.5, 30도 이하 꺾임은 직진으로 처리

        if abs(cross) < threshold:
            return "직진" if dot > 0 else "유턴"
        elif cross > 0:
            return "좌회전"
        else:
            return "우회전"

    def path_to_waypoints(self, path: List[str]) -> List[str]:
        if not path or len(path) <= 2:
            return path
        waypoints = [path[0]]
        for i in range(1, len(path) - 1):
            curr = path[i]
            if len(self.adjacency.get(curr, [])) >= 3:
                waypoints.append(curr)
                continue
            p = self.get_node_coords(path[i-1])
            c = self.get_node_coords(curr)
            n = self.get_node_coords(path[i+1])
            if p and c and n:
                v1 = (c[0]-p[0], c[1]-p[1])
                v2 = (n[0]-c[0], n[1]-c[1])
                d1 = math.hypot(*v1)
                d2 = math.hypot(*v2)
                if d1 > 0 and d2 > 0:
                    cos_a = (v1[0]*v2[0]+v1[1]*v2[1]) / (d1*d2)
                    cos_a = max(-1.0, min(1.0, cos_a))
                    if cos_a < 0.85:
                        waypoints.append(curr)
        waypoints.append(path[-1])
        return waypoints
