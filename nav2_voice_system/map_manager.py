"""
맵 매니저 - nuri.json 기반
노드 조회, BFS 경로 탐색, 벡터 외적 방향 계산, 주변 시설 조회

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

    def _load(self):
        with open(self.map_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for node in data.get('nodes', []):
            node_id = node['id']
            coords = node.get('coords', [0, 0])
            if coords and isinstance(coords[0], list):
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                coords = [sum(xs)/len(xs), sum(ys)/len(ys)]
            self.nodes[node_id] = {
                'type': node.get('type', ''),
                'coords': coords,
                'label': node.get('label', node_id),
            }

        for edge in data.get('edges', []):
            self.edges.append(edge)
            src = edge.get('from', edge.get('start', ''))
            dst = edge.get('to', edge.get('end', ''))
            if src and dst:
                self.adjacency.setdefault(src, []).append(dst)
                self.adjacency.setdefault(dst, []).append(src)

        for feat in data.get('features', []):
            self.features.append(feat)
            name = feat.get('name', '')
            node_id = feat.get('connected_node', '')
            feat_type = feat.get('type', '')
            if name:
                self.feature_name_index[name] = node_id
                self.feature_name_index[name.lower()] = node_id
            if feat_type:
                self.feature_type_index.setdefault(feat_type, []).append(node_id)

        print(f"[MapManager] nuri.json 로드 완료: "
              f"노드 {len(self.nodes)}개 / "
              f"엣지 {len(self.edges)}개 / "
              f"시설 {len(self.features)}개")

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
        """node_id → 사람이 읽기 좋은 이름
        우선순위: 시설(화장실/엘리베이터 등) > 강의실/사무실 > node label
        """
        priority_types = ['화장실', '엘리베이터', '편의점', '출입구', '계단', '내리막길']

        for feat in self.features:
            if feat.get('connected_node') == node_id:
                if feat.get('type') in priority_types:
                    return feat['name']

        for feat in self.features:
            if feat.get('connected_node') == node_id:
                return feat['name']

        node = self.nodes.get(node_id)
        if node:
            return node.get('label', node_id)
        return node_id

    def get_all_destinations(self) -> List[str]:
        excluded_keywords = ['접근금지', '금지구역']
        names = []
        for feat in self.features:
            name = feat.get('name', '')
            if any(k in name for k in excluded_keywords):
                continue
            if name:
                names.append(name)
        return names

    def find_node_by_name(self, query: str) -> Optional[str]:
        query_lower = query.lower().strip()

        if query_lower in self.feature_name_index:
            return self.feature_name_index[query_lower]

        for name, node_id in self.feature_name_index.items():
            if query_lower in name or name in query_lower:
                return node_id

        nums = re.findall(r'\d{3}', query)
        for num in nums:
            for name, node_id in self.feature_name_index.items():
                if num in name:
                    return node_id

        type_keywords = {
            '화장실': '화장실',
            '엘리베이터': '엘리베이터',
            '편의점': 'cu 편의점',
            'cu': 'cu 편의점',
            '계단': '계단',
            '출입구': '외부 출입구',
            '출구': '외부 출입구',
        }
        for kw, feat_name in type_keywords.items():
            if kw in query_lower and feat_name in self.feature_name_index:
                return self.feature_name_index[feat_name]

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
        nearby = []
        for feat in self.features:
            if feat.get('connected_node') == node_id:
                nearby.append({
                    'name': feat['name'],
                    'type': feat['type'],
                    'feature_coords': feat.get('feature_coords'),
                    'direction': None,
                })
        return nearby

    def get_location_guide_context(
        self,
        current_node_id: str,
        robot_x: float, robot_y: float, yaw: float
    ) -> Dict:
        node = self.nodes.get(current_node_id)
        if not node:
            return {}

        node_label = node.get('label', current_node_id)
        nearby = self.get_nearby_features(current_node_id)

        for feat in nearby:
            fc = feat.get('feature_coords')
            if fc:
                feat['direction'] = self.calc_feature_direction(
                    robot_x, robot_y, yaw, fc
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

        # 외적 z성분: 양수=왼쪽, 음수=오른쪽 (y축 위쪽 기준 표준 좌표계)
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
        """
        현재 위치(x, y)와 진행 방향(yaw) 추정

        우선순위:
        1. prev_candi → candi1 벡터 (가장 신뢰도 높음)
        2. candi1 → candi2 벡터
        3. full_path에서 추정
        4. 기본값 (위쪽, π/2)
        """
        c1 = self.get_node_coords(candi1)
        if not c1:
            return 0.0, 0.0, math.pi / 2

        # 위치: candi1과 candi2 중간
        c2 = self.get_node_coords(candi2) if candi2 and candi2 != candi1 else None
        if c2:
            x = (c1[0] + c2[0]) / 2
            y = (c1[1] + c2[1]) / 2
        else:
            x, y = c1[0], c1[1]

        # yaw 계산 우선순위 1: prev_candi → candi1
        if prev_candi and prev_candi != candi1:
            pc = self.get_node_coords(prev_candi)
            if pc:
                dx = c1[0] - pc[0]
                dy = c1[1] - pc[1]
                if math.hypot(dx, dy) > 0.001:
                    return x, y, math.atan2(dy, dx)

        # yaw 계산 우선순위 2: candi1 → candi2
        if c2:
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            if math.hypot(dx, dy) > 0.001:
                return x, y, math.atan2(dy, dx)

        # yaw 계산 우선순위 3: full_path 기반
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
                neighbor_type = self.nodes.get(neighbor, {}).get('type', '')
                if neighbor_type in exclude_types:
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
                prev_node = path[i - 1]
                next_node = path[i + 1]
                turn = self._calc_turn_at_node(prev_node, node_id, next_node)

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

        v1x = c[0] - p[0]
        v1y = c[1] - p[1]
        v2x = n[0] - c[0]
        v2y = n[1] - c[1]

        d1 = math.hypot(v1x, v1y)
        d2 = math.hypot(v2x, v2y)

        if d1 < 0.001 or d2 < 0.001:
            return "직진"

        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y

        threshold = 0.15 * d1 * d2
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
