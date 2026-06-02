"""
맵 매니저 - nuri.json 기반 (새 구조 지원)

[변경사항]
- edges: from/to → start/end (둘 다 지원)
- features 배열 파싱 지원:
    hall_318(복도 노드) ↔ 318호(강의실 feature) 분리 구조
    connected_node 를 통해 복도 노드와 강의실을 연결
- _load(): nodes + edges + features 세 배열 모두 로드
- get_all_destinations(): self.features 기반으로 목적지 목록 반환
- get_nearby_features(): self.features 기반으로 주변 강의실/시설 조회
- find_node_by_name(): 강의실 번호 정규식 매칭 추가 (예: "318호", "318호 강의실")
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
        self.features: List[Dict] = []          # JSON features 배열 + node 기반 facility/room
        self.adjacency: Dict[str, List[str]] = {}
        self.feature_name_index: Dict[str, str] = {}   # 이름(소문자) → connected_node id
        self.feature_type_index: Dict[str, List[str]] = {}
        # node_id → 해당 노드에 연결된 feature 목록 (빠른 조회용)
        self._node_features: Dict[str, List[Dict]] = {}
        self._position_history: List[Tuple[float, float]] = []
        self._load()

    # ── 좌표 변환 헬퍼 ────────────────────────────────────────
    @staticmethod
    def _raw_to_coords(raw: list) -> list:
        """[row, col] → [x=col, y=-row]"""
        return [raw[1], -raw[0]]

    # ── node id → 한국어 라벨 ─────────────────────────────────
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
            # 복도 노드 라벨 — "~호 앞" 으로 표현 (실제 강의실과 구분)
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
        return name.replace('_', ' ')

    # ── 로드 ─────────────────────────────────────────────────
    def _load(self):
        with open(self.map_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # ── 1) nodes ──────────────────────────────────────────
        for node in data.get('nodes', []):
            node_id = node['id']
            node_type = node.get('type', 'corridor')

            if 'range' in node:
                r = node['range']
                row = (r[0][0] + r[1][0]) / 2
                col = (r[0][1] + r[1][1]) / 2
                coords = [col, -row]
            else:
                coords = self._raw_to_coords(node.get('coords', [0, 0]))

            label = node.get('label', self._id_to_label(node_id))
            self.nodes[node_id] = {
                'type': node_type,
                'coords': coords,
                'label': label,
            }

            # facility/room 타입 노드는 feature로도 등록 (산학협력단, 화장실 등)
            if node_type in ('facility', 'room'):
                self._register_feature(
                    name=label,
                    feat_type=node_type,
                    feat_coords=coords,
                    connected_node=node_id,
                )

        # ── 2) edges ──────────────────────────────────────────
        for edge in data.get('edges', []):
            src = edge.get('start') or edge.get('from', '')
            dst = edge.get('end') or edge.get('to', '')
            if src and dst:
                self.edges.append({'from': src, 'to': dst})
                self.adjacency.setdefault(src, []).append(dst)
                self.adjacency.setdefault(dst, []).append(src)

        # ── 3) features (강의실 등 별도 정의) ─────────────────
        for feat in data.get('features', []):
            name        = feat.get('name', '').strip()
            node_id     = feat.get('connected_node', '')
            feat_type   = feat.get('type', feat.get('types', 'room'))
            raw_coords  = feat.get('coords')

            if not name or not node_id:
                continue
            if node_id not in self.nodes:
                print(f"[MapManager] 경고: feature '{name}'의 connected_node "
                      f"'{node_id}'가 nodes에 없습니다.")
                continue

            feat_coords = (self._raw_to_coords(raw_coords)
                           if raw_coords else self.nodes[node_id]['coords'])

            self._register_feature(
                name=name,
                feat_type=feat_type,
                feat_coords=feat_coords,
                connected_node=node_id,
            )

        print(f"[MapManager] 로드 완료: 노드 {len(self.nodes)}개 / "
              f"엣지 {len(self.edges)}개 / 시설·강의실 {len(self.features)}개")

    def _register_feature(self, name: str, feat_type: str,
                           feat_coords: list, connected_node: str):
        """feature 하나를 self.features / feature_name_index / _node_features 에 등록"""
        # 중복 방지 (같은 connected_node + 같은 name)
        if any(f['connected_node'] == connected_node and f['name'] == name
               for f in self.features):
            return

        feat = {
            'name': name,
            'type': feat_type,
            'feature_coords': feat_coords,
            'connected_node': connected_node,
        }
        self.features.append(feat)
        self._node_features.setdefault(connected_node, []).append(feat)
        self.feature_type_index.setdefault(feat_type, []).append(connected_node)

        # ── 인덱싱 ──────────────────────────────────────────
        # 이름 전체
        self.feature_name_index[name.lower()] = connected_node
        # node_id 자체도 검색 가능하게
        self.feature_name_index.setdefault(connected_node.lower(), connected_node)

        # 단어 단위 키워드
        for kw in name.split():
            if len(kw) > 1:
                self.feature_name_index.setdefault(kw.lower(), connected_node)

        # 강의실 번호 별칭: "318호" → "318", "318번", "318호 강의실", "318강의실"
        m = re.match(r'^(\d{3})호?$', name.strip())
        if m:
            num = m.group(1)
            for alias in [num, f'{num}번', f'{num}호 강의실', f'{num}강의실']:
                self.feature_name_index.setdefault(alias, connected_node)

    def reload(self):
        self.nodes.clear()
        self.edges.clear()
        self.features.clear()
        self.adjacency.clear()
        self.feature_name_index.clear()
        self.feature_type_index.clear()
        self._node_features.clear()
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
        """목적지로 사용 가능한 이름 목록 (features 배열 기반)"""
        excluded = ['접근금지', '금지', 'end']
        seen = set()
        names = []
        for feat in self.features:
            name = feat.get('name', '')
            if not name or any(k in name for k in excluded):
                continue
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    # ── 시설·강의실 검색 ──────────────────────────────────────

    def find_nearest_facility(self, facility: str, current_node: str) -> Optional[str]:
        """현재 위치에서 키워드와 일치하는 가장 가까운 feature의 connected_node 반환"""
        curr_c = self.get_node_coords(current_node) if current_node else None
        candidates = []
        for feat in self.features:
            if facility in feat['name'].lower() or facility in feat['name']:
                c = self.get_node_coords(feat['connected_node'])
                if c:
                    dist = (math.hypot(c[0] - curr_c[0], c[1] - curr_c[1])
                            if curr_c else 0)
                    candidates.append((dist, feat['connected_node']))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None

    def find_node_by_name(self, query: str, current_node: str = None) -> Optional[str]:
        """이름/키워드로 목적지 node_id(connected_node) 검색"""
        query_lower = query.lower().strip()

        # ── 1) 강의실 번호 정규식 매칭 (최우선) ──────────────
        # 지원 패턴: "318호", "318호 강의실", "318번", "318" (3자리)
        m = re.search(r'(\d{3})\s*(?:호|번|강의실)?', query_lower)
        if m:
            num = m.group(1)
            # feature_name_index 에서 숫자로 검색
            for alias in [f'{num}호', num, f'{num}번', f'{num}호 강의실']:
                if alias in self.feature_name_index:
                    return self.feature_name_index[alias]
            # 직접 node id 확인
            candidate_id = f'hall_{num}'
            if candidate_id in self.nodes:
                return candidate_id

        # ── 2) 시설 키워드 (화장실 / 엘리베이터 / 계단) ──────
        right_kw = ['오른쪽', '우측', '오른', 'right']
        left_kw  = ['왼쪽', '좌측', '왼', 'left']

        facility_aliases = {
            '화장실':   ['화장실'],
            '엘리베이터': ['엘리베이터', '엘베', '승강기'],
            '계단':     ['계단'],
        }
        for facility, aliases in facility_aliases.items():
            if any(a in query_lower for a in aliases):
                if any(k in query_lower for k in right_kw):
                    for cand in [f'오른쪽 {facility}', f'{facility} 우']:
                        if cand.lower() in self.feature_name_index:
                            return self.feature_name_index[cand.lower()]
                elif any(k in query_lower for k in left_kw):
                    for cand in [f'왼쪽 {facility}', f'{facility} 좌']:
                        if cand.lower() in self.feature_name_index:
                            return self.feature_name_index[cand.lower()]
                else:
                    return self.find_nearest_facility(facility, current_node)

        # ── 3) 산학협력단 ──────────────────────────────────
        if '산학' in query_lower:
            node_id = self.feature_name_index.get('산학협력단')
            if node_id:
                return node_id

        # ── 4) 정확 매칭 ───────────────────────────────────
        if query_lower in self.feature_name_index:
            return self.feature_name_index[query_lower]

        # ── 5) 포함 매칭 (3글자 이상) ──────────────────────
        for name, node_id in self.feature_name_index.items():
            if name in query_lower and len(name) >= 3:
                return node_id

        # ── 6) 숫자 fallback ───────────────────────────────
        nums = re.findall(r'\d+', query)
        for num in nums:
            for name, node_id in self.feature_name_index.items():
                if num in name:
                    return node_id

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
        """현재 노드 및 인접 노드에 연결된 features 반환"""
        nearby = []
        seen_names = set()

        def _add_feats(nid: str):
            # features 배열에서 connected_node 기준으로 조회
            for feat in self._node_features.get(nid, []):
                if feat['name'] not in seen_names:
                    seen_names.add(feat['name'])
                    nearby.append({
                        'name': feat['name'],
                        'type': feat['type'],
                        'feature_coords': feat['feature_coords'],
                        'direction': None,
                    })

        _add_feats(node_id)
        for nb_id in self.adjacency.get(node_id, []):
            _add_feats(nb_id)

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
        dot   = dir_x * to_feat_x + dir_y * to_feat_y

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
            nb_c   = self.get_node_coords(neighbors[0])
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
            label  = self.get_display_name(node_id)
            nearby = self.get_nearby_features(node_id)
            nearby_names = [f["name"] for f in nearby]

            if i == 0:
                turn = "출발"
            elif i == len(path) - 1:
                turn = "도착"
            else:
                prev_type = self.nodes.get(path[i-1], {}).get('type', '')
                if prev_type in ('endpoint',):
                    turn = "직진"
                else:
                    # 이전 노드와 현재 노드가 같은 좌표면 더 이전 노드로 거슬러서 계산
                    import math as _math
                    prev_id = path[i-1]
                    pc = self.get_node_coords(prev_id)
                    cc = self.get_node_coords(node_id)
                    if pc and cc and _math.hypot(cc[0]-pc[0], cc[1]-pc[1]) < 0.001:
                        j = i - 1
                        while j >= 0:
                            prev_id = path[j]
                            pc = self.get_node_coords(prev_id)
                            if pc and _math.hypot(cc[0]-pc[0], cc[1]-pc[1]) > 0.001:
                                break
                            j -= 1
                    turn = self._calc_turn_at_node(prev_id, node_id, path[i+1])

            directions.append({
                "node":      node_id,
                "label":     label,
                "direction": turn,
                "nearby":    nearby_names,
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
        dot   = v1x * v2x + v1y * v2y
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

        # 중복 좌표 쌍 제거 (목적지 제외)
        # hall_318=hall_right_top, hall_321=hall_right_bottom, hall_산학협력단=hall_left_bottom
        goal = path[-1]
        skip_nodes = {'hall_right_top', 'hall_318', 'hall_right_bottom', 'hall_321', 'hall_산학협력단', 'hall_left_bottom'}
        waypoints = [w for w in waypoints if w not in skip_nodes or w == goal]

        return waypoints
