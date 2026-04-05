"""
LLM 처리기 - 3단계 흐름에 맞게 프롬프트 분리

"""

import os
import time
import json
import csv
from typing import Optional, Dict, List
from map_manager import MapManager
from datetime import datetime


# ── 1단계: 목적지 대화 프롬프트 ──────────────────────────────
DESTINATION_PROMPT = """당신은 시각장애인을 위한 실내 길안내 AI 어시스턴트입니다.
사용자의 말을 듣고 목적지를 파악해서 JSON으로 답하세요.

현재 사용자 위치: {current_location}

이 건물에서 갈 수 있는 목적지 (이름: node_id 형식):
{destination_list}

규칙:
- goalpoint에는 반드시 node_id를 넣으세요 (이름 X)
- 목적지를 파악했으면 map_search를 1로 설정하세요
- 뉘앙스로 목적지를 유추하세요 (예: "볼일이 급해" → 화장실의 node_id)
- 목적지 목록에 없는 곳이면 unavailable을 true로 설정
- "회의실"처럼 여러 개인 경우 가장 가까운 것 선택

의도 분류:
- set_destination: 목적지 설정
- destination_list: 갈 수 있는 곳 질문
- where_am_i: 현재 위치 질문
- unknown: 알 수 없음

응답은 반드시 JSON만:
{{"intent": "의도", "goalpoint": "node_id 또는 null", "waypoints": [], "map_search": 0, "unavailable": false, "reason": "", "response": "사용자에게 할 말"}}"""


# ── 3단계-SLAM트리거: 위치 안내 생성 프롬프트 ────────────────
LOCATION_GUIDE_PROMPT = """당신은 시각장애인을 위한 실내 길안내 AI 어시스턴트입니다.
현재 위치와 다음 이동 방향을 바탕으로 자연스러운 음성 안내 문장을 만들어주세요.

현재 위치: {current_label}
다음 이동 방향: {next_direction}
주변 시설: {nearby_features}

규칙:
1. 10초 안에 말할 수 있는 길이로 (60자 이내)
2. 현재 위치 + 다음 방향을 포함해서 안내
3. 주변 시설이 있으면 방향과 함께 언급
4. 자연스러운 구어체로
5. JSON 없이 안내 문장만 출력
6. 예시: "지금 205호 앞입니다. 오른쪽으로 직진하세요. 왼쪽에 엘리베이터가 있어요"
7. 도착 지점이면: "화장실에 도착했습니다. 왼쪽에 있습니다"

안내 문장:"""

# ── 전체 경로 안내 프롬프트 (출발 시 한 번) ──────────────────
ROUTE_SUMMARY_PROMPT = """당신은 시각장애인을 위한 실내 길안내 AI 어시스턴트입니다.
경로 정보를 바탕으로 전체 경로를 간단히 안내해주세요.

출발지: {start_label}
목적지: {goal_label}
경유 지점 및 방향:
{waypoint_directions}

규칙:
1. 20초 안에 말할 수 있는 길이로
2. 주요 방향 전환 지점만 언급
3. 자연스러운 구어체로
4. JSON 없이 안내 문장만 출력
5. 예시: "205호 앞에서 오른쪽으로 꺾어 엘리베이터 앞을 지나, 계단을 통해 1층 중앙홀로 나오시면 됩니다"

안내 문장:"""


# ── 3단계-사용자 말걸기: 의도 판단 프롬프트 ─────────────────
INTENT_PROMPT = """당신은 시각장애인을 위한 실내 길안내 AI 어시스턴트입니다.
이동 중에 사용자가 말을 걸었습니다. 의도를 파악해서 JSON으로 답하세요.

현재 상태: {state}
사용자 발화: "{text}"

의도 분류:
- stop: 잠시 멈춤 요청 (예: "잠깐", "멈춰", "정지")
- finish: 안내 종료 요청 (예: "그만해", "종료", "됐어")
- resume: 다시 출발 (예: "다시 가자", "출발해")
- change_destination: 목적지 변경
- where_am_i: 현재 위치 질문
- unknown: 그 외

응답은 반드시 JSON만:
{{"intent": "의도", "user_interrupt": false, "user_finish": false, "response": "사용자에게 할 말", "new_destination": null}}"""


# ── 목적지 목록 정리 프롬프트 ────────────────────────────────
DESTINATION_ORGANIZE_PROMPT = """당신은 시각장애인 길안내 시스템입니다.
아래 목적지 목록을 카테고리별로 간결하게 정리해서 음성으로 안내할 문장을 만들어주세요.

규칙:
1. 10초 안에 말할 수 있는 길이로 (50자 이내)
2. 카테고리로 묶어서 말하기
3. 자연스러운 구어체로
4. JSON 없이 문장만 출력
5. 목록에 없는 목적지는 절대 언급하지 말 것

전체 목적지: {destinations}

안내 문장:"""

DESTINATION_UNAVAILABLE_PROMPT = """당신은 시각장애인 길안내 시스템입니다.
사용자가 요청한 목적지가 이 건물에 없습니다.
아래 형식으로 자연스럽게 안내 문장을 만들어주세요.

형식: "죄송합니다. [없는 목적지]은(는) 없습니다. 갈 수 있는 곳은 [카테고리별 정리]입니다."

규칙:
1. 10초 안에 말할 수 있는 길이로
2. 카테고리로 묶어서 말하기
3. 자연스러운 구어체로
4. JSON 없이 문장만 출력
5. 전체 목적지 목록만 사용할 것

없는 목적지: {unavailable_dest}
전체 목적지: {destinations}

안내 문장:"""


class PerformanceLogger:
    def __init__(self, log_file: str = "/tmp/llm_performance.csv"):
        self.log_file = log_file
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'input_text', 'state',
                    'elapsed_ms', 'prompt_tokens', 'completion_tokens',
                    'intent', 'used_quick_match', 'model'
                ])

    def log(self, input_text, state, elapsed_ms, prompt_tokens,
            completion_tokens, intent, used_quick_match, model):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), input_text, state,
                f"{elapsed_ms:.1f}", prompt_tokens, completion_tokens,
                intent, used_quick_match, model
            ])

    def print_summary(self):
        try:
            if not os.path.exists(self.log_file):
                return
            rows = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            if not rows:
                return
            total = len(rows)
            quick = sum(1 for r in rows if r['used_quick_match'] == 'True')
            api_rows = [r for r in rows if r['used_quick_match'] == 'False']
            print("\n[성능 요약]")
            print(f"  총 호출 수: {total}")
            print(f"  Quick match 비율: {quick/total*100:.1f}%")
            if api_rows:
                elapsed = [float(r['elapsed_ms']) for r in api_rows]
                in_tok = [int(r['prompt_tokens']) for r in api_rows]
                out_tok = [int(r['completion_tokens']) for r in api_rows]
                print(f"  API 평균 응답시간: {sum(elapsed)/len(elapsed):.1f}ms")
                print(f"  API 평균 토큰(입력): {sum(in_tok)/len(in_tok):.0f}")
                print(f"  API 평균 토큰(출력): {sum(out_tok)/len(out_tok):.0f}")
                cost = (sum(in_tok)/1000*0.00015 + sum(out_tok)/1000*0.0006)
                print(f"  누적 API 비용(추정): ${cost:.4f}")
        except Exception as e:
            print(f"[성능 요약 오류] {e}")


class TrainingDataLogger:
    def __init__(self, log_file: str = "/tmp/training_data.jsonl"):
        self.log_file = log_file

    def log(self, input_text, state, result, system_prompt):
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"현재 상태: {state}\n사용자 발화: \"{input_text}\""},
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}
            ]
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


class LLMProcessor:
    def __init__(self, map_manager: MapManager, config: Dict):
        self.map = map_manager
        self.config = config
        self.provider = config.get('llm', {}).get('provider', 'openai')
        self.perf_logger = PerformanceLogger()
        self.train_logger = TrainingDataLogger()
        self._total_calls = 0
        self._quick_match_hits = 0
        self._api_calls = 0
        self._setup_client()

    def _setup_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.model = self.config['llm'].get('model_openai', 'gpt-4o-mini')
            print(f"[LLM] OpenAI {self.model} 준비 완료")
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model = self.config['llm'].get('model_gemini', 'gemini-1.5-pro')
            self.client = genai.GenerativeModel(self.model)
            print(f"[LLM] Gemini {self.model} 준비 완료")

    def _call_api(self, system_prompt: str, user_prompt: str,
                  max_tokens: int = 200, temperature: float = 0.1):
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (response.choices[0].message.content,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens)
        elif self.provider == "gemini":
            response = self.client.generate_content(
                f"{system_prompt}\n\n{user_prompt}"
            )
            try:
                pt = response.usage_metadata.prompt_token_count
                ct = response.usage_metadata.candidates_token_count
            except Exception:
                pt, ct = 0, 0
            return response.text, pt, ct
        return "", 0, 0

    def _get_destination_list_str(self) -> str:
        """목적지 목록을 "이름: node_id" 형식으로 반환"""
        lines = []
        excluded = ['접근금지', '금지구역']
        for feat in self.map.features:
            name = feat.get('name', '')
            node_id = feat.get('connected_node', '')
            if any(k in name for k in excluded):
                continue
            if name and node_id:
                lines.append(f"- {name}: {node_id}")
        return "\n".join(lines)

    # ── 1단계: 목적지 대화 ────────────────────────────────────

    def analyze_destination(self, text: str, candi1: str, candi2: str,
                             state: str = "waiting_destination") -> Dict:
        self._total_calls += 1
        start = time.time()

        quick = self._quick_match_destination(text, state)
        if quick:
            self._quick_match_hits += 1
            elapsed = (time.time() - start) * 1000
            self.perf_logger.log(text, state, elapsed, 0, 0,
                                 quick['intent'], True, 'quick_match')
            print(f"[LLM] Quick match | {elapsed:.1f}ms | '{text}' → {quick['intent']}")
            return quick

        self._api_calls += 1
        location_desc = self.map.get_location_description(candi1, candi2)
        dest_str = self._get_destination_list_str()

        system_prompt = DESTINATION_PROMPT.format(
            current_location=location_desc,
            destination_list=dest_str
        )
        user_prompt = f'현재 상태: {state}\n사용자 발화: "{text}"'

        try:
            raw, pt, ct = self._call_api(system_prompt, user_prompt)
            elapsed = (time.time() - start) * 1000
            raw = raw.strip().strip('```json').strip('```').strip()
            result = json.loads(raw)

            self.perf_logger.log(text, state, elapsed, pt, ct,
                                 result.get('intent', 'unknown'), False, self.model)
            self.train_logger.log(text, state, result, system_prompt)
            print(f"[LLM] 목적지분석 | {elapsed:.1f}ms | 토큰 {pt}+{ct} | "
                  f"'{text}' → {result.get('intent')} goalpoint={result.get('goalpoint')}")
            return result

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"[LLM 오류] {elapsed:.1f}ms | {e}")
            return {"intent": "unknown", "goalpoint": None, "waypoints": [],
                    "map_search": 0, "unavailable": False,
                    "reason": "", "response": "다시 말씀해주세요"}

    # ── 3단계-SLAM트리거: 위치 안내 생성 ─────────────────────

    def generate_location_guide(self, current_node_id: str,
                                 robot_x: float, robot_y: float, yaw: float,
                                 next_node_id: str = None,
                                 turn_str: str = None) -> str:
        """
        위치 안내 생성
        yaw     : 진입 방향 (이전→현재 벡터, 주변 시설 방향 계산용)
        turn_str: 다음 꺾을 방향 문자열 (직진/왼쪽/오른쪽, node_b에서 계산해서 전달)
        """
        start = time.time()
        ctx = self.map.get_location_guide_context(current_node_id, robot_x, robot_y, yaw)

        if not ctx:
            return f"{self.map.get_display_name(current_node_id)} 근처입니다"

        # 다음 방향: turn_str이 있으면 그대로 사용 (정확하게 계산된 값)
        next_direction = "정보 없음"
        if turn_str and next_node_id:
            next_label = self.map.get_display_name(next_node_id)
            next_direction = f"{turn_str}으로 이동 ({next_label} 방향)"
        elif next_node_id and next_node_id != current_node_id:
            # fallback: 기존 방식
            next_coords = self.map.get_node_coords(next_node_id)
            if next_coords:
                dir_str = self.map.calc_feature_direction(
                    robot_x, robot_y, yaw, list(next_coords)
                )
                next_label = self.map.get_display_name(next_node_id)
                next_direction = f"{dir_str}으로 이동 ({next_label} 방향)"

        # 주변 시설 (진입 방향 yaw 기준으로 계산됨)
        nearby_text = ""
        for feat in ctx.get('nearby_features', []):
            direction = feat.get('direction', '')
            name = feat.get('name', '')
            if direction and name:
                nearby_text += f"- {name}: {direction}\n"
        if not nearby_text:
            nearby_text = "없음"

        print(f"[위치안내 입력] 위치={ctx['current_label']} 다음방향={next_direction} 주변={nearby_text.strip()}")

        try:
            raw, pt, ct = self._call_api(
                LOCATION_GUIDE_PROMPT.format(
                    current_label=ctx['current_label'],
                    next_direction=next_direction,
                    nearby_features=nearby_text.strip()
                ),
                "",
                max_tokens=100,
                temperature=0.3
            )
            elapsed = (time.time() - start) * 1000
            print(f"[LLM] 위치안내 | {elapsed:.1f}ms | 토큰 {pt}+{ct}")
            return raw.strip()

        except Exception as e:
            print(f"[LLM 위치안내 오류] {e}")
            label = ctx.get('current_label', current_node_id)
            return f"{label} 근처입니다"

    def generate_route_summary(self, path) -> str:
        """출발 시 전체 경로 요약 안내 생성"""
        if not path or len(path) < 2:
            return ""

        start = time.time()
        directions = self.map.get_path_directions(path)
        # 경로 요약에서 출발/도착은 node label 우선 사용
        # (hall_203 → "CU 편의점" 대신 "203호 앞" 같은 위치 표현이 더 자연스러움)
        def _node_label(node_id):
            node = self.map.nodes.get(node_id)
            if node:
                return node.get('label', node_id)
            return self.map.get_display_name(node_id)

        start_label = _node_label(path[0])
        goal_label = self.map.get_display_name(path[-1])  # 도착은 시설 이름

        wp_lines = []
        for d in directions:
            if d['direction'] in ('좌회전', '우회전', '유턴', '출발', '도착'):
                nearby_str = f" (주변: {', '.join(d['nearby'][:2])})" if d['nearby'] else ""
                wp_lines.append(f"- {d['label']}: {d['direction']}{nearby_str}")
        if not wp_lines:
            wp_lines = [f"- {d['label']}: {d['direction']}" for d in directions]
        wp_text = "\n".join(wp_lines)

        try:
            raw, pt, ct = self._call_api(
                ROUTE_SUMMARY_PROMPT.format(
                    start_label=start_label,
                    goal_label=goal_label,
                    waypoint_directions=wp_text
                ),
                "",
                max_tokens=150,
                temperature=0.3
            )
            elapsed = (time.time() - start) * 1000
            print(f"[LLM] 경로요약 | {elapsed:.1f}ms | 토큰 {pt}+{ct}")
            return raw.strip()

        except Exception as e:
            print(f"[LLM 경로요약 오류] {e}")
            return f"{start_label}에서 {goal_label}로 안내합니다"

    # ── 3단계-사용자 말걸기: 의도 판단 ──────────────────────

    def analyze_intent(self, text: str, state: str = "navigating") -> Dict:
        self._total_calls += 1
        start = time.time()

        quick = self._quick_match_intent(text)
        if quick:
            self._quick_match_hits += 1
            elapsed = (time.time() - start) * 1000
            print(f"[LLM] Quick match(intent) | {elapsed:.1f}ms | '{text}' → {quick['intent']}")
            return quick

        self._api_calls += 1
        system_prompt = INTENT_PROMPT.format(state=state, text=text)

        try:
            raw, pt, ct = self._call_api(system_prompt, "", max_tokens=150)
            elapsed = (time.time() - start) * 1000
            raw = raw.strip().strip('```json').strip('```').strip()
            result = json.loads(raw)
            print(f"[LLM] 의도판단 | {elapsed:.1f}ms | 토큰 {pt}+{ct} | "
                  f"'{text}' → {result.get('intent')}")
            return result

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"[LLM 오류] {elapsed:.1f}ms | {e}")
            return {"intent": "unknown", "user_interrupt": False,
                    "user_finish": False, "response": "", "new_destination": None}

    # ── 목적지 목록 정리 ──────────────────────────────────────

    def get_organized_destinations(self, user_text: str,
                                    unavailable: bool = False,
                                    unavailable_dest: str = "") -> str:
        destinations = self.map.get_all_destinations()
        dest_str = ', '.join(destinations)

        if len(destinations) <= 4:
            if unavailable:
                return f"죄송합니다. {unavailable_dest}은(는) 없습니다. 갈 수 있는 곳은 {dest_str} 입니다"
            return f"갈 수 있는 곳은 {dest_str} 입니다"

        if unavailable:
            prompt = DESTINATION_UNAVAILABLE_PROMPT.format(
                unavailable_dest=unavailable_dest,
                destinations=dest_str
            )
        else:
            prompt = DESTINATION_ORGANIZE_PROMPT.format(destinations=dest_str)

        try:
            raw, _, _ = self._call_api(prompt, "", max_tokens=100, temperature=0.3)
            return raw.strip()
        except Exception as e:
            print(f"[목적지 정리 오류] {e}")
            return self._fallback_organize(destinations)

    def _fallback_organize(self, destinations: List[str]) -> str:
        facility_kw = ['화장실', '엘리베이터', '계단', '출입구']
        classroom_kw = ['호', '강의실']
        facilities = [d for d in destinations if any(k in d for k in facility_kw)]
        classrooms = [d for d in destinations if any(k in d for k in classroom_kw)]
        others = [d for d in destinations if d not in facilities and d not in classrooms]
        parts = []
        if facilities:
            parts.append(', '.join(facilities))
        if classrooms:
            parts.append(f"강의실 {len(classrooms)}개" if len(classrooms) > 3 else ', '.join(classrooms))
        if others:
            parts.append(', '.join(others))
        return "갈 수 있는 곳은 " + ", ".join(parts) + " 입니다"

    # ── Quick match ───────────────────────────────────────────

    def _quick_match_destination(self, text: str, state: str) -> Optional[Dict]:
        t = text.lower().strip()

        if any(w in t for w in ['정지', '멈춰', '스톱', '위험', '그만', '잠깐']):
            return {"intent": "stop", "goalpoint": None, "waypoints": [],
                    "map_search": 0, "unavailable": False, "reason": "", "response": "정지합니다"}

        if any(w in t for w in ['어디 갈', '목적지 알려', '어디어디', '갈 수 있는']):
            return {"intent": "destination_list", "goalpoint": None, "waypoints": [],
                    "map_search": 0, "unavailable": False, "reason": "", "response": ""}

        if any(w in t for w in ['여기가 어디', '지금 어디', '현재 위치', '어디야']):
            return {"intent": "where_am_i", "goalpoint": None, "waypoints": [],
                    "map_search": 0, "unavailable": False, "reason": "", "response": ""}

        if state == 'waiting_confirm':
            if any(w in t for w in ['네', '예', '응', '좋아', '출발', '시작', '그래', '맞아', '어', '가줘', '안내해', '출발해']):
                return {"intent": "confirm_yes", "goalpoint": None, "waypoints": [],
                        "map_search": 0, "unavailable": False, "reason": "", "response": ""}
            if any(w in t for w in ['아니', '싫어', '안돼', '됐어', '다시', '말고', '취소']):
                return {"intent": "confirm_no", "goalpoint": None, "waypoints": [],
                        "map_search": 0, "unavailable": False, "reason": "", "response": ""}

        if state == 'waiting_arrival':
            if any(w in t for w in ['아니', '괜찮아', '됐어', '없어', '필요없어', '끝', '종료']):
                return {"intent": "arrival_response_no", "goalpoint": None, "waypoints": [],
                        "map_search": 0, "unavailable": False, "reason": "", "response": ""}
            if any(w in t for w in ['있어', '응', '네', '더', '또', '다른']):
                return {"intent": "arrival_response_yes", "goalpoint": None, "waypoints": [],
                        "map_search": 0, "unavailable": False, "reason": "", "response": ""}

        # 키워드로 목적지 직접 찾기 → node_id 반환
        node_id = self.map.find_node_by_name(t)
        if node_id:
            return {"intent": "set_destination", "goalpoint": node_id, "waypoints": [],
                    "map_search": 1, "unavailable": False, "reason": "", "response": ""}

        return None

    def _quick_match_intent(self, text: str) -> Optional[Dict]:
        t = text.lower().strip()

        if any(w in t for w in ['정지', '멈춰', '스톱', '위험', '잠깐', '잠깐만']):
            return {"intent": "stop", "user_interrupt": True, "user_finish": False,
                    "response": "잠시 멈추겠습니다", "new_destination": None}

        if any(w in t for w in ['종료', '끝내', '꺼줘', '필요없어', '안내 그만', '그만해']):
            return {"intent": "finish", "user_interrupt": False, "user_finish": True,
                    "response": "안내를 종료합니다", "new_destination": None}

        if any(w in t for w in ['다시 가', '출발해', '계속 가', '다시 출발']):
            return {"intent": "resume", "user_interrupt": False, "user_finish": False,
                    "response": "다시 출발합니다", "new_destination": None}

        if any(w in t for w in ['여기가 어디', '지금 어디', '현재 위치']):
            return {"intent": "where_am_i", "user_interrupt": False, "user_finish": False,
                    "response": "", "new_destination": None}

        return None

    def print_stats(self):
        print(f"\n[LLM 통계]")
        print(f"  총 호출: {self._total_calls}")
        print(f"  Quick match: {self._quick_match_hits} "
              f"({self._quick_match_hits/max(self._total_calls,1)*100:.1f}%)")
        print(f"  API 호출: {self._api_calls} "
              f"({self._api_calls/max(self._total_calls,1)*100:.1f}%)")
        self.perf_logger.print_summary()
