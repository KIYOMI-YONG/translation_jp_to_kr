from __future__ import annotations

import json
import re
import time
from pathlib import Path

import ollama


# =========================
# 기본 설정
# =========================

# 각 작업은 서로 다른 Ollama 모델을 사용할 수 있다.
# 후보 추출 모델은 설치된 소형 모델 태그로 변경할 수 있다.
GLOSSARY_MODEL_NAME = "gemma4:12b-it-qat"
TRANSLATION_MODEL_NAME = "gemma4:12b-it-qat"
TITLE_MODEL_NAME = "gemma4:e2b"
BASE_DIR = Path(__file__).resolve().parent
print(f"실행 중인 스크립트: {Path(__file__).resolve()}")

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GLOSSARY_PATH = BASE_DIR / "glossary.json"
GLOSSARY_CANDIDATES_PATH = BASE_DIR / "glossary_candidates.json"

TRANSLATION_DIRECTIONS = {
    "ja_ko": {
        "source_language": "Japanese",
        "target_language": "Korean",
        "source_label": "일본어",
        "target_label": "한국어",
        "output_suffix": "ko",
        "glossary_path": GLOSSARY_PATH,
        "candidates_path": GLOSSARY_CANDIDATES_PATH,
    },
    "ko_ja": {
        "source_language": "Korean",
        "target_language": "Japanese",
        "source_label": "한국어",
        "target_label": "일본어",
        "output_suffix": "ja",
        "glossary_path": BASE_DIR / "glossary_ko_ja.json",
        "candidates_path": BASE_DIR / "glossary_candidates_ko_ja.json",
    },
}

SUPPORTED_EXTENSIONS = {".txt"}

# 한 번에 전송할 대략적인 문자 수
# 모델의 실제 컨텍스트 길이와 컴퓨터 성능에 맞춰 조절
MAX_CHUNK_CHARS = 1500

# 번역 시 참고할 이전 원문/번역문의 최대 문자 수
CONTEXT_CHARS = 500

MAX_RETRIES = 3

# 일본어 고유명사 처리 방식
# "best_guess": 독음이 없으면 모델이 가장 가능성 높은 일본식 독음을 추정
# "keep_original": 독음이 불확실하면 한자를 그대로 유지
UNKNOWN_JAPANESE_NAME_POLICY = "best_guess"

FIRST_LINE_IS_TITLE = False

# =========================
# 파일 선택
# =========================
def select_operation() -> str:
    """용어집 후보 생성과 본문 번역을 별도 단계로 실행한다."""
    print()
    print("실행할 작업을 선택하세요.")
    print("1. 용어집 후보 생성")
    print("2. 확정 용어집으로 본문 번역")
    print("q. 종료")

    while True:
        choice = input("작업 번호: ").strip().lower()

        if choice == "1":
            return "extract_glossary"
        if choice == "2":
            return "translate"
        if choice in {"q", "quit", "exit"}:
            return "quit"

        print("1, 2 또는 q를 입력하세요.")


def select_translation_direction() -> str:
    """번역할 원문 언어와 목표 언어를 선택한다."""
    print()
    print("번역 방향을 선택하세요.")
    print("1. 일본어 → 한국어")
    print("2. 한국어 → 일본어")
    print("q. 이전 메뉴")

    while True:
        choice = input("번역 방향: ").strip().lower()

        if choice == "1":
            return "ja_ko"
        if choice == "2":
            return "ko_ja"
        if choice in {"q", "quit", "exit"}:
            return "back"

        print("1, 2 또는 q를 입력하세요.")


def select_input_file() -> Path | None:
    """
    input 폴더의 텍스트 파일을 목록으로 보여주고
    사용자가 번호를 선택하게 한다.
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print()
        print("input 폴더에 번역할 텍스트 파일이 없습니다.")
        print(f"파일을 넣을 위치: {INPUT_DIR}")
        return None

    print()
    print("번역할 파일을 선택하세요.")
    print()

    for index, file_path in enumerate(files, start=1):
        file_size = file_path.stat().st_size

        if file_size >= 1024 * 1024:
            size_text = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size >= 1024:
            size_text = f"{file_size / 1024:.1f} KB"
        else:
            size_text = f"{file_size} bytes"

        print(f"{index}. {file_path.name} ({size_text})")

    print()
    print("종료하려면 q를 입력하세요.")

    while True:
        choice = input("파일 번호: ").strip()

        if choice.lower() in {"q", "quit", "exit"}:
            return None

        if not choice.isdigit():
            print("파일 번호를 숫자로 입력하세요.")
            continue

        selected_index = int(choice) - 1

        if not 0 <= selected_index < len(files):
            print("목록에 있는 번호를 입력하세요.")
            continue

        return files[selected_index]
    

def read_source_text(path: Path, source_language: str) -> str:
    """
    UTF-8과 원문 언어에 맞는 Windows 인코딩을 차례대로 시도한다.
    """
    legacy_encoding = (
        "cp932"
        if source_language == "Japanese"
        else "cp949"
    )

    encodings = ["utf-8-sig", "utf-8", legacy_encoding]

    errors: list[str] = []

    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)

            print(f"파일 인코딩: {encoding}")
            return text

        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")

    error_details = "\n".join(errors)

    raise UnicodeError(
        f"파일 인코딩을 판별하지 못했습니다: {path.name}\n"
        f"{error_details}"
    )

def make_output_paths(
    input_path: Path,
    direction_key: str,
    translated_title: str | None = None,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_suffix = TRANSLATION_DIRECTIONS[direction_key][
        "output_suffix"
    ]

    if translated_title:
        output_stem = (
            f"{sanitize_filename(translated_title)}_{output_suffix}"
        )
    else:
        output_stem = f"{input_path.stem}_{output_suffix}"

    output_path = OUTPUT_DIR / f"{output_stem}.txt"

    # 진행 파일은 원본 파일명을 기준으로 유지한다.
    progress_path = (
        OUTPUT_DIR
        / (
            f".{input_path.stem}_{direction_key}"
            "_translation_progress.json"
        )
    )

    return output_path, progress_path

# =========================
# 언어 감지
# =========================

def detect_source_language(text: str) -> str:
    """
    간단한 문자 기반 언어 감지.
    한글이 있으면 Korean, 일본어 문자가 있으면 Japanese,
    그 외에는 English로 간주한다.
    """
    korean_pattern = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff]")
    japanese_pattern = re.compile(
        r"[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
    )

    if korean_pattern.search(text):
        return "Korean"

    if japanese_pattern.search(text):
        return "Japanese"

    return "English"


# =========================
# 용어집
# =========================

def get_glossary_target(entry: dict) -> str:
    """신규 target 필드와 기존 방향별 필드를 함께 지원한다."""
    return (
        entry.get("target", "")
        or entry.get("korean", "")
        or entry.get("japanese", "")
    )


def load_glossary(path: Path) -> dict:
    """
    두 가지 형식을 모두 지원한다.

    단순 형식:
    {
        "山田太郎": "야마다 타로"
    }

    확장 형식:
    {
     "小鳥遊六花": {
       "korean": "타카나시 릿카",
       "reading": "たかなし りっか",
       "type": "person"
     },
      "月城雪乃": {
       "korean": "츠키시로 유키노",
       "reading": "つきしろ ゆきの",
       "type": "person"
      },
     "桜川": {
       "korean": "사쿠라가와",
       "reading": "さくらがわ",
      "type": "place"
     },
     "月ヶ丘駅": {
       "korean": "츠키가오카역",
       "reading": "つきがおかえき",
       "type": "place"
     },
     "王都": {
       "korean": "왕도",
       "type": "term"
     },
     "魔王城": {
       "korean": "마왕성",
       "type": "term"
     }
    }
    """
    if not path.exists():
        return {}

    raw_text = path.read_text(encoding="utf-8-sig")

    if not raw_text.strip():
        return {}

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path.name}의 JSON 형식이 잘못되었습니다. "
            f"{error.lineno}행 {error.colno}열: {error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name}의 최상위 구조는 JSON 객체여야 합니다."
        )

    for source, entry in data.items():
        if isinstance(entry, str):
            continue

        if not isinstance(entry, dict):
            raise ValueError(
                f"용어집 항목 형식이 잘못되었습니다: {source}"
            )

        if not get_glossary_target(entry):
            raise ValueError(
                f"용어집 항목에 목표 언어 표기가 없습니다: {source}"
            )

    return data


def save_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def generate_glossary_candidates(
    source_text: str,
    source_language: str,
    target_language: str,
) -> dict:
    if source_language == "Japanese":
        proper_noun_rule = """
일본어 인명과 지명은 한국식 한자음이 아니라 실제 일본어 독음을 기준으로
한국어 표기를 작성하십시오.
""".strip()
    else:
        proper_noun_rule = """
한국 인명과 지명은 원래 한국어 발음을 기준으로 자연스러운 일본어 표기를
작성하십시오. 확정된 한자 표기가 없다면 인명은 가타카나 음역을 우선하십시오.
""".strip()

    prompt = f"""
다음 {source_language} 소설 원문에서 반복적으로 등장할 가능성이 있는
고유명사와 핵심 용어를 추출하십시오.

추출 대상:
- 인명
- 지명
- 기관명
- 학교명
- 조직명
- 작품 고유 설정 용어
- 반복 등장하는 마법, 기술, 계급, 종족명

제외 대상:
- 일반 명사
- 평범한 동사와 형용사
- 한 번만 등장하는 중요하지 않은 표현
- 문장 전체

{proper_noun_rule}

반드시 다음 JSON 형식만 출력하십시오.

{{
  "원문 표기": {{
    "target": "{target_language} 표기",
    "reading": "원어 독음",
    "type": "person | place | organization | term"
  }}
}}

설명이나 마크다운 코드 블록은 출력하지 마십시오.

원문:
{source_text}
""".strip()

    last_error: Exception | None = None
    last_raw_result = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=GLOSSARY_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                format="json",
                think=False,
                options={
                    "temperature": 0.0,
                    "num_ctx": 8192,
                    "num_predict": 2000,
                },
                keep_alive="30m",
            )

            last_raw_result = (
                response.message.content or ""
            ).strip()

            if not last_raw_result:
                raise RuntimeError(
                    "자동 용어집 모델이 빈 응답을 반환했습니다."
                )

            result = json.loads(last_raw_result)

            if not isinstance(result, dict):
                raise ValueError(
                    "자동 용어집 결과가 JSON 객체가 아닙니다."
                )

            return result

        except (
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
        ) as error:
            last_error = error
            print(
                f"용어집 후보 요청 실패 "
                f"({attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt < MAX_RETRIES:
                wait_seconds = 2 ** attempt
                print(f"{wait_seconds}초 후 다시 시도합니다.")
                time.sleep(wait_seconds)

    raise RuntimeError(
        "자동 용어집 결과를 JSON으로 읽지 못했습니다.\n"
        f"마지막 모델 출력:\n{last_raw_result}"
    ) from last_error

def merge_glossaries(
    existing: dict,
    candidates: dict,
) -> tuple[dict, dict]:
    """
    기존 항목은 유지한다.
    새로 추가된 항목도 따로 반환한다.
    """
    merged = dict(existing)
    added: dict = {}

    for source, entry in candidates.items():
        if source in merged:
            continue

        merged[source] = entry
        added[source] = entry

    return merged, added


def merge_candidate_results(
    accumulated: dict,
    new_candidates: dict,
) -> dict:
    """여러 청크의 후보를 병합하며 먼저 발견된 표기를 유지한다."""
    merged = dict(accumulated)

    for source, entry in new_candidates.items():
        if source not in merged:
            merged[source] = entry

    return merged


def apply_furigana_hints(
    candidates: dict,
    reading_hints: dict[str, str],
) -> dict:
    """모델의 추정보다 원문에 직접 적힌 후리가나를 우선한다."""
    updated = dict(candidates)

    for written_form, reading in reading_hints.items():
        entry = updated.get(written_form)

        if entry is None:
            updated[written_form] = {
                "target": "",
                "reading": reading,
                "type": "unknown",
            }
            continue

        if isinstance(entry, dict):
            updated[written_form] = {
                **entry,
                "reading": reading,
            }

    return updated


def save_glossary(path: Path, glossary: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            glossary,
            file,
            ensure_ascii=False,
            indent=2,
        )


def filter_glossary_for_text(
    glossary: dict,
    text: str,
) -> dict:
    """
    현재 원문과 이전 원문 문맥에 관련된 용어만 반환한다.
    별칭과 항상 포함 설정을 지원한다.
    """
    filtered: dict = {}

    for source, entry in glossary.items():
        if isinstance(entry, str):
            if source in text:
                filtered[source] = entry

            continue

        if entry.get("always_include", False):
            filtered[source] = entry
            continue

        search_terms = [source]
        aliases = entry.get("aliases", [])

        if isinstance(aliases, list):
            search_terms.extend(
                alias.strip()
                for alias in aliases
                if isinstance(alias, str) and alias.strip()
            )

        if any(term in text for term in search_terms if term):
            filtered[source] = entry

    return filtered


def format_glossary(glossary: dict) -> str:
    if not glossary:
        return "등록된 용어 없음"

    lines: list[str] = []

    for source, entry in glossary.items():
        # 기존 단순 문자열 형식
        if isinstance(entry, str):
            lines.append(f"- {source} → {entry}")
            continue

        target = get_glossary_target(entry)
        reading = entry.get("reading", "")
        entry_type = entry.get("type", "")

        details: list[str] = []

        if reading:
            details.append(f"원어 독음: {reading}")

        if entry_type:
            details.append(f"분류: {entry_type}")

        if details:
            lines.append(
                f"- {source} → {target} "
                f"({', '.join(details)})"
            )
        else:
            lines.append(f"- {source} → {target}")

    return "\n".join(lines)



def extract_furigana_hints(text: str) -> dict[str, str]:
    """
    원문에 포함된 후리가나 표기를 찾는다.

    지원 예시:
    小鳥遊《たかなし》
    月城（つきしろ）
    神代【かみしろ】
    """
    patterns = [
        r"([一-龯々〆ヵヶ]+)《([ぁ-んァ-ヶー\s]+)》",
        r"([一-龯々〆ヵヶ]+)（([ぁ-んァ-ヶー\s]+)）",
        r"([一-龯々〆ヵヶ]+)【([ぁ-んァ-ヶー\s]+)】",
    ]

    readings: dict[str, str] = {}

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for written_form, reading in matches:
            written_form = written_form.strip()
            reading = reading.strip()

            if written_form and reading:
                readings[written_form] = reading

    return readings


def format_reading_hints(readings: dict[str, str]) -> str:
    if not readings:
        return "현재 번역 범위에서 발견된 후리가나 없음"

    return "\n".join(
        f"- {written_form}의 독음: {reading}"
        for written_form, reading in readings.items()
    )

def make_japanese_proper_noun_rules(
    reading_hints: dict[str, str],
) -> str:
    hints_text = format_reading_hints(reading_hints)

    if UNKNOWN_JAPANESE_NAME_POLICY == "keep_original":
        unknown_name_rule = """
독음을 확신할 수 없는 인명이나 지명은 임의로 읽지 말고,
원문의 한자 표기를 그대로 유지한다.
"""
    else:
        unknown_name_rule = """
독음 정보가 없는 일본식 인명이나 지명은 문맥과 일반적인
일본어 이름 독법을 바탕으로 가장 가능성 높은 독음을 추정하여
한글로 음역한다. 한번 선택한 표기는 이후에도 일관되게 유지한다.
"""

    return f"""
일본어 고유명사 처리 규칙:

1. 일본식 인명과 지명은 한자의 한국식 한자음으로 번역하지 않는다.
2. 일본어에서 실제로 읽는 발음을 기준으로 한글로 음역한다.
3. 용어집에 등록된 표기를 가장 우선한다.
4. 후리가나, 가나 표기, 괄호 속 독음이 있으면 그 독음을 따른다.
5. 인명은 일본어 원문의 성명 순서를 유지한다.
6. 같은 인명과 지명은 번역 전체에서 같은 표기를 사용한다.
7. 일본 인명 뒤의 さん, ちゃん, くん, 先輩 등의 호칭은
   인물 관계와 문맥에 맞게 번역한다.
8. 일본식 고유명사만 음역하고 일반 명사는 정상적으로 번역한다.
9. 중국인, 한국인, 서양인의 이름을 무조건 일본식으로 읽지 않는다.
   해당 인물의 출신과 용어집 정보를 우선한다.
10. 독음이나 번역에 관한 설명, 괄호 주석, 후보 목록은 출력하지 않는다.

고유명사와 일반 명사 구분 예시:

- 山田太郎 → 야마다 타로
- 東京 → 도쿄
- 京都 → 교토
- 桜川高校 → 사쿠라가와 고등학교
- 月ヶ丘駅 → 츠키가오카역
- 黒森村 → 쿠로모리 마을
- 王都 → 왕도
- 魔王城 → 마왕성
- 騎士団 → 기사단

독음이 확인된 항목:

{hints_text}

독음 불명 항목 처리:

{unknown_name_rule.strip()}
""".strip()

# =========================
# 텍스트 분할
# =========================

def split_into_blocks(text: str) -> list[str]:
    """
    빈 줄도 별도 블록으로 보존한다.

    반환 예시:
    [
        "첫 번째 문단",
        "\\n\\n",
        "두 번째 문단"
    ]
    """
    return re.split(r"(\n\s*\n)", text)


def build_chunks(
    blocks: list[str],
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """
    문단 경계를 가급적 유지하면서 여러 문단을 하나의 청크로 묶는다.
    빈 줄 블록도 그대로 포함한다.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for block in blocks:
        # 빈 줄만 있는 블록
        if not block.strip():
            current_parts.append(block)
            current_length += len(block)
            continue

        # 현재 청크에 추가하면 제한을 넘는 경우
        if current_parts and current_length + len(block) > max_chars:
            chunks.append("".join(current_parts))
            current_parts = []
            current_length = 0

        # 문단 하나가 너무 긴 경우 문장 기준으로 추가 분할
        if len(block) > max_chars:
            if current_parts:
                chunks.append("".join(current_parts))
                current_parts = []
                current_length = 0

            chunks.extend(split_long_block(block, max_chars))
            continue

        current_parts.append(block)
        current_length += len(block)

    if current_parts:
        chunks.append("".join(current_parts))

    return chunks


def split_long_block(text: str, max_chars: int) -> list[str]:
    """
    지나치게 긴 문단을 문장부호 근처에서 나눈다.
    일본어와 영어 문장부호를 함께 처리한다.
    """
    sentences = re.split(r"(?<=[。！？.!?])", text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current += sentence

    if current:
        chunks.append(current)

    return chunks


# =========================
# 프롬프트
# =========================

def make_system_prompt(
    source_language: str,
    target_language: str,
    glossary: dict,
    current_text: str = "",
) -> str:
    glossary_text = format_glossary(glossary)

    if source_language == "Japanese" and target_language == "Korean":
        reading_hints = extract_furigana_hints(current_text)

        language_specific_rules = f"""
{make_japanese_proper_noun_rules(reading_hints)}

일본어 문체 처리 규칙:

1. 일본어에서 생략된 주어를 불필요하게 보충하지 않는다.
2. 원문의 1인칭과 인물별 말투 차이를 한국어에 반영한다.
3. 존댓말, 반말, 호칭과 인물 관계를 일관되게 유지한다.
4. 일본어식 직역투는 피하되 원문의 의미를 생략하지 않는다.
5. 문장 끝의 말투와 감정적 뉘앙스를 살린다.
""".strip()
        script_rules = """
11. 최종 번역문에는 히라가나, 가타카나, 일본식 한자가 남아 있어서는 안 됩니다.
12. 일본어 감탄사, 비명, 의성어와 의태어도 자연스러운 한글로 완전히 옮깁니다.
13. 한글과 일본어 문자를 섞은 표기를 만들지 않습니다.
14. 예: ひゃっ → "꺅", "히익", "으악" 등 문맥에 맞는 한국어 표현
15. 예: えっ → "어?", "뭐?", "엣?" 중 문맥에 맞는 표현
16. 예: きゃあ → "꺄악", "꺅"
""".strip()

    elif source_language == "Korean" and target_language == "Japanese":
        language_specific_rules = """
한국어 문체 처리 규칙:

1. 자연스러운 일본어 소설 문체와 문장 호흡으로 번역한다.
2. 존댓말, 반말, 호칭과 인물 관계를 일본어에서도 일관되게 유지한다.
3. 인물별 1인칭과 말투를 문맥에 맞게 선택하고 이후에도 유지한다.
4. 한국 인명과 지명은 용어집 표기를 우선하고, 없으면 원래 발음을 기준으로 음역한다.
5. 작품 설정상 일본식 명칭으로 확정된 항목은 용어집의 한자·가나 표기를 따른다.
6. 한국어 감탄사, 비명, 의성어와 의태어를 자연스러운 일본어 표현으로 옮긴다.
7. 원문에 없는 일본식 설정이나 문화 요소를 임의로 추가하지 않는다.
""".strip()
        script_rules = """
11. 최종 번역문에는 한글이 남아 있어서는 안 됩니다.
12. 한국어 감탄사, 비명, 의성어와 의태어도 일본어로 완전히 옮깁니다.
13. 일본어와 한글을 섞은 표기를 만들지 않습니다.
14. 한국 인명은 용어집에 다른 표기가 없으면 가타카나 음역을 우선합니다.
""".strip()
    else:
        raise ValueError(
            f"지원하지 않는 번역 방향입니다: "
            f"{source_language} → {target_language}"
        )

    return f"""
당신은 {source_language} 소설을 자연스러운 {target_language}로 번역하는
전문 문학 번역가입니다.

다음 원칙을 반드시 지키십시오.

1. 원문의 의미, 분위기, 감정과 정보량을 보존합니다.
2. 직역투를 피하고 자연스러운 {target_language} 소설 문체로 번역합니다.
3. 인물의 말투, 존댓말, 반말과 성격을 일관되게 유지합니다.
4. 원문에 없는 설명이나 내용을 추가하지 않습니다.
5. 원문 내용을 요약하거나 생략하지 않습니다.
6. 대사와 서술의 구분을 유지합니다.
7. 문단과 줄바꿈 구조를 가능한 한 보존합니다.
8. 고유명사 표기를 번역 전체에서 일관되게 유지합니다.
9. 번역문 외의 설명, 주석, 서문, 감상은 출력하지 않습니다.
10. 이전 문맥은 참고만 하고 현재 번역 대상만 출력합니다.
{script_rules}

언어별 세부 규칙:

{language_specific_rules}

용어집:

{glossary_text}
""".strip()

def make_user_prompt(
    text: str,
    target_language: str,
    previous_source: str = "",
    previous_translation: str = "",
) -> str:
    """
    현재 번역 대상과 이전 문맥을 사용자 프롬프트로 구성한다.
    """
    sections: list[str] = []

    if previous_source or previous_translation:
        sections.append(
            f"""
[이전 문맥 참고용]
이전 원문:
{previous_source or "없음"}

이전 번역:
{previous_translation or "없음"}

위 문맥은 말투, 지시 대상, 고유명사 일관성만 참고하십시오.
이전 문맥은 다시 번역하거나 출력하지 마십시오.
""".strip()
        )

    sections.append(
        f"""
[현재 번역 대상 시작]
{text}
[현재 번역 대상 끝]

현재 번역 대상만 {target_language}로 번역하십시오.
""".strip()
    )

    return "\n\n".join(sections)

# =========================
# 번역 요청
# =========================

def translate_chunk(
    text: str,
    glossary: dict,
    source_language: str,
    target_language: str,
    previous_source: str = "",
    previous_translation: str = "",
) -> str:
    glossary_search_text = (
        previous_source
        + "\n"
        + text
    )

    relevant_glossary = filter_glossary_for_text(
        glossary=glossary,
        text=glossary_search_text,
    )

    print(
        f"적용 용어 수: "
        f"{len(relevant_glossary)}/{len(glossary)}"
    )

    messages = [
        {
            "role": "system",
            "content": make_system_prompt(
                source_language=source_language,
                target_language=target_language,
                glossary=relevant_glossary,
                current_text=text,
            ),
        },
        {
            "role": "user",
            "content": make_user_prompt(
                text=text,
                target_language=target_language,
                previous_source=previous_source,
                previous_translation=previous_translation,
            ),
        },
    ]

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print("모델 응답을 기다리는 중...", flush=True)

            started_at = time.perf_counter()

            stream = ollama.chat(
                model=TRANSLATION_MODEL_NAME,
                messages=messages,
                options={
                    "temperature": 0.1,
                    "num_ctx": 4096,

                    # 번역문이 끝없이 길어지는 것을 방지
                    "num_predict": 1600,
                },
                keep_alive="10m",
                stream=True,
                think=False,
            )

            translated_parts: list[str] = []
            first_content_received = False
            thinking_detected = False
            thinking_chunks = 0

            print()
            print("----- 실시간 번역 시작 -----")

            for chunk in stream:
                message = chunk.message

                # 모델이 별도의 thinking 출력을 보내는 경우
                thinking = getattr(message, "thinking", "") or ""

                if thinking:
                    thinking_chunks += 1

                    if not thinking_detected:
                        print(
                            "모델이 번역 내용을 구성하고 있습니다.",
                            flush=True,
                        )
                        thinking_detected = True

                    # 너무 자주 출력하지 않고 활동 표시만 한다.
                    if thinking_chunks % 20 == 0:
                        print(".", end="", flush=True)

                content = message.content or ""

                if content:
                    if not first_content_received:
                        if thinking_detected:
                            print()

                        print("첫 번역 응답을 받았습니다.")
                        first_content_received = True

                    translated_parts.append(content)

                    # 생성되는 번역문을 콘솔에 실시간 출력
                    print(content, end="", flush=True)

            print()
            print("----- 실시간 번역 완료 -----")

            translated = "".join(translated_parts).strip()

            if not translated:
                raise RuntimeError(
                    "모델이 빈 번역문을 반환했습니다."
                )

            elapsed = time.perf_counter() - started_at

            print(f"번역 응답 시간: {elapsed:.1f}초")

            return translated

        except Exception as error:
            last_error = error

            print()
            print(
                f"번역 요청 실패 "
                f"({attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt < MAX_RETRIES:
                wait_seconds = 2 ** attempt

                print(
                    f"{wait_seconds}초 후 다시 시도합니다."
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"{MAX_RETRIES}번 재시도했지만 번역하지 못했습니다."
    ) from last_error
    
    
#파일 이름 번역

def sanitize_filename(filename: str) -> str:
    """
    Windows 파일명에 사용할 수 없는 문자를 제거한다.
    """
    sanitized = re.sub(r'[\\/:*?"<>|]', "", filename)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized.rstrip(". ")

    if not sanitized:
        return "번역본"

    # 파일명이 지나치게 길어지는 것을 방지
    return sanitized[:100]



def extract_title_and_body(text: str) -> tuple[str, str]:
    """
    첫 번째 비어 있지 않은 줄을 제목으로 간주한다.
    나머지는 본문으로 반환한다.
    """
    lines = text.splitlines(keepends=True)

    title_index: int | None = None

    for index, line in enumerate(lines):
        if line.strip():
            title_index = index
            break

    if title_index is None:
        raise ValueError("입력 파일에 내용이 없습니다.")

    title = lines[title_index].strip()

    body_lines = (
        lines[:title_index]
        + lines[title_index + 1:]
    )

    body = "".join(body_lines).lstrip("\r\n")

    return title, body



def translate_title(
    title: str,
    glossary: dict,
    source_language: str,
    target_language: str,
) -> str:
    if source_language == "Japanese":
        proper_noun_rule = (
            "일본 인명과 지명은 일본어 발음에 따라 "
            "한글로 음역합니다."
        )
    else:
        proper_noun_rule = (
            "한국 인명과 지명은 용어집 표기를 우선하고, "
            "없으면 원래 발음에 따라 일본어로 음역합니다."
        )

    messages = [
        {
            "role": "system",
            "content": f"""
당신은 {source_language} 소설 제목을 {target_language}로 번역하는 번역가입니다.

규칙:
1. 소설 제목답게 자연스럽고 간결하게 번역합니다.
2. 원문의 의미와 분위기를 유지합니다.
3. {proper_noun_rule}
4. 용어집 표기를 우선합니다.
5. 번역된 제목만 출력합니다.
6. 설명, 따옴표, 후보 목록은 출력하지 않습니다.

용어집:
{format_glossary(glossary)}
""".strip(),
        },
        {
            "role": "user",
            "content": title,
        },
    ]

    response = ollama.chat(
        model=TITLE_MODEL_NAME,
        messages=messages,
        think=False,
        options={
            "temperature": 0.1,
            "num_ctx": 2048,
            "num_predict": 100,
        },
    )

    translated_title = (response.message.content or "").strip()

    if not translated_title:
        raise RuntimeError("제목 번역 결과가 비어 있습니다.")

    return translated_title
# =========================
# 진행 상태 저장
# =========================

def load_progress(path: Path) -> dict:
    if not path.exists():
        return {
            "completed_chunks": 0,
            "translations": [],
        }

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def save_progress(path: Path, progress: dict) -> None:
    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            progress,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


# =========================
# 전체 번역
# =========================

def prepare_glossary_candidates(
    input_path: Path,
    direction_key: str,
) -> None:
    """원문을 청크별로 분석해 검토용 용어집 후보를 만든다."""
    direction = TRANSLATION_DIRECTIONS[direction_key]
    source_language = direction["source_language"]
    target_language = direction["target_language"]
    glossary_path = direction["glossary_path"]
    candidates_path = direction["candidates_path"]

    print()
    print(f"입력 파일: {input_path.name}")
    print()

    source_text = read_source_text(input_path, source_language)

    if not source_text.strip():
        raise ValueError("입력 파일에 내용이 없습니다.")

    print(
        f"번역 방향: {direction['source_label']} → "
        f"{direction['target_label']}"
    )

    blocks = split_into_blocks(source_text)
    chunks = build_chunks(blocks)

    if not chunks:
        raise ValueError("용어를 추출할 청크를 생성하지 못했습니다.")

    existing_glossary = load_glossary(glossary_path)
    candidates: dict = {}

    for index, chunk in enumerate(chunks, start=1):
        if not chunk.strip():
            continue

        print(f"용어집 후보 추출 중: {index}/{len(chunks)}")

        chunk_candidates = generate_glossary_candidates(
            source_text=chunk,
            source_language=source_language,
            target_language=target_language,
        )
        candidates = merge_candidate_results(
            candidates,
            chunk_candidates,
        )

        for source in existing_glossary:
            candidates.pop(source, None)

        save_json_file(candidates_path, candidates)

    if source_language == "Japanese":
        reading_hints = extract_furigana_hints(source_text)
        candidates = apply_furigana_hints(
            candidates,
            reading_hints,
        )

    for source in existing_glossary:
        candidates.pop(source, None)

    save_json_file(candidates_path, candidates)

    print()
    print(f"용어집 후보 저장: {candidates_path.name}")
    print(f"추가된 용어집 후보: {len(candidates)}개")

    if candidates:
        print()
        print("----- 추가된 후보 검토 -----")

        for source, entry in candidates.items():
            if isinstance(entry, str):
                print(f"- {source} → {entry}")
                continue

            target = get_glossary_target(entry) or "미입력"
            reading = entry.get("reading", "") or "미확인"
            entry_type = entry.get("type", "") or "미분류"

            print(
                f"- {source} → {target} "
                f"(독음: {reading}, 분류: {entry_type})"
            )

        print("----- 후보 검토 끝 -----")

    print(
        "후보를 검토한 뒤 확정 항목을 "
        f"{glossary_path.name}에 반영하세요."
    )


def translate_novel(input_path: Path, direction_key: str) -> None:
    direction = TRANSLATION_DIRECTIONS[direction_key]
    source_language = direction["source_language"]
    target_language = direction["target_language"]
    glossary_path = direction["glossary_path"]

    print()
    print(f"입력 파일: {input_path.name}")
    print()

    source_text = read_source_text(input_path, source_language)

    if not source_text.strip():
        raise ValueError("입력 파일에 번역할 내용이 없습니다.")

    print(
        f"번역 방향: {direction['source_label']} → "
        f"{direction['target_label']}"
    )

    # 용어집 불러오기
    glossary = load_glossary(glossary_path)

    # 첫 번째 비어 있지 않은 줄을 제목으로 분리

    if FIRST_LINE_IS_TITLE:
        original_title, body_text = extract_title_and_body(source_text)
    else:
        original_title = input_path.stem
        body_text = source_text

    print(f"원문 제목: {original_title}")
    print("제목을 번역합니다.")

    translated_title = translate_title(
        title=original_title,
        glossary=glossary,
        source_language=source_language,
        target_language=target_language,
    )

    print(f"번역 제목: {translated_title}")

    output_path, progress_path = make_output_paths(
        input_path=input_path,
        direction_key=direction_key,
        translated_title=translated_title,
    )

    print(f"출력 파일: {output_path.name}")
    print()

    source_text = body_text

    if not source_text.strip():
        raise ValueError(
            "제목을 제외한 본문에 번역할 내용이 없습니다."
        )

    print(f"본문 전체 문자 수: {len(source_text)}")
    print(f"본문 미리보기: {source_text[:100]!r}")

    # 텍스트 분할
    blocks = split_into_blocks(source_text)
    chunks = build_chunks(blocks)

    if not chunks:
        raise ValueError("번역할 청크를 생성하지 못했습니다.")

    # 기존 진행 상태 불러오기
    progress = load_progress(progress_path)

    completed_chunks = int(
        progress.get("completed_chunks", 0)
    )

    translations: list[str] = progress.get(
        "translations",
        [],
    )

    saved_source_file = progress.get("source_file")
    saved_direction = progress.get("direction")

    # 다른 파일의 진행 기록이면 초기화
    if (
        (saved_source_file and saved_source_file != input_path.name)
        or (saved_direction and saved_direction != direction_key)
    ):
        print(
            "진행 파일과 입력 파일이 일치하지 않아 "
            "처음부터 번역합니다."
        )

        completed_chunks = 0
        translations = []

    # 진행 기록이 현재 청크 수보다 크면 초기화
    if completed_chunks > len(chunks):
        print(
            "기존 진행 정보가 현재 원문과 맞지 않아 "
            "처음부터 번역합니다."
        )

        completed_chunks = 0
        translations = []

    # 완료 청크 수와 번역문 개수가 다르면 안전하게 보정
    if len(translations) != completed_chunks:
        print(
            "진행 정보가 불완전하여 "
            "처음부터 번역합니다."
        )

        completed_chunks = 0
        translations = []

    print(f"전체 청크 수: {len(chunks)}")
    print(f"완료된 청크 수: {completed_chunks}")
    print()

    # 번역 시작
    for index in range(completed_chunks, len(chunks)):
        chunk = chunks[index]

        print(f"청크 {index + 1} 처리 시작")
        print(f"청크 문자 수: {len(chunk)}")
        print(f"공백 제외 문자 수: {len(chunk.strip())}")

        # 공백뿐인 청크는 그대로 보존
        if not chunk.strip():
            print("공백 청크이므로 번역하지 않고 보존합니다.")
            translated = chunk

        else:
            previous_source = (
                chunks[index - 1][-CONTEXT_CHARS:]
                if index > 0
                else ""
            )

            previous_translation = (
                translations[index - 1][-CONTEXT_CHARS:]
                if index > 0 and translations
                else ""
            )

            print(
                f"[{index + 1}/{len(chunks)}] "
                f"{len(chunk)}자 번역 중..."
            )
            print("Ollama에 번역 요청을 보냅니다.")

            translated = translate_chunk(
                text=chunk,
                glossary=glossary,
                source_language=source_language,
                target_language=target_language,
                previous_source=previous_source,
                previous_translation=previous_translation,
            )

            print("Ollama 응답을 받았습니다.")

            # 원문의 청크 끝 줄바꿈 보존
            if chunk.endswith("\n\n"):
                translated = translated.rstrip() + "\n\n"

            elif chunk.endswith("\n"):
                translated = translated.rstrip() + "\n"

        translations.append(translated)

        # 진행 상태 저장
        progress = {
            "source_file": input_path.name,
            "direction": direction_key,
            "completed_chunks": index + 1,
            "total_chunks": len(chunks),
            "translations": translations,
        }

        save_progress(progress_path, progress)

        # 번역 결과 즉시 저장
        output_path.write_text(
            "".join(translations),
            encoding="utf-8",
        )

        print(
            f"청크 {index + 1} 저장 완료 "
            f"({index + 1}/{len(chunks)})"
        )
        print()

    print(f"번역 완료: {output_path}")

    # 모든 번역이 끝나면 진행 파일 삭제
    if progress_path.exists():
        progress_path.unlink()
        
        
# =========================
# 프로그램 종료
# =========================
        
def main() -> int:
    try:
        while True:
            operation = select_operation()

            if operation == "quit":
                print("프로그램을 종료합니다.")
                return 0

            direction_key = select_translation_direction()

            if direction_key == "back":
                continue

            input_path = select_input_file()

            if input_path is None:
                print("파일 선택을 취소했습니다.")
                continue

            if operation == "extract_glossary":
                prepare_glossary_candidates(input_path, direction_key)
            else:
                translate_novel(input_path, direction_key)

            print()
            print("작업이 완료되었습니다.")
            print("다른 작업을 계속 선택할 수 있습니다.")

    except KeyboardInterrupt:
        print()
        print("사용자가 번역을 중단했습니다.")
        print("다음 실행 시 저장된 지점부터 이어집니다.")
        return 130

    except Exception as error:
        print()
        print("번역 중 오류가 발생했습니다.")
        print(f"오류 내용: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())