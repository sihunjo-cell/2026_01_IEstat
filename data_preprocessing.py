# Auto-generated from data_preprocessing.ipynb (code cells only, markdown/outputs stripped)

# CPU나 RAM은 영어 한국어 매칭이 안되기 때문에 deep-translator이용
# 구글 번역기 기능을 사용하는 모듈
# [nb-magic] !pip install deep-translator

# 실행속도 과도하게 오래걸리는 문제로 인해 해당되는 것들 mapping해주는 딕셔너리만들기로 해결
from deep_translator import GoogleTranslator

def kor_to_eng(word):
    # 한국어를 영어로 번역하는 번역기 설정
    translator = GoogleTranslator(source='ko', target='en')

    # 번역 테스트
    translated_word = translator.translate(word)
    return translated_word

print(kor_to_eng("AMD 라이젠 스레드리퍼 1900X (화이트헤븐)"))

# 우선 GPU모델명을 기존 데이터 셋에서 불러온다.
import pandas as pd
import re

# kaggle GPU specs 데이터셋

GPU_SPECS = pd.read_csv("./gpu_1986-2026.csv")
print(f"{GPU_SPECS.shape[0]}행 {GPU_SPECS.shape[1]}열")

# 2017년 이전의 GPU 모델은 분석에서 제외 
import re

# 1900년도, 2000년도의 숫자 패턴
year_pattern = re.compile(r'(?<!\d)(?:19\d{2}|20\d{2})(?!\d)')

def keep_row(val):
    years = [int(y) for y in year_pattern.findall(str(val))]
    return bool(years) and not any(y < 2017 for y in years)

# 다나와 데이터는 geforce를 지포스라고 저장해두어 패턴 매칭 시 문제될 경우 사전에 방지
GPU_SPECS = GPU_SPECS[GPU_SPECS["source_file"].apply(keep_row)].copy()
GPU_SPECS["Name"] = GPU_SPECS["Name"].str.replace(r"^\S+\s*", "", regex=True)
print(f"{GPU_SPECS.shape[0]} 행으로 줄음")
GPU_NAMES = GPU_SPECS["Name"].to_list()
print(GPU_NAMES)

import pandas as pd

MARCH2026_GPU = pd.read_csv("./GPU_PREPARE/2026_03_GPU.csv")

# 3월 기준 993행 33열
print(f"행: {MARCH2026_GPU.shape[0]}, 열: {MARCH2026_GPU.shape[1]}")

# 시간을 시,분,초 날리고 날짜별 string으로 변환 (index기준 2열부터 날짜 시작 날짜는 31일기준 빈거 없음)
time_range = [str(x).split(' ')[0] for x in MARCH2026_GPU.columns[2:]]
MARCH2026_GPU.columns = list(MARCH2026_GPU.columns[:2]) + time_range


# 일단 strip으로 빈공간 정리
def norm_text(s):
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 모델 리스트 정규화 + 중복 제거 + 긴 것부터 정렬
gpu_model_list = sorted(
    {norm_text(x) for x in GPU_NAMES if str(x).strip()},
    key=len,
    reverse=True
)

# 모델명 regex
# 긴 모델명이 먼저 매칭되도록 길이순 정렬 후 join
model_pattern = re.compile(
    "|".join(re.escape(m) for m in gpu_model_list),
    flags=re.IGNORECASE
)

# 맨 뒤 괄호만 추출
tail_paren_pattern = re.compile(r"\(([^()]*)\)\s*$")

# 용량 추출
capacity_pattern = re.compile(r"(?<!\d)(\d+\s*(?:TB|GB|MB))(?!\w)", flags=re.IGNORECASE)


# Name 하나 분리
def split_name(name):
    s = norm_text(name)

    # 뒤 괄호 추출
    paren_match = tail_paren_pattern.search(s)
    if paren_match:
        tail_paren = paren_match.group(1).strip()
        s_wo_paren = tail_paren_pattern.sub("", s).strip()
    else:
        tail_paren = "0"
        s_wo_paren = s

    # 용량 추출
    cap_match = capacity_pattern.search(s_wo_paren)
    if cap_match:
        capacity = re.sub(r"\s+", "", cap_match.group(1).upper())
        s_wo_cap = (s_wo_paren[:cap_match.start()] + " " + s_wo_paren[cap_match.end():]).strip()
        s_wo_cap = re.sub(r"\s+", " ", s_wo_cap)
    else:
        capacity = "0"
        s_wo_cap = s_wo_paren

    # 모델명 추출
    model_match = model_pattern.search(s_wo_cap)
    if model_match:
        gpu_model = norm_text(model_match.group(0))
        s_rest = (s_wo_cap[:model_match.start()] + " " + s_wo_cap[model_match.end():]).strip()
        s_rest = re.sub(r"\s+", " ", s_rest)
    else:
        gpu_model = "0"
        s_rest = s_wo_cap

    # 나머지 문자열 정리
    name_rest = norm_text(s_rest)
    if not name_rest:
        name_rest = "0"

    # gpu_model은 해당 모델 이름, capacity는 용량, tail_paren은 중고인지 해외배송인지
    return pd.Series({
        "gpu_model": gpu_model,
        "capacity": capacity,
        "tail_paren": tail_paren,
        "name_rest": name_rest
    })


# 적용
result = MARCH2026_GPU["Name"].apply(split_name)
MARCH2026_GPU = pd.concat([MARCH2026_GPU, result], axis=1)
MARCH2026_GPU = MARCH2026_GPU[MARCH2026_GPU["gpu_model"] != "0"]
print(len(MARCH2026_GPU))
print(MARCH2026_GPU["tail_paren"])

from pathlib import Path
import pandas as pd
import re

folder_path = Path("./GPU_PREPARE")

def process_GPU(file_path):
    df = pd.read_csv(file_path)
    print(f"[원본] {file_path.name} -> 행: {df.shape[0]}, 열: {df.shape[1]}")

    # 날짜 컬럼명 정리
    time_range = [str(x).split(" ")[0] for x in df.columns[2:]]
    df.columns = list(df.columns[:2]) + time_range

    def norm_text(s):
        s = "" if pd.isna(s) else str(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    gpu_model_list = sorted(
        {norm_text(x) for x in GPU_NAMES if str(x).strip()},
        key=len,
        reverse=True
    )

    model_pattern = re.compile(
        "|".join(re.escape(m) for m in gpu_model_list),
        flags=re.IGNORECASE
    )

    tail_paren_pattern = re.compile(r"\(([^()]*)\)\s*$")
    capacity_pattern = re.compile(r"(?<!\d)(\d+\s*(?:TB|GB|MB))(?!\w)", flags=re.IGNORECASE)

    def split_name(name):
        s = norm_text(name)

        # 뒤 괄호 추출
        paren_match = tail_paren_pattern.search(s)
        if paren_match:
            tail_paren = paren_match.group(1).strip()
            s_wo_paren = tail_paren_pattern.sub("", s).strip()
        else:
            tail_paren = "0"
            s_wo_paren = s

        # 용량 추출
        cap_match = capacity_pattern.search(s_wo_paren)
        if cap_match:
            capacity = re.sub(r"\s+", "", cap_match.group(1).upper())
            s_wo_cap = (s_wo_paren[:cap_match.start()] + " " + s_wo_paren[cap_match.end():]).strip()
            s_wo_cap = re.sub(r"\s+", " ", s_wo_cap)
        else:
            capacity = "0"
            s_wo_cap = s_wo_paren

        # 모델명 추출
        model_match = model_pattern.search(s_wo_cap)
        if model_match:
            gpu_model = norm_text(model_match.group(0))
            s_rest = (s_wo_cap[:model_match.start()] + " " + s_wo_cap[model_match.end():]).strip()
            s_rest = re.sub(r"\s+", " ", s_rest)
        else:
            gpu_model = "0"
            s_rest = s_wo_cap

        name_rest = norm_text(s_rest)
        if not name_rest:
            name_rest = "0"

        return pd.Series({
            "gpu_model": gpu_model,
            "capacity": capacity,
            "tail_paren": tail_paren,
            "name_rest": name_rest
        })

    result = df["Name"].apply(split_name)
    df = pd.concat([df, result], axis=1)

    # gpu_model 못 찾은 행 제거
    df = df[df["gpu_model"] != "0"].copy()

    print(f"[가공 후] 행: {df.shape[0]}, 열: {df.shape[1]}")

    return df


for file in folder_path.glob("*.csv"):
    if not file.is_file():
        continue

    # 이미 만든 결과 파일은 다시 처리하지 않음
    if file.name.startswith("processed_"):
        continue

    processed_df = process_GPU(file)

    save_path = folder_path / f"processed_{file.name}"
    processed_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {save_path}")
    print(f"{file.name}은 끝")

from pathlib import Path
import pandas as pd
import re

folder_path = Path("./CPU_PREPARE")

def process_CPU(file_path):
    df = pd.read_csv(file_path)
    print(f"[원본] {file_path.name} -> 행: {df.shape[0]}, 열: {df.shape[1]}")

    # 날짜 컬럼명 정리
    time_range = [str(x).split(" ")[0] for x in df.columns[2:]]
    df.columns = list(df.columns[:2]) + time_range
    date_cols = list(df.columns[2:])

    def norm_text(s):
        s = "" if pd.isna(s) else str(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Name: 맨 뒤 괄호만 코드네임으로 분리
    tail_paren_pattern = re.compile(r"\(([^()]*)\)\s*$")

    # 날짜 셀: 판매형태_가격
    value_pattern = re.compile(r"^(.*?)_([\d,]+)$")

    def split_name(name):
        s = norm_text(name)

        paren_match = tail_paren_pattern.search(s)
        if paren_match:
            cpu_codename = paren_match.group(1).strip()
            cpu_model = tail_paren_pattern.sub("", s).strip()
        else:
            cpu_codename = "0"
            cpu_model = s

        if not cpu_model:
            cpu_model = "0"

        return pd.Series({
            "cpu_model": cpu_model,
            "cpu_codename": cpu_codename
        })
    def split_value(x):
        s = norm_text(x)

        if s == "" or s == "0":
            return "0", 0

        m = value_pattern.match(s)
        if m:
            sale_type = norm_text(m.group(1))
            price_krw = int(m.group(2).replace(",", ""))
            return sale_type if sale_type else "0", price_krw

        # 언더바 없이 숫자만 있는 경우
        digits = re.sub(r"[^\d]", "", s)
        if digits:
            return "0", int(digits)

        return "0", 0

    # Name 분리
    name_result = df["Name"].apply(split_name)
    df = pd.concat([df, name_result], axis=1)

    # 날짜 컬럼은 가격만 남기고, 부가정보는 sale_info 하나로 묶기
    sale_info_list = []

    for idx, row in df.iterrows():
        info_set = set()

        for col in date_cols:
            sale_type, price_krw = split_value(row[col])
            df.at[idx, col] = price_krw

            if sale_type != "0":
                info_set.add(sale_type)

        sale_info = " | ".join(sorted(info_set)) if info_set else "0"
        sale_info_list.append(sale_info)

    df["sale_info"] = sale_info_list

    # 날짜 컬럼 숫자형 정리
    for col in date_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 컬럼 순서 정리
    df = df[["Id", "Name", "cpu_model", "cpu_codename", "sale_info"] + date_cols]

    print(f"[가공 후] 행: {df.shape[0]}, 열: {df.shape[1]}")
   
    return df


for file in folder_path.glob("*.csv"):
    if not file.is_file():
        continue

    if file.name.startswith("processed_"):
        continue

    processed_df = process_CPU(file)

    save_path = folder_path / f"processed_{file.name}"
    processed_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {save_path}")
    print(f"{file.name}은 끝")

from pathlib import Path
import pandas as pd
import re

folder_path = Path("./RAM_PREPARE")

def process_RAM(file_path):
    df = pd.read_csv(file_path)
    print(f"[원본] {file_path.name} -> 행: {df.shape[0]}, 열: {df.shape[1]}")

    # 날짜 컬럼명 정리
    time_range = [str(x).split(" ")[0] for x in df.columns[2:]]
    df.columns = list(df.columns[:2]) + time_range
    date_cols = list(df.columns[2:])

    def norm_text(s):
        s = "" if pd.isna(s) else str(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Name에서 DDR세대/속도, CL만 뽑고 나머지는 ram_model로 둠
    ddr_pattern = re.compile(r"\b(DDR[345])(?:-(\d+))?\b", flags=re.IGNORECASE)
    cl_pattern = re.compile(r"\bCL\s?(\d+)\b", flags=re.IGNORECASE)

    # 날짜 셀에서 용량/GB당가격/총가격 추출
    # 예: _32GB_13,277원/1GB_424,860
    value_pattern = re.compile(
        r"_*(\d+GB(?:\([^)]*\))?)_([\d,]+)원/1GB_([\d,]+)",
        flags=re.IGNORECASE
    )

    def split_name(name):
        s = norm_text(name)

        ddr_match = ddr_pattern.search(s)
        if ddr_match:
            memory_gen = ddr_match.group(1).upper()
            memory_speed = ddr_match.group(2) if ddr_match.group(2) else "0"
            s = ddr_pattern.sub("", s).strip()
        else:
            memory_gen = "0"
            memory_speed = "0"

        cl_match = cl_pattern.search(s)
        if cl_match:
            cl_value = f"CL{cl_match.group(1)}"
            s = cl_pattern.sub("", s).strip()
        else:
            cl_value = "0"

        ram_model = norm_text(s)
        if not ram_model:
            ram_model = "0"

        return pd.Series({
            "ram_model": ram_model,
            "memory_gen": memory_gen,
            "memory_speed": memory_speed,
            "cl_value": cl_value
        })

    # Name 분리
    name_result = df["Name"].apply(split_name)
    df = pd.concat([df, name_result], axis=1)

    rows = []

    for _, row in df.iterrows():
        # 같은 Name 안의 여러 capacity를 따로 저장할 dict
        # key = capacity, value = 날짜별 총가격
        cap_price_map = {}

        for col in date_cols:
            cell = norm_text(row[col])

            if cell == "" or cell == "0":
                continue

            # 한 셀 안에 | 로 여러 용량이 같이 들어있음
            chunks = [x.strip() for x in cell.split("|") if norm_text(x)]

            for ch in chunks:
                m = value_pattern.search(ch)
                if not m:
                    continue

                capacity = m.group(1).upper()          # 32GB, 16GB(8Gx2) 등
                per_gb_price = int(m.group(2).replace(",", ""))   # 읽기만 함
                total_price = int(m.group(3).replace(",", ""))    # 최종 날짜값으로 씀

                if capacity not in cap_price_map:
                    cap_price_map[capacity] = {d: 0 for d in date_cols}

                cap_price_map[capacity][col] = total_price

        # capacity별로 행 생성
        for capacity, date_price_dict in cap_price_map.items():
            cap_match = re.match(r"(\d+)GB", capacity)
            capacity_gb = int(cap_match.group(1)) if cap_match else 0

            new_row = {
                "Id": row["Id"],
                "Name": row["Name"],
                "ram_model": row["ram_model"],
                "memory_gen": row["memory_gen"],
                "memory_speed": row["memory_speed"],
                "cl_value": row["cl_value"],
                "capacity": capacity,
                "capacity_gb": capacity_gb
            }

            for d in date_cols:
                new_row[d] = date_price_dict[d]

            rows.append(new_row)

    result_df = pd.DataFrame(rows)

    # 날짜 컬럼 숫자형 정리
    for col in date_cols:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(0).astype(int)

    # 컬럼 순서
    result_df = result_df[
        ["Id", "Name", "ram_model", "memory_gen", "memory_speed", "cl_value", "capacity", "capacity_gb"] + date_cols
    ]

    print(f"[가공 후] 행: {result_df.shape[0]}, 열: {result_df.shape[1]}")
    
    return result_df


for file in folder_path.glob("*.csv"):
    if not file.is_file():
        continue

    if file.name.startswith("processed_"):
        continue

    processed_df = process_RAM(file)

    save_path = folder_path / f"processed_{file.name}"
    processed_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {save_path}")
    print(f"{file.name}은 끝")

import numpy as np
import pandas as pd
import re

GPU_
