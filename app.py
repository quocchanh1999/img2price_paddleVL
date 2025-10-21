import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from rapidfuzz import process, fuzz
import unicodedata
import nltk
from nltk.stem.snowball import SnowballStemmer
import easyocr
from PIL import Image
import io
import asyncio
import sys
import google.generativeai as genai
import os

class LogScaler:
    def transform(self, x):
        return np.log1p(x)
    def inverse_transform(self, x):
        return np.expm1(x)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(layout="centered", page_title="Dự đoán Giá thuốc")

@st.cache_resource
def initialize_tools():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    stemmer = SnowballStemmer("english")
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    return stemmer, reader

stemmer, ocr_reader = initialize_tools()

@st.cache_resource
def load_artifacts():
    try:
        model_giaThanh = joblib.load(os.path.join(BASE_DIR, "final_model_giaThanh.joblib"))
        model_giaBanBuon = joblib.load(os.path.join(BASE_DIR, "final_model_giaBanBuon.joblib"))
        imputer = joblib.load(os.path.join(BASE_DIR, "imputer.joblib"))
        scaler_giaThanh = joblib.load(os.path.join(BASE_DIR, "scaler_giaThanh.joblib"))
        scaler_giaBanBuon = joblib.load(os.path.join(BASE_DIR, "scaler_giaBanBuon.joblib"))
        train_cols = joblib.load(os.path.join(BASE_DIR, "train_cols.joblib"))
        # Remove duplicates from train_cols
        train_cols = list(dict.fromkeys(train_cols))
        df_full = pd.read_excel(os.path.join(BASE_DIR, "dichvucong_medicines_Final.xlsx"))
        return model_giaThanh, model_giaBanBuon, imputer, scaler_giaThanh, scaler_giaBanBuon, train_cols, df_full
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

model_giaThanh, model_giaBanBuon, imputer, scaler_giaThanh, scaler_giaBanBuon, train_cols, df_full = load_artifacts()

categorical_cols = ['doanhNghiepSanXuat', 'nuocSanXuat', 'dangBaoChe_final', 'hoat_chat_chinh', 'loaiDongGoiChinh', 'donViCoSo', 'is_low_price']

REPLACEMENTS_DNSX = {
    'ctcp': 'công ty cổ phần', 'tnhh': 'trách nhiệm hữu hạn', 'dp': 'dược phẩm', 'tw': 'trung ương',
    'cty': 'công ty', 'ct': 'công ty', 'cp': 'cổ phần', 'sx': 'sản xuất', 'tm': 'thương mại',
    'ld': 'liên doanh', 'mtv': 'một thành viên'
}
GENERIC_TERMS_DNSX = [
    'công ty cổ phần', 'công ty tnhh', 'công ty', 'trách nhiệm hữu hạn', 'một thành viên',
    'liên doanh', 'cổ phần', 'sản xuất', 'thương mại', 'trung ương', 'limited', 'ltd', 'pvt',
    'inc', 'corp', 'corporation', 'gmbh', 'co', 'kg', 'ag', 'srl', 'international',
    'pharma', 'pharmaceuticals', 'pharmaceutical', 'laboratories', 'industries'
]
GENERIC_TERMS_DNSX.sort(key=len, reverse=True)

def ultimate_company_name_cleaner(name_series, stemmer):
    def clean_single_name(name):
        name = str(name).lower()
        name = re.sub(r'\([^)]*\)', '', name)
        name = re.sub(r'[^a-z0-9\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        for old, new in REPLACEMENTS_DNSX.items():
            name = name.replace(old, new)
        tokens = name.split()
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token]
        name = " ".join(stemmed_tokens)
        for term in GENERIC_TERMS_DNSX:
            stemmed_term = " ".join([stemmer.stem(t) for t in term.split()])
            if stemmed_term:
                name = name.replace(stemmed_term, '')
        return re.sub(r'\s+', ' ', name).strip()
    return name_series.apply(clean_single_name)

DEFINITIVE_DBC_MAP = {
    'Thuốc cấy/Que cấy': ['cấy dưới da', 'que cấy'],
    'Dạng xịt dưới lưỡi': ['xịt dưới lưỡi'],
    'Khí dung/Hít': ['khí dung', 'aerosol', 'inhaler', 'hít', 'phun mù'],
    'Thuốc đặt': ['thuốc đạn', 'viên đặt', 'đạn đặt', 'suppository', 'viên đạn'],
    'Thuốc gây mê đường hô hấp': ['gây mê', 'hô hấp'],
    'Trà túi lọc': ['trà túi lọc'],
    'Bột pha tiêm/truyền': ['bột đông khô pha tiêm', 'bột pha tiêm', 'powder for injection', 'bột và dung môi pha tiêm', 'bột đông khô', 'dung môi pha tiêm', 'bột vô khuẩn pha tiêm'],
    'Dung dịch tiêm/truyền': ['dung dịch tiêm', 'thuốc tiêm', 'bơm tiêm', 'injection', 'solution for injection', 'dịch truyền', 'dịch treo vô khuẩn', 'lọ', 'ống', 'dung dich tiêm'],
    'Hỗn dịch tiêm/truyền': ['hỗn dịch tiêm', 'suspension for injection'],
    'Nhũ tương tiêm/truyền': ['nhũ tương tiêm', 'emulsion for injection'],
    'Hoàn (YHCT)': ['hoàn mềm', 'hoàn cứng', 'viên hoàn'],
    'Cao lỏng (YHCT)': ['cao lỏng'],
    'Cao xoa/dán (YHCT)': ['cao xoa', 'cao dán'],
    'Dầu xoa/gió': ['dầu xoa', 'dầu gió', 'dầu xoa bóp', 'dầu bôi ngoài da'],
    'Kem bôi da': ['kem bôi', 'kem', 'cream'],
    'Gel bôi da': ['gel bôi', 'gel'],
    'Thuốc mỡ bôi da': ['thuốc mỡ', 'ointment', 'thuốc mỡ', 'mỡ bôi da', 'mỡ bôi ngoài da'],
    'Miếng dán': ['miếng dán', 'patch'],
    'Lotion': ['lotion'],
    'Cồn/Rượu thuốc': ['cồn thuốc', 'cồn xoa bóp', 'rượu thuốc'],
    'Nước súc miệng/Rơ miệng': ['nước súc miệng', 'rơ miệng'],
    'Dầu gội': ['dầu gội'],
    'Dung dịch nhỏ (Mắt/Mũi/Tai)': ['nhỏ mắt', 'nhỏ mũi', 'nhỏ tai', 'eye drops', 'nasal drops', 'dung dịch nhỏ mắt'],
    'Dung dịch xịt (Mũi/Tai)': ['xịt mũi', 'xịt', 'nasal spray', 'spray'],
    'Thuốc mỡ (Mắt/Mũi/Tai)': ['mỡ tra mắt', 'eye ointment'],
    'Viên nang': ['viên nang', 'nang', 'capsule', 'cap'],
    'Viên sủi': ['viên sủi', 'effervescent', 'cốm sủi bọt'],
    'Viên ngậm': ['viên ngậm', 'sublingual'],
    'Viên nén': ['viên nén', 'viên bao', 'tablet', 'nén bao', 'viên nhai', 'viên phân tán', 'viên'],
    'Siro': ['siro', 'sirô', 'siiro', 'syrup'],
    'Hỗn dịch uống': ['hỗn dịch uống', 'hỗn dịch', 'oral suspension', 'suspension'],
    'Nhũ tương uống': ['nhũ tương uống', 'nhũ tương', 'nhũ dịch uống', 'oral emulsion', 'nhỏ giọt'],
    'Dung dịch uống': ['dung dịch uống', 'oral solution', 'solution', 'thuốc nước uống', 'thuốc nước'],
    'Thuốc cốm uống': ['thuốc cốm', 'cốm pha', 'granules', 'cốm'],
    'Thuốc bột uống': ['thuốc bột pha uống', 'thuốc bột', 'powder', 'bột pha uống', 'bột'],
    'Dung dịch (Chung)': ['dung dịch'],
    'Dùng ngoài (Chung)': ['dùng ngoài', 'external', 'topical'],
    'Nguyên liệu': ['nguyên liệu', 'active ingredient'],
}

def classify_dangBaoChe_final(text):
    if pd.isnull(text):
        return "Không xác định"
    s = unicodedata.normalize('NFKC', str(text).lower())
    s = re.sub(r'[^a-z0-9à-ỹ\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    for standard_form, keywords in DEFINITIVE_DBC_MAP.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', s) for keyword in keywords):
            return standard_form
    return 'Khác (Chưa phân loại)'

def extract_quantity(text):
    if pd.isnull(text):
        return pd.Series([1.0, 1.0, 1.0, 1], index=['min_so_luong', 'max_so_luong', 'mean_so_luong', 'num_pack_options'])
    patterns = re.compile(r'[;,]\s*|\s*(?:hoặc|or)\s*', re.IGNORECASE)
    packs = [p.strip() for p in patterns.split(str(text)) if p.strip()]
    quantities = []
    for pack in packs:
        numbers = re.findall(r'(\d+\.?\d*)\s*(viên|ml|g|ống|nang|gói|túi|tuýp|chai|lọ)', pack, re.IGNORECASE)
        if not numbers:
            numbers = re.findall(r'(\d+\.?\d*)', pack)
            if not numbers:
                continue
        total = 1.0
        if isinstance(numbers[0], tuple):
            for num, unit in numbers:
                total *= float(num)
        else:
            for num in numbers:
                total *= float(num)
        quantities.append(total)
    if not quantities:
        quantities = [1.0]
    return pd.Series([min(quantities), max(quantities), np.mean(quantities), len(quantities)],
                     index=['min_so_luong', 'max_so_luong', 'mean_so_luong', 'num_pack_options'])

ULTIMATE_UNIT_CONVERSION_MAP = {
    'kg': 1_000_000, 'g': 1_000, 'mg': 1, 'mcg': 0.001, 'µg': 0.001, 'ml': 1_000, 'l': 1_000_000
}

def normalize_hamluong_to_mg(hamluong_str):
    total_mg = 0
    dosages = re.findall(r'(\d+\.?\d*)\s*(mg|g|ml|l|kg|iu|ui|%)', str(hamluong_str).lower())
    if not dosages:
        return np.nan
    for value, unit in dosages:
        value = float(value)
        if unit == 'g':
            total_mg += value * 1000
        elif unit == 'kg':
            total_mg += value * 1000000
        elif unit == 'l':
            total_mg += value * 1000
        elif unit == 'ml':
            total_mg += value
        elif unit in ['iu', 'ui']:
            continue
        else:
            total_mg += value
    return total_mg if total_mg > 0 else np.nan

def extract_ingredient_features_ultimate(row):
    hoatChat_str = str(row.get('hoatChat', '')).lower().strip().replace('/', ';')
    hamLuong_val = row.get('hamLuong')
    hoatChat_list = [hc.strip() for hc in hoatChat_str.split(';') if hc.strip()]
    so_luong_hoat_chat = len(hoatChat_list)
    hoat_chat_chinh = hoatChat_list[0] if so_luong_hoat_chat > 0 else "missing"
    if pd.isnull(hamLuong_val) or str(hamLuong_val).strip() == '':
        return pd.Series([so_luong_hoat_chat, hoat_chat_chinh, 0.0, 0.0, 0.0],
                         index=['so_luong_hoat_chat', 'hoat_chat_chinh', 'hl_chinh_mg', 'tong_hl_phu_mg', 'tong_hl_iu'])
    hamLuong_str = str(hamLuong_val).replace(',', '.')
    dosages = re.findall(r'(\d+\.?\d*)\s*(mcg|µg|mg|g|kg|ml|l|iu|ui|%)', hamLuong_str.lower())
    total_mg = 0.0
    total_iu = 0.0
    converted_dosages_mg = []
    for value_str, unit in dosages:
        value = float(value_str)
        if unit in ULTIMATE_UNIT_CONVERSION_MAP:
            converted_value = value * ULTIMATE_UNIT_CONVERSION_MAP[unit]
            total_mg += converted_value
            converted_dosages_mg.append(converted_value)
        elif unit in ['iu', 'ui']:
            total_iu += value
    hl_chinh_mg = converted_dosages_mg[0] if converted_dosages_mg else 0.0
    tong_hl_phu_mg = sum(converted_dosages_mg[1:]) if len(converted_dosages_mg) > 1 else 0.0
    return pd.Series([so_luong_hoat_chat, hoat_chat_chinh, hl_chinh_mg, tong_hl_phu_mg, total_iu],
                     index=['so_luong_hoat_chat', 'hoat_chat_chinh', 'hl_chinh_mg', 'tong_hl_phu_mg', 'tong_hl_iu'])

def get_packaging_type(text):
    text = str(text).lower()
    for key, value in [
        ('lọ', 'lọ'), ('chai', 'chai'), ('tuýp', 'tuýp'), ('ống', 'ống'),
        ('vỉ', 'vỉ'), ('hộp', 'hộp'), ('gói', 'gói'), ('túi', 'túi'), ('bình', 'bình')
    ]:
        if key in text:
            return value
    return 'khác'

def get_base_unit(text):
    text = str(text).lower()
    for key, value in [
        ('viên nang', 'nang'), ('nang', 'nang'), ('viên', 'viên'),
        ('ml', 'ml'), ('g', 'g'), ('gam', 'g'), ('gói', 'gói'), ('ống', 'ống')
    ]:
        if key in text:
            return value
    return 'khác'

def parse_dosage_value(hamLuong_val):
    if pd.isnull(hamLuong_val):
        return 0.0
    hamLuong_normalized = str(hamLuong_val).replace(',', '.')
    dosages = re.findall(r'(\d+\.?\d*)\s*(mcg|µg|mg|g|kg|ml|l|iu|ui|%)', hamLuong_normalized.lower())
    total_mg = 0.0
    for value_str, unit in dosages:
        value = float(value_str)
        if unit in ULTIMATE_UNIT_CONVERSION_MAP:
            total_mg += value * ULTIMATE_UNIT_CONVERSION_MAP[unit]
    return total_mg

def transform_hybrid_data(hybrid_data, train_cols, categorical_cols, imputer):
    df = pd.DataFrame([hybrid_data])
    df['doanhNghiepSanXuat'] = df['doanhNghiepSanXuat'].replace(['', 'nan', None], np.nan).fillna('missing').astype(str)
    df['nuocSanXuat'] = df['nuocSanXuat'].replace(['', 'nan', None], np.nan).fillna('missing').astype(str)
    df['is_low_price'] = 0
    if 'giaBanBuonDuKien' in df.columns:
        df['is_low_price'] = (pd.to_numeric(df['giaBanBuonDuKien'], errors='coerce') < 500).astype(int)
    pack_features = df['quyCachDongGoi'].apply(extract_quantity)
    df = pd.concat([df, pack_features], axis=1)
    df['tongHamLuong_mg'] = df['hamLuong'].apply(normalize_hamluong_to_mg)
    df['loaiDongGoiChinh'] = df['quyCachDongGoi'].apply(get_packaging_type)
    df['donViCoSo'] = df['quyCachDongGoi'].apply(get_base_unit)
    df['is_dangBaoChe_missing'] = df['dangBaoChe'].isnull().astype(int)
    df['dangBaoChe_final'] = df['dangBaoChe'].apply(classify_dangBaoChe_final)
    df.loc[df['is_dangBaoChe_missing'] == 1, 'dangBaoChe_final'] = 'Không xác định'
    ingredient_features = df.apply(extract_ingredient_features_ultimate, axis=1)
    df = pd.concat([df, ingredient_features], axis=1)
    df['hl_chinh_mg'].fillna(0, inplace=True)
    df['tong_hl_phu_mg'].fillna(0, inplace=True)
    df['tong_hl_iu'].fillna(0, inplace=True)
    df['has_multiple_packs'] = (df['num_pack_options'] > 1).astype(int)
    df['hamluong_so_luong'] = df['tongHamLuong_mg'] * df['mean_so_luong']
    df['gia_per_unit'] = df['tongHamLuong_mg'] / df['mean_so_luong'].replace(0, 1)
    df = df.drop(columns=['tenThuoc', 'hoatChat', 'hamLuong', 'quyCachDongGoi', 'dangBaoChe', 'soLuong', 'donViTinh'], errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]
    missing_cols = [col for col in train_cols if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan
    df = df[train_cols]
    for col in categorical_cols:
        df[col] = df[col].replace(['', 'nan', None], np.nan).fillna('missing').astype(str)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = imputer.transform(df[numerical_cols])
    return df

def parse_user_query(query):
    parsed = {"tenThuoc": "N/A", "hoatChat": np.nan, "hamLuong": np.nan, "soLuong": "N/A", "donViTinh": "N/A"}
    temp_query = query
    match_hc = re.search(r'^(.*?)\s*\((.*?)\)', temp_query)
    if match_hc:
        parsed["hoatChat"] = match_hc.group(1).strip()
        parsed["tenThuoc"] = match_hc.group(2).strip()
        temp_query = temp_query.replace(match_hc.group(0), parsed["tenThuoc"])
    else:
        parsed["tenThuoc"] = temp_query.strip()
    hamluong_pattern = r'(\d+[\.,]?\d*\s*(?:mg|g|mcg|ml|l|iu|ui|kg)(?:\s*/\s*(?:ml|g|viên))?)'
    match_hl = re.search(hamluong_pattern, temp_query, re.IGNORECASE)
    if match_hl:
        parsed["hamLuong"] = match_hl.group(1).strip()
        temp_query = temp_query.replace(match_hl.group(0), '')
    unit_keywords = ['viên nang', 'viên nén', 'nang', 'viên', 'gói', 'ống', 'chai', 'lọ', 'hộp', 'tuýp']
    unit_pattern_sl = '|'.join(unit_keywords)
    match_sl = re.search(r'(\d+)\s*(' + unit_pattern_sl + r')\b', temp_query, re.IGNORECASE)
    if match_sl:
        parsed["soLuong"] = f"{match_sl.group(1)} {match_sl.group(2)}"
        parsed["donViTinh"] = match_sl.group(2).capitalize()
        temp_query = temp_query.replace(match_sl.group(0), '')
    if parsed["tenThuoc"] == "N/A":
        parsed["tenThuoc"] = temp_query.strip()
    parsed["quyCachDongGoi"] = parsed["soLuong"] if pd.notna(parsed["soLuong"]) else parsed["tenThuoc"]
    return parsed

@st.cache_data
def call_gemini_parser(ocr_text):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        prompt = f"""
        Giờ tôi gửi bạn một đoạn OCR từ toa thuốc. Nhiệm vụ của bạn là liệt kê những thuốc trong đó ra theo danh sách tương ứng như sau:

        - Tên thuốc:
        - Hoạt chất:
        - Hàm lượng:
        - Số lượng:
        - Đơn vị tính:

        Một ví dụ như sau:

        1. Tên thuốc: UNASYN
        Hoạt chất: Sultamicillin
        Hàm lượng: 375mg
        Số lượng: 8
        Đơn vị tính: Viên

        2. Tên thuốc: NEXT G CAL
        Hoạt chất: (Chưa xác định)
        Hàm lượng: (Chưa xác định)
        Số lượng: 30
        Đơn vị tính: Viên

        3. Tên thuốc: HEMO Q MOM
        Hoạt chất: (Chưa xác định)
        Hàm lượng: (Chưa xác định)
        Số lượng: 30
        Đơn vị: Viên

        4. Tên thuốc: POVIDINE
        Hoạt chất: (Chưa xác định)
        Hàm lượng: 10% 90ML
        Số lượng: 1
        Đơn vị: Chai

        5. Tên thuốc: Bộ chăm sóc rốn
        Hoạt chất: (Chưa xác định)
        Hàm lượng: (Chưa xác định)
        Số lượng: 1
        Đơn vị: Bộ

        Ở những thuốc có dấu (), ví dụ như Gliclazid (Staclazide 30 MR) thì toàn bộ phần trong dấu () là tên thuốc, phần ở ngoài là hoạt chất, như trường hợp này thì tên thuốc là Staclazide 30 MR, hoạt chất là Gliclazid. không được phép lấy cả Gliclazid (Staclazide 30 MR) cho tên thuốc.

        Còn những trường hợp chỉ có chữ in hoa như NEXT G CAL thì NEXT G CAL là tên thuốc luôn. chữ O ngay trước đơn vị như mg là số 0 bị OCR nhầm.

        Tuy nhiên, nếu thuốc có () và có chứa cả chữ in hoa toàn bộ, như YESOM 40 40mg (Esomeprazol 40mg) thì phần trong ngoặc mới là hoạt chất, còn phần in hoa toàn bộ luôn luôn là tên thuốc. Như trong trường hợp này thì YESOM 40 là tên thuôc, Esomeprazol là hoạt chất.

        Đôi khi kết quả OCR cũng sẽ bị sai chính tả, ví dụ 'Cac lonỉ vitamin' thì bạn sửa lại cho đúng là 'Các loại vitamin'.

        Khi trả về Số lượng thì chỉ cần trả về đúng con số thôi, không trả về kèm đơn vị tính, ví dụ sẽ trả về là Số lượng: 30 chứ không được trả về Số lượng: 30 viên.

        Nhớ lấy hàm lượng một cách thông minh để cố gắng không bị sót hàm lượng để trả về nan nhé.

        Bạn hiểu chưa, và chỉ trả lời bằng cách liệt kê ra tên thuốc và các giá trị tương ứng theo đúng định dạng mà ví dụ tôi đã cung cấp. không trả lời bất cứ gì khác. chỉ trả lời phần cần thiết không giải thích bất cứ gì khác, không mở ngoặc đề chú thích. phải trả lời theo dạng liệt kê xuống dòng như trên. Nhớ là phải liệt kê đủ số lượng đấy nhé.
        OCR đây:
        ---
        {ocr_text}
        ---
        """
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi gọi API của Google AI: {e}")
        return None

def parse_gemini_response(response_text):
    parsed_drugs = []
    blocks = re.split(r'\n*(?=[-\d\.]*\s*Tên thuốc:)', response_text)
    for block in blocks:
        if not block.strip():
            continue
        ten_thuoc = re.search(r"Tên thuốc:\s*(.*)", block)
        hoat_chat = re.search(r"Hoạt chất:\s*(.*)", block)
        ham_luong = re.search(r"Hàm lượng:\s*(.*)", block)
        so_luong = re.search(r"Số lượng:\s*(.*)", block)
        don_vi_tinh = re.search(r"Đơn vị tính:\s*(.*)", block)
        so_luong_value = np.nan
        if so_luong and "(Chưa xác định)" not in so_luong.group(1):
            so_luong_text = so_luong.group(1).strip()
            number_match = re.match(r'(\d+\.?\d*)', so_luong_text)
            if number_match:
                so_luong_value = number_match.group(1)
        drug_dict = {
            "tenThuoc": ten_thuoc.group(1).strip() if ten_thuoc and "(Chưa xác định)" not in ten_thuoc.group(1) else np.nan,
            "hoatChat": hoat_chat.group(1).strip() if hoat_chat and "(Chưa xác định)" not in hoat_chat.group(1) else np.nan,
            "hamLuong": ham_luong.group(1).strip() if ham_luong and "(Chưa xác định)" not in ham_luong.group(1) else np.nan,
            "soLuong": so_luong_value,
            "donViTinh": don_vi_tinh.group(1).strip() if don_vi_tinh and "(Chưa xác định)" not in don_vi_tinh.group(1) else np.nan,
        }
        if pd.notna(drug_dict["tenThuoc"]) or pd.notna(drug_dict["hoatChat"]):
            parsed_drugs.append(drug_dict)
    return parsed_drugs

st.title("Gợi ý Giá thuốc")

if df_full is not None:
    user_query_text = st.text_input("", placeholder="Nhập tên thuốc, ví dụ tên thuốc (hoạt chất) hàm lượng số lượng...", label_visibility="collapsed")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    query_to_process = None
    source = "text"

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(uploaded_file, use_container_width=True)
        with st.spinner("Đang đọc ảnh..."):
            ocr_text = " ".join(ocr_reader.readtext(image_bytes, detail=0))
        query_to_process = ocr_text
        source = "ocr"
        with st.expander("Xem toàn bộ văn bản nhận dạng được"):
            st.text_area("", ocr_text, height=150)
    elif user_query_text:
        query_to_process = user_query_text

    if query_to_process:
        st.markdown("---")
        with st.spinner("AI đang phân tích đơn thuốc..."):
            if source == "ocr":
                structured_response = call_gemini_parser(query_to_process)
                if not structured_response:
                    st.error("AI không thể phân tích văn bản từ ảnh này.")
                    st.stop()
                lines_to_process = parse_gemini_response(structured_response)
            else:
                lines_to_process = [parse_user_query(query_to_process)]

        if not lines_to_process:
            st.warning("Không nhận dạng được đơn thuốc nào.")
            st.stop()

        valid_drug_count = 0
        total_gia_kk = 0.0
        total_gia_tt = 0.0
        price_details = []

        for i, parsed_info in enumerate(lines_to_process):
            ten_thuoc = parsed_info.get('tenThuoc', '').strip() if pd.notna(parsed_info.get('tenThuoc')) else ''
            hoat_chat = parsed_info.get('hoatChat', '').strip() if pd.notna(parsed_info.get('hoatChat')) else ''
            if not ten_thuoc and not hoat_chat:
                continue

            valid_drug_count += 1
            if len(lines_to_process) > 1:
                st.markdown(f"--- \n ### 💊 Kết quả cho đơn thuốc {valid_drug_count}")

            with st.spinner(f"Đang xử lý: '{(ten_thuoc or hoat_chat)[:50]}...'"):
                st.markdown(f"**Tên thuốc:** {parsed_info.get('tenThuoc') or '(Chưa xác định)'}")
                st.markdown(f"**Hoạt chất:** {parsed_info.get('hoatChat') or '(Chưa xác định)'}")
                st.markdown(f"**Hàm lượng:** {parsed_info.get('hamLuong') or '(Chưa xác định)'}")
                st.markdown(f"**Số lượng:** {parsed_info.get('soLuong') or 'N/A'}")
                st.markdown(f"**Đơn vị tính:** {parsed_info.get('donViTinh') or 'N/A'}")
                st.markdown("---")

                quantity = float(parsed_info.get('soLuong', '1')) if parsed_info.get('soLuong') and parsed_info.get('soLuong') != 'N/A' else 1.0

                # Step 1: Compare tenThuoc only (>= 95%)
                choices = df_full['tenThuoc'].dropna().tolist()
                best_match, score, _ = process.extractOne(ten_thuoc, choices, scorer=fuzz.ratio) if ten_thuoc else (None, 0, None)

                if best_match and score >= 95:
                    # Direct match based on tenThuoc
                    drug_info_row = df_full[df_full['tenThuoc'] == best_match].iloc[0]
                    st.markdown(f"**Phương thức:** `Levenshtein Distance (Thuốc: {best_match}, độ tương đồng: {score:.0f}%)`")
                    gia_kk = drug_info_row['giaBanBuonDuKien'] * quantity
                    gia_tt = drug_info_row.get('giaThanh', np.nan) * quantity if pd.notna(drug_info_row.get('giaThanh')) else 0.0
                    st.metric("Giá Kê Khai", f"{gia_kk:,.0f} VND" if pd.notna(gia_kk) else "Không có dữ liệu")
                    st.metric("Giá Thị Trường", f"{gia_tt:,.0f} VND" if gia_tt > 0 else "Không có dữ liệu")
                    price_details.append({
                        'tenThuoc': parsed_info.get('tenThuoc'),
                        'gia_kk': gia_kk if pd.notna(gia_kk) else 0.0,
                        'gia_tt': gia_tt,
                        'quantity': quantity,
                        'donViTinh': parsed_info.get('donViTinh') or 'N/A'
                    })
                    total_gia_kk += gia_kk if pd.notna(gia_kk) else 0.0
                    total_gia_tt += gia_tt
                else:
                    # Step 2: Compare tenThuoc + hoatChat (>= 90%)
                    search_key = f"{ten_thuoc} {hoat_chat}".strip()
                    choices_combined = (df_full['tenThuoc'].fillna('') + ' ' + df_full['hoatChat'].fillna('')).str.strip().tolist()
                    best_match, score, idx = process.extractOne(search_key, choices_combined, scorer=fuzz.ratio) if search_key else (None, 0, None)

                    if best_match and score >= 90:
                        drug_info_row = df_full.iloc[idx]
                        # Check if dosage differs for extrapolation
                        can_extrapolate = False
                        if pd.notna(parsed_info.get('hamLuong')) and pd.notna(drug_info_row.get('hamLuong')):
                            user_dosage_mg = parse_dosage_value(parsed_info.get('hamLuong'))
                            db_dosage_mg = parse_dosage_value(drug_info_row.get('hamLuong'))
                            if user_dosage_mg > 0 and db_dosage_mg > 0 and user_dosage_mg != db_dosage_mg:
                                can_extrapolate = True

                        if can_extrapolate:
                            ratio = user_dosage_mg / db_dosage_mg
                            st.markdown(f"**Phương thức:** `Extrapolation by Dosage (Tỷ lệ: {ratio:.2f}x)`")
                            st.caption(f"Dựa trên giá của *{drug_info_row['tenThuoc']}* (độ tương đồng: {score:.0f}%)")
                            gia_kk_base = drug_info_row['giaBanBuonDuKien']
                            gia_tt_base = drug_info_row.get('giaThanh', np.nan)
                            gia_kk_extrapolated = gia_kk_base * ratio * quantity
                            gia_tt_extrapolated = gia_tt_base * ratio * quantity if pd.notna(gia_tt_base) else 0.0
                            st.metric("Giá Kê Khai (Ước tính)", f"{gia_kk_extrapolated:,.0f} VND" if pd.notna(gia_kk_base) else "Không có dữ liệu")
                            st.metric("Giá Thị Trường (Ước tính)", f"{gia_tt_extrapolated:,.0f} VND" if gia_tt_extrapolated > 0 else "Không có dữ liệu")
                            price_details.append({
                                'tenThuoc': parsed_info.get('tenThuoc'),
                                'gia_kk': gia_kk_extrapolated if pd.notna(gia_kk_base) else 0.0,
                                'gia_tt': gia_tt_extrapolated,
                                'quantity': quantity,
                                'donViTinh': parsed_info.get('donViTinh') or 'N/A'
                            })
                            total_gia_kk += gia_kk_extrapolated if pd.notna(gia_kk_base) else 0.0
                            total_gia_tt += gia_tt_extrapolated
                        else:
                            # Direct match based on tenThuoc + hoatChat
                            st.markdown(f"**Phương thức:** `Levenshtein Distance (Thuốc: {drug_info_row['tenThuoc']}, độ tương đồng: {score:.0f}%)`")
                            gia_kk = drug_info_row['giaBanBuonDuKien'] * quantity
                            gia_tt = drug_info_row.get('giaThanh', np.nan) * quantity if pd.notna(drug_info_row.get('giaThanh')) else 0.0
                            st.metric("Giá Kê Khai", f"{gia_kk:,.0f} VND" if pd.notna(gia_kk) else "Không có dữ liệu")
                            st.metric("Giá Thị Trường", f"{gia_tt:,.0f} VND" if gia_tt > 0 else "Không có dữ liệu")
                            price_details.append({
                                'tenThuoc': parsed_info.get('tenThuoc'),
                                'gia_kk': gia_kk if pd.notna(gia_kk) else 0.0,
                                'gia_tt': gia_tt,
                                'quantity': quantity,
                                'donViTinh': parsed_info.get('donViTinh') or 'N/A'
                            })
                            total_gia_kk += gia_kk if pd.notna(gia_kk) else 0.0
                            total_gia_tt += gia_tt
                    else:
                        # Use CatBoost Regressor
                        st.markdown(f"**Phương thức:** `CatBoost Regressor`")
                        st.caption(f"Sử dụng thông tin bổ sung (nhà SX, nước SX, Dạng bào chế...) từ thuốc tương tự nhất: *{best_match or 'N/A'}*")
                        hybrid_data = {
                            'tenThuoc': parsed_info.get('tenThuoc', 'missing'),
                            'hoatChat': parsed_info.get('hoatChat', 'missing') or (drug_info_row.get('hoatChat', 'missing') if best_match and 'drug_info_row' in locals() else 'missing'),
                            'hamLuong': parsed_info.get('hamLuong', np.nan),
                            'quyCachDongGoi': parsed_info.get('quyCachDongGoi', 'missing'),
                            'doanhNghiepSanXuat': drug_info_row.get('doanhNghiepSanXuat', 'missing') if best_match and 'drug_info_row' in locals() else 'missing',
                            'nuocSanXuat': drug_info_row.get('nuocSanXuat', 'missing') if best_match and 'drug_info_row' in locals() else 'missing',
                            'dangBaoChe': drug_info_row.get('dangBaoChe', 'missing') if best_match and 'drug_info_row' in locals() else 'missing'
                        }
                        try:
                            transformed_data = transform_hybrid_data(hybrid_data, train_cols, categorical_cols, imputer)
                            y_pred_giaThanh_log = model_giaThanh.predict(transformed_data)
                            y_pred_giaBanBuon_log = model_giaBanBuon.predict(transformed_data)
                            gia_tt_pred = scaler_giaThanh.inverse_transform(y_pred_giaThanh_log.reshape(-1, 1)).flatten()[0] * quantity
                            gia_kk_pred = scaler_giaBanBuon.inverse_transform(y_pred_giaBanBuon_log.reshape(-1, 1)).flatten()[0] * quantity
                            st.metric("Giá Kê Khai (Dự đoán)", f"{gia_kk_pred:,.0f} VND")
                            st.metric("Giá Thị Trường (Dự đoán)", f"{gia_tt_pred:,.0f} VND")
                            price_details.append({
                                'tenThuoc': parsed_info.get('tenThuoc'),
                                'gia_kk': gia_kk_pred,
                                'gia_tt': gia_tt_pred,
                                'quantity': quantity,
                                'donViTinh': parsed_info.get('donViTinh') or 'N/A'
                            })
                            total_gia_kk += gia_kk_pred
                            total_gia_tt += gia_tt_pred
                        except Exception as e:
                            st.error(f"Lỗi khi dự đoán: {e}")
                            price_details.append({
                                'tenThuoc': parsed_info.get('tenThuoc'),
                                'gia_kk': 0.0,
                                'gia_tt': 0.0,
                                'quantity': quantity,
                                'donViTinh': parsed_info.get('donViTinh') or 'N/A'
                            })

        if valid_drug_count > 0:
            st.markdown("--- \n ### Kết quả cho toa thuốc (dự đoán)")
            st.metric("Tổng Giá Kê Khai", f"{total_gia_kk:,.0f} VND")
            st.metric("Tổng Giá Thị Trường", f"{total_gia_tt:,.0f} VND")
            st.markdown("#### Chi tiết giá từng thuốc")
            for detail in price_details:
                st.write(f"- **{detail['tenThuoc']}** ({detail['quantity']:.0f} {detail['donViTinh']}): "
                         f"Giá Kê Khai: {detail['gia_kk']:,.0f} VND, "
                         f"Giá Thị Trường: {detail['gia_tt']:,.0f} VND")
