# ====================================================================================================

from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize
import numpy as np
import unicodedata
import time
import json
import csv
import re
from LLM import Process_LLM

def normalize_text(text):
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower().strip()
    return text
def extract_keywords(text):
    def is_number(text):
        return bool(re.fullmatch(r'[\d,. ]+', text))
    specialchars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '..', '...', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    specialwords = ["giấy tờ", "thủ tục", "giấy", "gì", "cần", "nào", "sắp", "đang", "sẽ", "của", "bị", "hoặc", "với", "và", "thì", "muốn", "gì", "mình", "tôi", "phải", "làm sao", "để", "cho", "làm", "như", "đối với", "từ", "theo", "là", "được", "ở", "đã", "về", "có", "các", "tại", "đến", "vào", "do", "vì", "bởi vì", "thuộc"]
    words = word_tokenize(text)
    words = [w.lower().strip() for w in words]
    words = list(set(words))
    words = [w for w in words if not is_number(w)]
    words = [w for w in words if w not in specialchars]
    words = [w for w in words if w not in specialwords]
    words_normalized = [normalize_text(w) for w in words if " " in w]
    return list(set(words + words_normalized))

# ====================================================================================================

# ---------- Load models and embeddings ----------
model_e5 = SentenceTransformer("onelevelstudio/M-E5-BASE")
model_mpnet = SentenceTransformer("onelevelstudio/M-MPNET-BASE")
embs_e5 = np.load("url/embs_e5")
embs_mpnet = np.load("url/embs_mpnet")

# ---------- Load thutucs from cache ----------
with open('url/cache', mode='r', newline='', encoding='utf-8') as f:
    thutucs = list(csv.DictReader(f))
    for i in range(len(thutucs)):
        thutucs[i]["keywords"] = extract_keywords(thutucs[i]["Tên thủ tục"])

# ====================================================================================================

def retrieve_idx_semantic(text, pre_embs, emb_model, top=5):
    q_emb = emb_model.encode(text)
    similarities = emb_model.similarity(q_emb, pre_embs)[0]
    top_5_idx = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top]
    return top_5_idx

def retrieve_idx_exactmatch(text, thutucs, top=5):
    thutuc_scores = []
    for e in thutucs:
        score = 0
        # ----------
        if normalize_text(text) in normalize_text(e["Tên thủ tục"]):
            score += 1
        if text.lower().strip() in e["Tên thủ tục"].lower():
            score += 1
        # ----------
        thutuc_scores.append(score)
    thutuc_scores_idx = sorted(range(len(thutuc_scores)), key=lambda i: thutuc_scores[i], reverse=True)
    scores = [(idx, thutuc_scores[idx]) for idx in thutuc_scores_idx]
    scores = [e for e in scores if e[1] != 0]
    return [e[0] for e in scores][:min(len(scores), top)]

def retrieve_idx_keywordmatch(text, thutucs, top=5):
    q_keywords = extract_keywords(text)
    # print(q_keywords)
    thutuc_scores = []
    for e in thutucs:
        score = 0
        # ----------
        for k in q_keywords:
            if k in e["keywords"]:
                score += 1
        # ----------
        thutuc_scores.append(score)
    thutuc_scores_idx = sorted(range(len(thutuc_scores)), key=lambda i: thutuc_scores[i], reverse=True)
    scores = [(idx, thutuc_scores[idx]) for idx in thutuc_scores_idx]
    scores = [e for e in scores if e[1] != 0]
    return [e[0] for e in scores][:min(len(scores), top)]

# ====================================================================================================

def fn_ohyeahhhhhhhhhhhhhhhhhhhhhhhhhh(message, history):

    print("="*100)
    print(f"> {message}")

    input_text = message.strip()
    if input_text == "":
        history.append({"role": "assistant", "content": "✨"}); yield "", history; return

    # ----------------------------------------------------------------------------------------------------
    res_exactmatch_idx = retrieve_idx_exactmatch(text=input_text, thutucs=thutucs, top=3)
    res_keywordmatch_idx = retrieve_idx_keywordmatch(text=input_text, thutucs=thutucs, top=3)
    res_semantic_idx_1 = retrieve_idx_semantic(text=input_text, pre_embs=embs_e5, emb_model=model_e5, top=3)
    res_semantic_idx_2 = retrieve_idx_semantic(text=input_text, pre_embs=embs_mpnet, emb_model=model_mpnet, top=3)

    # ----------------------------------------------------------------------------------------------------
    # Case 1: There is exact match -> only keep the exact match
    if len(res_exactmatch_idx) > 0:
        res_all_idx = res_exactmatch_idx
    # Case 2: There is no exact match -> keyword + sementic
    else:
        res_all_idx = res_keywordmatch_idx + res_semantic_idx_1 + res_semantic_idx_2

    # ----------------------------------------------------------------------------------------------------
    p_danhsachthutuc = [{"Mã chuẩn": thutucs[idx]["Mã chuẩn"], "Tên thủ tục": thutucs[idx]["Tên thủ tục"]} for idx in res_all_idx]
    p_json_schema = """\
    {
        "type": "object",
        "properties": {
            "Mã chuẩn": {"type": "string", "description": "Mã chuẩn của thủ tục liên quan nhất"},
            "Tên thủ tục": {"type": "string", "description": "Tên của thủ tục liên quan nhất"}
        }
    }"""
    prompt_1 = f"""\
    Bạn sẽ được cung cấp: (1) Câu hỏi của người dùng, (2) Danh sách thủ tục hiện có, và (3) Schema cấu trúc của kết quả.
    Nhiệm vụ của bạn là: (4) Trích xuất duy nhất 1 thủ tục liên quan nhất đến câu hỏi của người dùng.

    ### (1) Câu hỏi của người dùng:
    "{input_text}"

    ### (2) Danh sách thủ tục hiện có:
    {p_danhsachthutuc}

    ### (3) Schema cấu trúc của kết quả:
    {p_json_schema}

    ### (4) Nhiệm vụ:
    Từ câu hỏi của người dùng, tìm ra duy nhất 1 thủ tục liên quan nhất đến câu hỏi của người dùng, tuân thủ schema một cách chính xác.
    Định dạng kết quả: Không giải thích, không bình luận, không văn bản thừa. Chỉ trả về kết quả JSON hợp lệ. Bắt đầu bằng "{{", kết thúc bằng "}}".
    """
    # print(prompt_1)

    # ----------------------------------------------------------------------------------------------------

    for i in range(5):
        llm_text_1 = Process_LLM(prompt=prompt_1, vendor="ollama")
        regex_match = re.search(r'\{.*\}', llm_text_1, re.S)
        if regex_match:
            try:
                llm_object_1 = json.loads(regex_match.group())
                # --------------------------------------------------
                idx = next((i for i, d in enumerate(thutucs) if d["Mã chuẩn"] == llm_object_1["Mã chuẩn"].strip()), -1)
                if idx != -1:
                    print(f"✅ {thutucs[idx]['Tên thủ tục']} ({thutucs[idx]['thutuc_Link']})")
                    # --------------------------------------------------
                    bot_response = f"Thủ tục: {thutucs[idx]['Tên thủ tục']}"
                    history.append({"role": "assistant", "content": ""})
                    for chr in bot_response:
                        time.sleep(0.01)
                        history[-1]["content"] += chr
                        yield "", history
                    bot_response = f"Link: {thutucs[idx]['thutuc_Link']}"
                    history.append({"role": "assistant", "content": ""})
                    for chr in bot_response:
                        time.sleep(0.01)
                        history[-1]["content"] += chr
                        yield "", history
                    return
                    # --------------------------------------------------
                    break
                # --------------------------------------------------
            except:
                pass

    # ----------------------------------------------------------------------------------------------------
    history.append({"role": "assistant", "content": "⚠️"}); yield "", history; return






# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

import gradio as gr

# ====================================================================================================

def fn_1(message, history):
    history.append({"role": "user", "content": message})
    return history

# ====================================================================================================

with gr.Blocks(title="Chatbot Dịch vụ công", analytics_enabled=False) as demo:
    with gr.Row(elem_id="cmp_row"):
        with gr.Column(elem_id="cmp_col_left"):
            gr.Markdown()
        with gr.Column(elem_id="cmp_col_mid"):
            cmp_history = gr.Chatbot(elem_id="cmp_history", placeholder="## Xin chào!", type="messages", group_consecutive_messages=False, container=False)
            cmp_message = gr.Textbox(elem_id="cmp_message", placeholder="Nhập câu hỏi ở đây", submit_btn=True, show_label=False)
        with gr.Column(elem_id="cmp_col_right"):
            gr.Markdown()

    cmp_message.submit(fn=fn_1, inputs=[cmp_message, cmp_history], outputs=[cmp_history]
                ).then(fn=fn_ohyeahhhhhhhhhhhhhhhhhhhhhhhhhh, inputs=[cmp_message, cmp_history], outputs=[cmp_message, cmp_history])

# ====================================================================================================

if __name__ == "__main__":
    demo.launch()