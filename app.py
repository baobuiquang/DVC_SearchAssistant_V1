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
        history.append({"role": "assistant", "content": "Mình có thể giúp gì được cho bạn?"}); yield "", history; return

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
                    # -------------------------------------------------- Part 1
                    eee = thutucs[idx]
                    CHARACTERS_LIMIT = 200
                    XEMCHITIET_TEXT = f"... [(xem chi tiết)]({thutucs[idx]['thutuc_Link']})"
                    bot_response = f"""\
## Thủ tục: {eee['Tên thủ tục'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['Tên thủ tục']) > CHARACTERS_LIMIT else ""}
\n### Trình tự thực hiện:
{eee['thutuc_Trình tự thực hiện'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Trình tự thực hiện']) > CHARACTERS_LIMIT else ""}
\n### Cách thức thực hiện:
{eee['thutuc_Cách thức thực hiện'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Cách thức thực hiện']) > CHARACTERS_LIMIT else ""}
\n### Thành phần hồ sơ:
{eee['thutuc_Thành phần hồ sơ'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Thành phần hồ sơ']) > CHARACTERS_LIMIT else ""}
\n### Thời gian giải quyết:
{eee['thutuc_Thời gian giải quyết'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Thời gian giải quyết']) > CHARACTERS_LIMIT else ""}
\n### Đối tượng thực hiện:
{eee['thutuc_Đối tượng thực hiện'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Đối tượng thực hiện']) > CHARACTERS_LIMIT else ""}
\n### Cơ quan thực hiện:
{eee['thutuc_Cơ quan thực hiện'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Cơ quan thực hiện']) > CHARACTERS_LIMIT else ""}
\n### Kết quả:
{eee['thutuc_Kết quả'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Kết quả']) > CHARACTERS_LIMIT else ""}
\n### Phí, lệ phí:
{eee['thutuc_Phí, lệ phí'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Phí, lệ phí']) > CHARACTERS_LIMIT else ""}
\n### Tên mẫu đơn, tờ khai:
{eee['thutuc_Tên mẫu đơn, tờ khai'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Tên mẫu đơn, tờ khai']) > CHARACTERS_LIMIT else ""}
\n### Yêu cầu, điều kiện:
{eee['thutuc_Yêu cầu, điều kiện'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Yêu cầu, điều kiện']) > CHARACTERS_LIMIT else ""}
\n### Căn cứ pháp lý:
{eee['thutuc_Căn cứ pháp lý'][:CHARACTERS_LIMIT]}{XEMCHITIET_TEXT if len(eee['thutuc_Căn cứ pháp lý']) > CHARACTERS_LIMIT else ""}
                    """
                    history.append({"role": "assistant", "content": ""})
                    for chr in re.split(r'(\s)', bot_response):
                        time.sleep(0.001)
                        history[-1]["content"] += chr
                        yield "", history
                    # -------------------------------------------------- Part 2
                    bot_response = f"Xem đầy đủ văn bản thủ tục tại: {thutucs[idx]['thutuc_Link']}"
                    history.append({"role": "assistant", "content": ""})
                    for chr in re.split(r'(\s)', bot_response):
                        time.sleep(0.001)
                        history[-1]["content"] += chr
                        yield "", history
                    # -------------------------------------------------- Part 3
                    history.append({"role": "assistant", "content": gr.HTML(f"<a href='{thutucs[idx]['thutuc_Link']}' target='_blank' id='button-open-link'><div>Mở văn bản thủ tục ➝</div></a>")})
                    yield "", history
                    # --------------------------------------------------
                    return
                    break
                # --------------------------------------------------
            except:
                pass

    # ----------------------------------------------------------------------------------------------------
    history.append({"role": "assistant", "content": "Hiện tại mình chưa đủ thông tin để trả lời câu hỏi này."})
    history.append({"role": "assistant", "content": "Mình có thể giúp gì được cho bạn?"})
    yield "", history; return






# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

import gradio as gr

# ====================================================================================================

theme = gr.themes.Base(
    primary_hue="neutral",
    secondary_hue="neutral",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont('Inter')], 
    font_mono=[gr.themes.GoogleFont('Ubuntu Mono')]
)
head = """
<link rel="icon" href="https://raw.githubusercontent.com/baobuiquang/VNPT_DVC_SemSearchAPI/refs/heads/main/logo.png">
"""
css = """
main {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
}
footer { display: none !important; }
* { -ms-overflow-style: none; scrollbar-width: none; }
*::-webkit-scrollbar { display: none; }

#cmp_row {
    gap: 0;
}

#cmp_col_left, #cmp_col_right {
    min-width: min(100px, 100%) !important;
}
@media screen and (max-width: 1000px) {
    #cmp_col_left {
        display: none;
    }
}

#cmp_col_mid {
    min-width: min(700px, 100%) !important;
    height: 100svh;
    gap: 0;
    padding: 0 16px;
}

/* -------------------------------------------------- */

#cmp_history {
    border: none !important;
    background: transparent;
    flex-grow: 1;
}
#cmp_history .bubble-wrap {
    background: transparent;
}
#cmp_history .bubble-wrap .message-wrap {
    margin: 70svh 0 20px 0;
}
#cmp_history .bubble-wrap .message-wrap .message-row {
    margin: 12px 0;
}
#cmp_history .bubble-wrap .message-wrap .message {
    border: solid 1px #80808020 !important;
    padding: 12px 18px;
}
#cmp_history .bubble-wrap .message-wrap .message.user {
    background: #ce7a5830;
    border-radius: 12px 12px 2px 12px;
}
#cmp_history .bubble-wrap .message-wrap .message.bot {
    background: #80808010;
    border-radius: 12px 12px 12px 12px;
}
#cmp_history .bubble-wrap .message-wrap .message.pending {
    background: transparent;
}

#cmp_history .examples {
    grid-template-columns: repeat(auto-fit, minmax(98px, 1fr));
    margin: 0;
    padding: 0 0 16px 0;
}
#cmp_history .examples .example {
    padding: 10px 0px;
    border-radius: 8px;
    border: solid 1px #80808020;
    background: #80808010;
    transform: none;
    transition: all ease 0.3s;
}
#cmp_history .examples .example:hover {
    background: #80808020;
}
#cmp_history .examples .example * {
    text-align: center;
    width: 100%;
}

#cmp_history .icon-button-wrapper {
    position: fixed;
    top: 8px;
    right: 8px;
    border: none;
    background: transparent;
}
#cmp_history .icon-button-wrapper button[title="Clear"] {
    background: #80808010;
    border: solid 1px #80808020;
    border-radius: 6px;
    transition: all ease 0.3s;
}
#cmp_history .icon-button-wrapper button[title="Clear"]:hover {
    background: #80808060;
}
#cmp_history .icon-button-wrapper button[title="Clear"] div {
    display: none;
}
#cmp_history .icon-button-wrapper button[title="Clear"]::after {
    content: "✎ Cửa sổ mới";
    padding: 8px 14px;
}
#cmp_history .bubble-wrap .message-wrap .message-row.user-row .avatar-container {
    display: none;
}

#cmp_history .placeholder {
    opacity: 0.9;
}
#cmp_history .placeholder img {
    height: 32px;
}

/* -------------------------------------------------- */

#cmp_message {
    background:#80808020;
    border-radius: 8px;
    border: solid 2px #80808060 !important;
    transition: all ease 0.3s;
}
#cmp_message:hover, #cmp_message:focus-within {
    border: solid 2px #80808090 !important;
}
#cmp_message .full-container {
    padding: 0;
}
#cmp_message textarea {
    background: transparent;
    padding: 20px 4px 23px 22px;
    font-size: 16px;
}
#cmp_message button.upload-button,
#cmp_message button.submit-button {
    background: #80808020;
    margin: 0 8px 13px 8px;
    height: 40px;
    width: 40px;
    border-radius: 6px;
    transition: all ease 0.3s;
}
#cmp_message button.upload-button:hover,
#cmp_message button.submit-button:hover {
    background: #80808060;
}
#cmp_message .generating {
    border: none;
}

/* -------------------------------------------------- */

#cmp_footer * {
    margin: 10px 0 12px 0;
    text-align: center;
    font-size: 14px;
    color: #808080;
}

.progress-text {
    color: transparent;
    background: transparent !important;
}

#cmp_top_bar {
    background: transparent;
    height: 60px;
    width: 100vw;
    position: fixed;
    top: 0;
    left: 0;
    border-radius: 0;
}
@media screen and (max-width: 1000px) {
    #cmp_top_bar {
        background: #222222;
    }
}
#cmp_top_bar a, #cmp_top_bar a img {
    height: 40px;
    width: 40px;
    display: block;
}

/* -------------------------------------------------- */

a#button-open-link {
    text-decoration: none;
}
a#button-open-link div {
    background: #ce7a5830;
    text-align: center;
    padding: 12px 0;
    border-radius: 8px;
    transition: all ease 0.3s;
    border: solid 1px #80808030;
    color: #000000aa;
    font-weight: 600;
}

a#button-open-link div:hover {
    background: #ce7a5860;
    border: solid 1px #80808060;
}
"""

# ====================================================================================================

def fn_1(message, history):
    history.append({"role": "user", "content": message})
    return history

# ====================================================================================================

with gr.Blocks(title="Chatbot hỗ trợ tìm kiếm thủ tục", theme=theme, head=head, css=css, analytics_enabled=False) as demo:
    with gr.Row(elem_id="cmp_row"):
        with gr.Column(elem_id="cmp_col_left"):
            gr.Markdown()
        with gr.Column(elem_id="cmp_col_mid"):
            cmp_history = gr.Chatbot(elem_id="cmp_history", placeholder="![image](https://raw.githubusercontent.com/baobuiquang/VNPT_DVC_SemSearchAPI/refs/heads/main/logo.png)\n## Xin chào!\nMình là chatbot hỗ trợ tìm kiếm thủ tục dịch vụ công.", type="messages", group_consecutive_messages=False, container=False,
                avatar_images=("https://raw.githubusercontent.com/baobuiquang/VNPT_DVC_SemSearchAPI/refs/heads/main/logo.png", "https://raw.githubusercontent.com/baobuiquang/VNPT_DVC_SemSearchAPI/refs/heads/main/logo.png"))
            cmp_message = gr.Textbox(elem_id="cmp_message", placeholder="Nhập câu hỏi ở đây", submit_btn=True, container=False)
            cmp_footer = gr.Markdown("AI có thể nhầm lẫn và sai sót. Hãy kiểm tra lại thông tin trên [cổng dịch vụ công](https://dichvucong.lamdong.gov.vn).", elem_id="cmp_footer", container=False)
        with gr.Column(elem_id="cmp_col_right"):
            gr.Markdown()

    cmp_message.submit(fn=fn_1, inputs=[cmp_message, cmp_history], outputs=[cmp_history]
                ).then(fn=fn_ohyeahhhhhhhhhhhhhhhhhhhhhhhhhh, inputs=[cmp_message, cmp_history], outputs=[cmp_message, cmp_history])

# ====================================================================================================

if __name__ == "__main__":
    demo.launch(share=False)