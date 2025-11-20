# app2.py — PBI 风险预测（与旧版交互一致；对接新模型 14 特征 & 二元口径）
# ------------------------------------------------------------------
# 依赖（requirements.txt 建议）:
# streamlit==1.38.0
# pandas==2.2.3
# numpy==1.26.4
# scikit-learn==1.3.2
# joblib==1.3.2
# skops==0.13.0

import json
from pathlib import Path

import numpy as np
import pandas as pd
import json, io, os, zipfile
import matplotlib.pyplot as plt
import streamlit as st            # ← 新增：修复 NameError
import joblib                     # ← 新增：回退用 joblib.load 时需要
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


BINARY_FEATURES = {"antenatal_mgso4","surgery","NRDS","AOP","sex_male"}


# =========================
# 1) 多语言文本
# =========================
TEXT = {
    "lang_label": {"zh": "界面语言", "en": "Interface language"},
    "title": {"zh": "PBI 风险预测 · 原型（研究性）", "en": "PBI Risk Prediction · Prototype"},
    "caption": {
        "zh": "说明：本工具仅用于科研/质控，不构成临床诊断依据。",
        "en": "Note: Research/QC only. Not for clinical diagnosis.",
    },
    "band_title": {"zh": "阈值与分级（四档）", "en": "Thresholds & four-level risk bands"},
    "view_imgs": {"zh": "查看单独图片", "en": "View individual images"},
    "download_zip_all": {
    "zh": "下载导出（包含分级、waterfall/force、贡献CSV）",
    "en": "Download (bands + waterfall/force + contributions CSV)"
    },
    "meta": {"zh": "版本/来源信息", "en": "Release / Meta"},
    "input_section": {"zh": "输入特征", "en": "Input features"},
    "thresh_mode": {"zh": "阈值策略", "en": "Threshold mode"},
    "predict": {"zh": "预测", "en": "Predict"},
    "youd": {"zh": "Youden", "en": "Youden"},
    "hsens": {"zh": "高敏感", "en": "High sensitivity"},
    "probability": {"zh": "预测概率", "en": "Predicted probability"},
    "two_rules": {"zh": "两套阈值判定：", "en": "Decisions under two thresholds:"},
    "pos": {"zh": "阳性", "en": "Positive"},
    "neg": {"zh": "阴性", "en": "Negative"},
    "current_mode": {"zh": "当前策略", "en": "Current mode"},
    "result": {"zh": "结果", "en": "Result"},
    "download_html": {"zh": "下载 HTML 报告", "en": "Download HTML report"},
    "batch_title": {"zh": "批量 CSV 预测", "en": "Batch CSV prediction"},
    "batch_caption": {
        "zh": "上传包含同名列的 CSV（列顺序将自动对齐；缺失值由模型内的 SimpleImputer 处理）",
        "en": "Upload CSV with matching columns (reordered automatically; missing values imputed by model).",
    },
    "upload_csv": {"zh": "选择 CSV 文件", "en": "Select CSV file"},
    "download_csv": {"zh": "下载结果 CSV", "en": "Download result CSV"},
    "range_help": {"zh": "范围", "en": "Range"},
    "step_help": {"zh": "步进", "en": "Step"},
    "binary_help": {"zh": "二元变量：1=发生/是，0=未发生/否；其中 sex_male：1=男性，0=女性", "en": "Binary: 1=present/yes, 0=absent/no; sex_male: 1=male, 0=female"},
    "risk_low": {"zh": "低", "en": "low"},
    "risk_mid": {"zh": "中", "en": "medium"},
    "risk_high": {"zh": "高", "en": "high"},
    "risk_sentence": {
        "zh": "该患儿脑损伤的风险较{level}（概率 {p:.1%}）。",
        "en": "The infant's risk of brain injury is {level} (probability {p:.1%}).",
    },
    "skops_warn": {
        "zh": "安全提示：已根据 skops>=0.10 的要求使用 get_untrusted_types(file=...) 获取受信类型列表进行加载。",
        "en": "Security note: Using get_untrusted_types(file=...) per skops>=0.10 to build trusted list.",
    },
    "no_model": {
        "zh": "未找到模型：请将 final_pipeline.skops 放在仓库根目录（或提供 final_pipeline.joblib 作为回退）。",
        "en": "Model not found: place final_pipeline.skops at repo root (or provide final_pipeline.joblib as fallback).",
    },
    "done_n": {"zh": "预测完成：{n} 条", "en": "Done: {n} rows"},
}

# =============== 2) 语言选择 ===============
st.set_page_config(page_title="PBI Risk Prediction · Prototype", layout="centered")

lang_choice = st.sidebar.radio(
    f'{TEXT["lang_label"]["zh"]} / {TEXT["lang_label"]["en"]}',
    ["中文", "English"],
    index=0,
    horizontal=True,
)
LANG = "zh" if lang_choice == "中文" else "en"
st.title(TEXT["title"][LANG])
st.caption(TEXT["caption"][LANG])

# =============== 3) 常量：新模型 14 特征与二元口径 ===============
# 顺序与 step5 的 final_features.json 一致
DEFAULT_FEATURE_ORDER = [
    "PLT","LAC","GA_weeks_decimal","inv_vent_days","ALB","birth_weight_g",
    "WBC","Hb","BE","antenatal_mgso4","surgery","NRDS","AOP","sex_male"
]

# —— 风险分级：用两条阈值派生四档（极低/低/中/高）——
def risk_band_from_prob(p: float, hs_thr: float, y_thr: float) -> str:
    """基于高敏阈值与 Youden 阈值给出四分级。
    规则：p >= y_thr → 高；hs_thr ≤ p < y_thr → 中；0.5*hs_thr ≤ p < hs_thr → 低；否则 极低
    """
    if p >= y_thr:
        return "高"
    elif p >= hs_thr:
        return "中"
    elif p >= 0.5 * hs_thr:
        return "低"
    else:
        return "极低"

# —— 计算 LightGBM 贡献：使用 pipeline 的“前处理 + LGBMClassifier”——
def compute_lgbm_contrib(pipe, x_df):
    """
    返回 base(log-odds) 与逐特征贡献（list[dict]）。
    兼容：
      - 任意多层 Pipeline
      - CalibratedClassifierCV（不同版本字段名：estimator / classifier / base_estimator）
    只使用 getattr(..., None) 安全取属性，避免 AttributeError。
    """
    def _find_inner_estimator(m):
        # Pipeline → 递归到最后一步
        if isinstance(m, Pipeline):
            return _find_inner_estimator(m.steps[-1][1])

        # Calibrated → 遍历已拟合的 calibrated_classifiers_ 列表
        if isinstance(m, CalibratedClassifierCV):
            ccs = getattr(m, "calibrated_classifiers_", None)
            if ccs:
                # 逐个尝试拿内部估计器
                for cc in ccs:
                    for key in ("estimator", "classifier", "base_estimator"):
                        est = getattr(cc, key, None)
                        if est is not None:
                            return _find_inner_estimator(est)
            # 有些版本还会在 CalibratedClassifierCV 自身挂一个 estimator/base_estimator
            for key in ("estimator", "base_estimator", "classifier"):
                est = getattr(m, key, None)
                if est is not None:
                    return _find_inner_estimator(est)
            raise RuntimeError("CalibratedClassifierCV 未暴露内部估计器（estimator/classifier/base_estimator 均不存在）。")

        # 若对象本身就是模型（如 LGBMClassifier），直接返回
        return m

    try:
        # 拆前处理与最后一步
        if hasattr(pipe, "steps") and len(pipe.steps) > 1:
            pre = Pipeline(steps=pipe.steps[:-1])
            last = pipe.steps[-1][1]
        else:
            pre, last = None, pipe

        Xp = pre.transform(x_df) if pre is not None else x_df

        # 剥离到底层 LGBMClassifier
        raw_est = _find_inner_estimator(last)

        # 先尝试 raw_est.predict(..., pred_contrib=True)
        contrib = None
        if hasattr(raw_est, "predict"):
            try:
                contrib = raw_est.predict(Xp, pred_contrib=True)
            except TypeError:
                contrib = None

        # 再尝试 booster_.predict
        if contrib is None and hasattr(raw_est, "booster_"):
            try:
                contrib = raw_est.booster_.predict(Xp, pred_contrib=True)
            except Exception:
                contrib = None

        if contrib is None:
            raise RuntimeError(
                "底层估计器不支持 LightGBM 的 pred_contrib；"
                f"实际类型：{type(raw_est)}，可用属性：{dir(raw_est)}"
            )

        contrib = np.asarray(contrib)
        vec = contrib[0] if contrib.ndim == 2 else contrib
        base = float(vec[-1])      # 最后一列是 bias/base value
        vals = vec[:-1]

        items = []
        for name, val in zip(x_df.columns.tolist(), vals):
            items.append({
                "feature": name,
                "value": float(x_df.iloc[0][name]),
                "contribution": float(val)
            })
        return base, items

    except Exception as e:
        raise RuntimeError(f"无法计算特征贡献（pred_contrib）：{e}")



# —— 升级版：更贴近论文中 SHAP 报告风格（顶部一句话 + Waterfall + Force）——
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _format_risk_text(prob, band, lang="zh"):
    band_map_en = {"极低":"very low", "低":"low", "中":"medium", "高":"high"}
    if lang == "en":
        return f"This patient has {band_map_en.get(band, band)} risk of PBI with the probability of {prob:.4f}."
    else:
        return f"该患儿脑损伤风险为「{band}」；预测概率 {prob:.4f}"

def save_waterfall_tif(base, items, out_path: str, top_k: int = 14, lang="zh"):
    """
    论文风格：横向逐步累计 Waterfall
    - y 轴是按 |贡献| 排序后的特征（从上到下）
    - 每一行是一个“水平浮动条”，从累计起点 left 到 left+width
    - 左侧标 E[f(X)]，右侧标 f(x)
    """
    # 1) 取前 top_k 项（按贡献绝对值大到小）
    items_sorted = sorted(items, key=lambda d: abs(d["contribution"]), reverse=True)[:top_k]
    feats  = [f'{it["feature"]}={it["value"]:.2f}' for it in items_sorted]
    deltas = [float(it["contribution"]) for it in items_sorted]

    # 2) 逐步累计（横向）
    starts = []
    cur = float(base)
    for d in deltas:
        starts.append(cur if d >= 0 else cur + d)
        cur += d
    fx = float(cur)

    # 3) 画图
    n = len(items_sorted)
    y = np.arange(n)[::-1]  # 让“最大贡献”在最上
    colors = ["#d62728" if d >= 0 else "#1f77b4" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=300)
    ax.barh(y, width=[abs(d) for d in deltas], left=starts, height=0.62, color=colors, edgecolor="none")

    # 4) 每段标注 +0.75/-0.22
    for yi, left, d in zip(y, starts, deltas):
        x_pos = left + (abs(d) * (1 if d >= 0 else -1))  # 段末尾位置
        ax.text(x_pos, yi, f"{d:+.2f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8, color="black")

    # 5) 坐标/标签/标题
    ax.set_yticks(y)
    ax.set_yticklabels([feats[i] for i in range(n)][::-1], fontsize=8)
    ax.set_xlabel("f(x) (log-odds)")
    ax.set_title("Waterfall plot of this patient")

    # 6) 范围与基线/终点标注
    xmin = min(base, fx, *(s if d >= 0 else s for s, d in zip(starts, deltas))) - 0.5
    xmax = max(base, fx, *(s + abs(d) for s, d in zip(starts, deltas))) + 0.5
    ax.set_xlim(xmin, xmax)

    ax.axvline(0, color="k", lw=0.6)
    ax.text(base,  n + 0.2, f"E[f(X)] = {base:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(fx,    n + 0.2, f"f(x) = {fx:.3f}",   ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_force_tif(base, items, out_path: str, top_k: int = 14, lang="zh"):
    """
    论文风格：单轴 Force-like 图
    - 在一条水平轴上，从 base 向右(红)/向左(蓝)逐段累计到 f(x)
    - 顶部显示 higher / lower 标签；底部显示 base value 与 f(x)
    """
    items_sorted = sorted(items, key=lambda d: abs(d["contribution"]), reverse=True)[:top_k]

    # 累计区段
    segs = []
    cur = float(base)
    for it in items_sorted:
        c = float(it["contribution"])
        left = cur if c >= 0 else cur + c
        right = cur + c if c >= 0 else cur
        segs.append((left, right, it))
        cur += c
    fx = float(cur)

    # 计算显示范围
    xs = [base, fx] + [p for lr in segs for p in lr[:2]]
    xmin, xmax = min(xs) - 0.5, max(xs) + 0.5

    # 作图
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=300)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    # 主轴
    ax.hlines(0.5, xmin, xmax, colors="#888", lw=1)
    ax.axvline(0, color="k", lw=0.6)

    # 区段 + 标签
    for left, right, it in segs:
        color = "#d62728" if right >= left else "#1f77b4"
        ax.add_patch(Rectangle((min(left, right), 0.36), abs(right-left), 0.28,
                               facecolor=color, edgecolor="none", alpha=0.95))
        ax.text((left+right)/2, 0.50,
                f'{it["feature"]}={it["value"]:.2f}',
                ha="center", va="center", fontsize=8, color="white")

    # 标注 base 与 f(x)
    ax.text(base, 0.86, f"base value = {base:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(fx,   0.86, f"f(x) = {fx:.3f}",       ha="center", va="bottom", fontsize=9)

    # higher / lower
    ax.text(xmax, 0.95, "higher", color="#d62728", ha="right", va="top", fontsize=9)
    ax.text(xmin, 0.95, "lower",  color="#1f77b4", ha="left",  va="top", fontsize=9)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("contribution to f(x) (log-odds)")
    ax.set_title("The force plot of this patient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_shap_report_tif(base, items, prob, band_text, out_path: str, lang="zh", top_k: int = 14):
    """
    一页式报告：顶部“这名患儿…概率 0.xxxx”，下方 Waterfall + Force
    - 将“高/中/低/极低”与概率高亮红色，模仿论文排版
    """
    tmp_wf = "_tmp_wf.tif"
    tmp_fc = "_tmp_fc.tif"
    save_waterfall_tif(base, items, tmp_wf, top_k=top_k, lang=lang)
    save_force_tif(base, items, tmp_fc, top_k=top_k, lang=lang)

    wf = plt.imread(tmp_wf)
    fc = plt.imread(tmp_fc)

    fig, ax = plt.subplots(figsize=(10, 11.2), dpi=300)
    ax.axis("off")

    # —— 顶部大标题（分两次写字以高亮“风险等级”和“概率”）——
    if lang == "en":
        ax.text(0.02, 0.975, "This patient has ", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.31, 0.975, f"{band_text} risk", color="crimson", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.47, 0.975, " of PBI with the probability of ", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.86, 0.975, f"{prob:.4f}", color="crimson", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.02, 0.94,
                "Waterfall plots show how each feature moves the model output from the expected value (E[f(X)]) "
                "to the prediction f(x). Red increases risk; blue decreases risk.", fontsize=10, color="#444",
                ha="left", va="top", transform=ax.transAxes)
    else:
        ax.text(0.02, 0.975, "该患儿脑损伤风险为 ", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.30, 0.975, f"「{band_text}」", color="crimson", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.40, 0.975, "；预测概率 ", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.52, 0.975, f"{prob:.4f}", color="crimson", ha="left", va="top", fontsize=18, transform=ax.transAxes)
        ax.text(0.02, 0.94,
                "瀑布图展示从期望输出 E[f(X)] 到个体输出 f(x) 的逐步贡献；红色↑风险，蓝色↓风险。",
                fontsize=10, color="#444", ha="left", va="top", transform=ax.transAxes)

    # —— 拼图：上半瀑布、下半 force ——
    ax.imshow(wf, extent=[0, 1, 0.44, 0.92])
    ax.imshow(fc, extent=[0, 1, 0.03, 0.41])

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 清理临时图
    try:
        import os
        os.remove(tmp_wf); os.remove(tmp_fc)
    except Exception:
        pass

# =============== 4) 路径与资产 ===============
ROOT = Path(__file__).parent
SKOPS_PATH = ROOT / "final_pipeline.skops"   # 推荐
PIPE_PATH  = ROOT / "final_pipeline.joblib"  # 回退
SCHEMA_PATH = ROOT / "feature_schema.json"   # 若无则使用默认顺序
THR_PATH    = ROOT / "thresholds.json"
RECAL_PATH  = ROOT / "external_recal.json"
META_PATHS = [ROOT / "version.json", ROOT / "release_meta.json"]

# 显示映射（中英）
DISPLAY = {
    "PLT": {"zh": "血小板计数 (PLT, 10^9/L)", "en": "Platelets (PLT, 10^9/L)"},
    "LAC": {"zh": "乳酸 (LAC, mmol/L)", "en": "Lactate (LAC, mmol/L)"},
    "GA_weeks_decimal": {"zh": "胎龄（周）", "en": "Gestational age (weeks, decimal)"},
    "inv_vent_days": {"zh": "有创通气天数 (d)", "en": "Invasive ventilation (days)"},
    "ALB": {"zh": "白蛋白 (ALB, g/L)", "en": "Albumin (ALB, g/L)"},
    "birth_weight_g": {"zh": "出生体重 (g)", "en": "Birth weight (g)"},
    "WBC": {"zh": "白细胞计数 (WBC, 10^9/L)", "en": "WBC (10^9/L)"},
    "Hb": {"zh": "血红蛋白 (Hb, g/L)", "en": "Hemoglobin (g/L)"},
    "BE": {"zh": "碱剩余 (BE, mmol/L)", "en": "Base excess (mmol/L)"},
    "antenatal_mgso4": {"zh": "产前硫酸镁治疗 (0/1)", "en": "Antenatal MgSO4 (0/1)"},
    "surgery": {"zh": "手术 (0/1)", "en": "Surgery (0/1)"},
    "NRDS": {"zh": "新生儿呼吸窘迫综合征 (0/1)", "en": "NRDS (0/1)"},
    "AOP": {"zh": "早产儿贫血 (0/1)", "en": "AOP (0/1)"},
    "sex_male": {"zh": "性别：1=男性，0=女性", "en": "Sex: 1=male, 0=female"},
}

# =============== 5) 加载模型与配置（缓存） ===============
@st.cache_resource
def load_assets():
    # 模型
    pipe = None
    if SKOPS_PATH.exists():
        try:
            import skops.io as sio, inspect
            # 兼容 >=0.10 的关键字参数名
            sig = None
            try:
                sig = inspect.signature(sio.get_untrusted_types)
            except Exception:
                sig = None
            trusted = True
            if sig:
                params = sig.parameters
                if "file" in params:
                    trusted = sio.get_untrusted_types(file=str(SKOPS_PATH))
                elif "path" in params:
                    trusted = sio.get_untrusted_types(path=str(SKOPS_PATH))
                else:
                    trusted = sio.get_untrusted_types(str(SKOPS_PATH))
            pipe = sio.load(str(SKOPS_PATH), trusted=trusted)
            st.info(TEXT["skops_warn"][LANG])
        except Exception as e:
            st.warning(f"读取 SKOPS 失败，将回退到 joblib：{e}")

    if pipe is None:
        if PIPE_PATH.exists():
            pipe = joblib.load(PIPE_PATH)
        else:
            st.error(TEXT["no_model"][LANG]); st.stop()

    # schema / 阈值
    # 若没有 schema，则使用 DEFAULT_FEATURE_ORDER 并构造简单定义
    order = DEFAULT_FEATURE_ORDER.copy()
    feat_defs = {n: {"name": n, "dtype": ("binary" if n in BINARY_FEATURES else "float"),
                     "allowed_range": [None, None], "step": (1 if n in BINARY_FEATURES else 0.1)}
                 for n in order}

    if SCHEMA_PATH.exists():
        try:
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            order = schema.get("order") or [d["name"] for d in schema.get("features", [])] or order
            feats = schema.get("features", [])
            if feats:
                feat_defs = {d["name"]: d for d in feats}
        except Exception as e:
            st.warning(f"读取 feature_schema.json 失败，将使用默认顺序：{e}")

    if not THR_PATH.exists():
        st.error(f"缺少 {THR_PATH.name}"); st.stop()
    thr = json.loads(THR_PATH.read_text(encoding="utf-8"))
    def pick(obj, *keys):
        for k in keys:
            if k in obj:
                v = obj[k]
                return float(v["thr"]) if isinstance(v, dict) else float(v)
        return None
    youden = pick(thr, "youden", "Youden")
    highs  = pick(thr, "high_sensitivity", "HighSens", "highsens")

    # 元信息（有则显示）
    meta = {}
    for m in META_PATHS:
        if m.exists():
            try:
                meta = json.loads(m.read_text(encoding="utf-8"))
                break
            except Exception:
                pass
    recal = None
    if RECAL_PATH.exists():
        try:
            recal = json.loads(RECAL_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"读取 {RECAL_PATH.name} 失败：{e}")

    return pipe, order, feat_defs, {"youden": youden, "highs": highs}, meta, recal

pipe, order, featdefs, thrs, meta, recal = load_assets()


with st.expander(TEXT["meta"][LANG], expanded=False):
    st.json(meta or {"note": "N/A"})

# =============== 6) 工具函数 ===============
def is_binary(name: str, defn: dict) -> bool:
    if name in BINARY_FEATURES:
        return True
    dtype = str(defn.get("dtype","")).lower()
    if dtype == "binary":
        return True
    rng = defn.get("allowed_range") or [None, None]
    if rng[0] == 0 and rng[1] == 1:
        return True
    return False

def label_for(name: str) -> str:
    d = DISPLAY.get(name, {})
    return d.get(LANG, name)

def help_for(defn: dict, is_bin: bool) -> str:
    lo, hi = (defn.get("allowed_range") or [None, None])
    step = defn.get("step", 1 if is_bin else 0.1)
    parts = []
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        parts.append(f'{TEXT["range_help"][LANG]}: {lo}–{hi}')
    parts.append(f'{TEXT["step_help"][LANG]}: {step}')
    if is_bin:
        parts.append(TEXT["binary_help"][LANG])
    return " · ".join(parts)

# =============== 7) 表单输入 ===============
st.markdown(f"### {TEXT['input_section'][LANG]}")
cols = st.columns(2)
values = {}

for i, name in enumerate(order):
    d = featdefs.get(name, {})
    bin_flag = is_binary(name, d)
    lbl = label_for(name)
    help_text = help_for(d, bin_flag)
    lo, hi = (d.get("allowed_range") or [None, None])

    with cols[i % 2]:
        if bin_flag:
            # 0/1 选择；sex_male: 1=男性, 0=女性
            values[name] = st.selectbox(lbl, options=[0, 1], index=0, help=help_text, key=f"bin_{name}")
        else:
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                default = float((lo + hi) / 2.0)
                step = float(d.get("step", 0.1))
                values[name] = st.number_input(lbl, value=default,
                                               min_value=float(lo), max_value=float(hi),
                                               step=step, format="%.3f", help=help_text)
            else:
                values[name] = st.number_input(lbl, value=0.0,
                                               step=float(d.get("step", 0.1)),
                                               format="%.3f", help=help_text)

st.caption(TEXT["binary_help"][LANG])

st.divider()
mode = st.radio(TEXT["thresh_mode"][LANG], [TEXT["youd"][LANG], TEXT["hsens"][LANG]], horizontal=True)
btn_predict = st.button(TEXT["predict"][LANG], type="primary")

# =============== 8) 阈值处理 ===============
def pick_thresholds(thrs_dict):
    t_youden = thrs_dict.get("youden")
    t_hsens  = thrs_dict.get("highs")
    if t_youden is None: t_youden = 0.5
    if t_hsens  is None: t_hsens  = 0.2
    lo, hi = sorted([float(t_youden), float(t_hsens)])
    return lo, hi, float(t_youden), float(t_hsens)
low_thr, high_thr, t_youden, t_hsens = pick_thresholds(thrs)

def risk_bucket(p: float, lang="zh") -> str:
    if p < low_thr: return TEXT["risk_low"][lang]
    elif p < high_thr: return TEXT["risk_mid"][lang]
    else: return TEXT["risk_high"][lang]

# =============== 9) 单例预测 + 报告下载 ===============
if btn_predict:
    x = pd.DataFrame([values])[order]  # 严格列顺序
    try:
        p = float(pipe.predict_proba(x)[:, 1][0])
        # 部署端再校准（可选）
        if recal:
            a = float(recal.get("intercept", 0.0))
            b = float(recal.get("slope", 1.0))
            eps = 1e-12
            z = np.log((p + eps) / (1 - p + eps))
            p = 1.0 / (1.0 + np.exp(-(a + b * z)))
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.stop()

    youden_thr = float(thrs["youden"])
    hs_thr = float(thrs["highs"])

    st.subheader(TEXT["probability"][LANG])
    st.metric(TEXT["probability"][LANG], f"{p:.4f}")

    # —— 两种策略下的四档风险分级 ——
    band_youden = risk_band_from_prob(p, hs_thr, youden_thr)
    band_highs = risk_band_from_prob(p, hs_thr, youden_thr)  # 分级规则一致，只是口径不同时你也可单独给阈值

    st.write(f"**{TEXT['band_title'][LANG]}**")

    c1, c2 = st.columns(2)


    # 小工具：把“极低/低/中/高”翻成英文
    def _band_disp(b):
        return {"极低": "very low", "低": "low", "中": "medium", "高": "high"}.get(b, b)


    with c1:
        if LANG == "en":
            dec_y = "Positive" if p >= youden_thr else "Negative"
            st.info(
                f"**Youden**: threshold={youden_thr:.6f} → decision: **{dec_y}**; "
                f"risk band: **{_band_disp(band_youden)}**"
            )
        else:
            dec_y = "阳性" if p >= youden_thr else "阴性"
            st.info(
                f"**Youden**：阈值={youden_thr:.6f} → 判定：**{dec_y}**；风险分级：**{band_youden}**"
            )

    with c2:
        if LANG == "en":
            dec_h = "Positive" if p >= hs_thr else "Negative"
            st.info(
                f"**{TEXT['hsens'][LANG]}**: threshold={hs_thr:.6f} → decision: **{dec_h}**; "
                f"risk band: **{_band_disp(band_highs)}**"
            )
        else:
            dec_h = "阳性" if p >= hs_thr else "阴性"
            st.info(
                f"**高敏**：阈值={hs_thr:.6f} → 判定：**{dec_h}**；风险分级：**{band_highs}**"
            )

    # —— 计算 LightGBM 贡献并绘图/导出 ——
    import pandas as pd, numpy as np, io, zipfile, os

    out_files = {}

    try:
        base, items = compute_lgbm_contrib(pipe, x)
        # 保存 CSV（贡献明细）
        df_contrib = pd.DataFrame(items)  # feature, value, contribution
        out_files["shap_contrib.csv"] = df_contrib.to_csv(index=False).encode("utf-8")
        # 保存 summary.csv（概率、阈值、分级）
        df_summary = pd.DataFrame([{
            "p_hat": p,
            "youden_thr": youden_thr, "decision_youden": ("POS" if p >= youden_thr else "NEG"),
            "highs_thr": hs_thr, "decision_highs": ("POS" if p >= hs_thr else "NEG"),
            "band_youden": band_youden, "band_highs": band_highs,
            "base_logit": base
        }])
        out_files["summary.csv"] = df_summary.to_csv(index=False).encode("utf-8")

        # 仅生成与展示两张单图（不再生成“顶部结论”的整页报告）
        import matplotlib
        import matplotlib.pyplot as plt


        def _set_step5_style():
            try:
                matplotlib.rcParams["font.family"] = "Times New Roman"
            except Exception:
                pass
            matplotlib.rcParams["axes.unicode_minus"] = False


        def save_waterfall_tif(base, items, out_path: str, top_k: int = 14, lang="zh", dpi: int = 600):
            _set_step5_style()
            # 取贡献绝对值最大的 top_k 个
            items_sorted = sorted(items, key=lambda d: abs(d["contribution"]), reverse=True)[:top_k]

            deltas, labels = [], []
            for it in items_sorted:
                deltas.append(float(it["contribution"]))
                v = it["value"]
                v_txt = f"{int(v)}" if isinstance(v, (int, float)) and float(v).is_integer() else f"{float(v):.2f}"
                labels.append(f'{it["feature"]}={v_txt}')

            # 逐段累计（竖向 waterfall：每个柱子的 bottom=累计到上一段的值）
            starts, cur = [], float(base)
            for d in deltas:
                starts.append(cur)
                cur += d

            colors = ["#d62728" if d >= 0 else "#1f77b4" for d in deltas]
            fig, ax = plt.subplots(figsize=(10, 5.2), dpi=dpi)
            bars = ax.bar(range(len(deltas)), deltas, bottom=starts, color=colors, width=0.6, edgecolor="none")
            ax.axhline(0, color="#333", lw=0.6)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=9)
            ax.set_ylabel("f(x) (log-odds)", fontsize=11)
            ax.set_title("Waterfall plot of this patient", fontsize=12)

            # —— 在每个条上标出具体的正负数值 —— #
            # 计算 y 方向边距，用于文字轻微偏移，避免与柱边缘重合
            tops = [s + d for s, d in zip(starts, deltas)]
            ymin = min([base] + starts + tops)
            ymax = max([base] + starts + tops)
            yspan = max(1e-6, ymax - ymin)
            pad = 0.015 * yspan

            for i, (s, d) in enumerate(zip(starts, deltas)):
                y_text = s + d + (pad if d >= 0 else -pad)
                ax.text(i, y_text, f"{d:+.2f}", ha="center",
                        va=("bottom" if d >= 0 else "top"),
                        fontsize=9, color="black")

            # 标注 E[f(X)]
            ax.text(-0.5, base, f"E[f(X)] = {base:.3f}", va="center", fontsize=9, color="#444")

            # y 轴留一点空隙，避免最上/最下文字被裁切
            ax.set_ylim(ymin - 0.05 * yspan, ymax + 0.08 * yspan)

            fig.tight_layout()
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)


        def save_force_tif(base, items, out_path: str, top_k: int = 14, lang="zh", dpi: int = 600):
            _set_step5_style()
            # 先按贡献从小到大排，随后做“负向前半 + 正向后半”的截取，保证两边信息对称
            items_sorted = sorted(items, key=lambda d: d["contribution"])
            if top_k and top_k < len(items_sorted):
                neg = [it for it in items_sorted if it["contribution"] < 0]
                pos = [it for it in items_sorted if it["contribution"] >= 0]
                k2 = max(1, top_k // 2)
                items_sorted = neg[:k2] + pos[-k2:]

            vals, labels = [], []
            for it in items_sorted:
                c = float(it["contribution"])
                vals.append(c)
                v = it["value"]
                v_txt = f"{int(v)}" if isinstance(v, (int, float)) and float(v).is_integer() else f"{float(v):.2f}"
                labels.append(f'{it["feature"]}={v_txt}')

            colors = ["#1f77b4" if v < 0 else "#d62728" for v in vals]
            h = max(2.8, 0.55 * len(vals) + 1.8)  # 动态高度，避免“堆在一起”
            fig, ax = plt.subplots(figsize=(10, h), dpi=dpi)
            y = list(range(len(vals)))
            ax.barh(y, vals, color=colors, edgecolor="none")
            ax.axvline(0, color="#333", lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("contribution to f(x) (log-odds)", fontsize=11)
            ax.set_title("The force plot of this patient", fontsize=12)

            # —— 为每个横条标出数值；并自动留左右边距 —— #
            xmin = min(0.0, min(vals))  # 基于数据估一个范围
            xmax = max(0.0, max(vals))
            xspan = max(1e-6, xmax - xmin)
            pad = 0.02 * xspan

            for yi, v in enumerate(vals):
                # 数值贴在条形末端外侧：正数靠右、负数靠左
                x_text = v + (pad if v >= 0 else -pad)
                ax.text(x_text, yi, f"{v:+.2f}",
                        va="center",
                        ha=("left" if v >= 0 else "right"),
                        fontsize=9, color="black")

            # 给坐标两端各加 8% 空间，确保文字不被裁切
            ax.set_xlim(xmin - 0.08 * xspan, xmax + 0.08 * xspan)

            fig.tight_layout()
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)


        # === 实际调用：生成图片 + 页面展示 + 加入导出 ===
        wf_path = "waterfall_patient.tif"
        fc_path = "force_patient.tif"
        save_waterfall_tif(base, items, wf_path, top_k=14, dpi=600)
        save_force_tif(base, items, fc_path, top_k=14, dpi=600)

        with st.expander(TEXT["view_imgs"][LANG], expanded=True):
            st.image(wf_path, caption="Waterfall", use_column_width=True)
            st.image(fc_path, caption="Force", use_column_width=True)

        from pathlib import Path

        out_files["waterfall.tif"] = Path(wf_path).read_bytes()
        out_files["force.tif"] = Path(fc_path).read_bytes()



    except Exception as e:
        st.warning(f"未能生成贡献图：{e}")

    # —— 打包导出 ——
    if out_files:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for nm, data in out_files.items():
                zf.writestr(nm, data)
        st.download_button(TEXT["download_zip_all"][LANG],
                           data=zip_buf.getvalue(),
                           file_name="pbi_single_prediction_exports.zip",
                           mime="application/zip")

    y1 = int(p >= t_youden)
    y2 = int(p >= t_hsens)

    st.metric(TEXT["probability"][LANG], f"{p:.4f}")
    st.write(f"**{TEXT['two_rules'][LANG]}**")
    st.write(f"- Youden @ {t_youden:.6f} → **{TEXT['pos'][LANG] if y1 else TEXT['neg'][LANG]}**")
    st.write(f"- {TEXT['hsens'][LANG]} @ {t_hsens:.6f} → **{TEXT['pos'][LANG] if y2 else TEXT['neg'][LANG]}**")

    final_label = y1 if mode == TEXT["youd"][LANG] else y2
    st.success(f"{TEXT['current_mode'][LANG]}：**{mode}** → {TEXT['result'][LANG]}：**{TEXT['pos'][LANG] if final_label else TEXT['neg'][LANG]}**")

    risk_sent = TEXT["risk_sentence"][LANG].format(level=risk_bucket(p, LANG), p=p)
    st.write(risk_sent)

    # 简易 HTML 报告
    rows_html = "".join([f"<tr><td>{n}</td><td>{values[n]}</td></tr>" for n in order])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PBI Report</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:24px auto;}}
h1{{font-size:22px;}} table{{border-collapse:collapse;width:100%;}}
td,th{{border:1px solid #ddd;padding:6px 8px;}} .small{{color:#666;font-size:12px;}}
</style></head>
<body>
<h1>{TEXT['title'][LANG]}</h1>
<div class="small">version: {meta.get('model_name','N/A')} · build: {meta.get('timestamp','N/A')}</div>
<h2>{TEXT['input_section'][LANG]}</h2>
<table><tr><th>Feature</th><th>Value</th></tr>{rows_html}</table>
<h2>{TEXT['result'][LANG]}</h2>
<p>{TEXT['probability'][LANG]}：<b>{p:.4f}</b></p>
<ul>
<li>Youden@{t_youden:.6f} → {'POS' if y1 else 'NEG'}</li>
<li>{TEXT['hsens'][LANG]}@{t_hsens:.6f} → {'POS' if y2 else 'NEG'}</li>
</ul>
<p>{risk_sent}</p>
<p class="small">{TEXT['caption'][LANG]}</p>
</body></html>"""
    st.download_button(TEXT["download_html"][LANG], data=html.encode("utf-8"), file_name="pbi_report.html",
                       mime="text/html")

# =============== 10) 批量 CSV 预测 ===============
st.divider()
st.markdown(f"### {TEXT['batch_title'][LANG]}")
st.caption(TEXT["batch_caption"][LANG])
up = st.file_uploader(TEXT["upload_csv"][LANG], type=["csv"])
if up is not None:
    try:
        df = pd.read_csv(up)
        # 丢失列补 NaN；多余列忽略；严格顺序
        for col in order:
            if col not in df.columns:
                df[col] = np.nan
        X = df[order]
        proba = pipe.predict_proba(X)[:, 1]
        df_out = df.copy()
        df_out["p_hat"] = proba
        df_out["label_youden"] = (df_out["p_hat"] >= t_youden).astype(int)
        df_out["label_highsens"] = (df_out["p_hat"] >= t_hsens).astype(int)
        csv_bytes = df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.success(TEXT["done_n"][LANG].format(n=len(df_out)))
        st.download_button(TEXT["download_csv"][LANG], data=csv_bytes, file_name="pbi_pred_results.csv",
                           mime="text/csv")
    except Exception as e:
        st.error(f"读取或预测失败：{e}")
