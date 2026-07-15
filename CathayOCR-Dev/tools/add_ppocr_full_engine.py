#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modify UMI-OCR-Proxy UI to add ppocr_full engine registration"""
import sys, re

UI_PATH = r'D:\OCR-Proxy-v2\UMI-OCR-Proxy\umi_ocr_pdf_processor_ui.py'

with open(UI_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# ============================================================
# 1. Add register_engine for ppocr_full (after ncnn_cpu)
# ============================================================
anchor1 = "    priority=4\n)\n\n\n# ============================================================\n# 路径自动查找 - 多引擎支持"
new_engine = """    priority=4
)


register_engine(
    'ppocr_full', 'PP-OCR Full (Paddle完整版)',
    'PaddleOCR 完整版引擎\\n支持全部官方语言:\\n拉丁(50) / 阿拉伯(8) / 西里尔(33) / 天城文(13) / 韩文/泰文等\\n自动选择PP-OCR v5/v6版本\\n基于PaddlePaddle框架\\nGPU模式需CUDA',
    'ppocr_full', 'PaddleOCR-Full.bat', 'pipe',
    supports_gpu=True, supports_cpu=True,
    model_options=[('auto', '自动(按语言选版本)')],
    supported_params=['cls', 'det', 'rec_batch_num'],
    priority=11
)


# ============================================================
# 路径自动查找 - 多引擎支持"""

if anchor1 in content:
    content = content.replace(anchor1, new_engine)
    changes += 1
    print(f"[1/5] ✓ Register ppocr_full engine (line ~151)")
else:
    print(f"[1/5] ✗ anchor1 not found")


# ============================================================
# 2. Add ppocr_full language handling in build_engine_params
# ============================================================
# After the win7_classic handling
anchor2 = """        elif engine_id == "win7_classic":
            params["enable_mkldnn"] = True
            params["cpu_threads"] = extra_params.get("cpu_threads", 4)
            params["config_path"] = "models/config_chinese.txt\""""
new_params = """        elif engine_id == "win7_classic":
            params["enable_mkldnn"] = True
            params["cpu_threads"] = extra_params.get("cpu_threads", 4)
            params["config_path"] = "models/config_chinese.txt"
        elif engine_id == "ppocr_full":
            lang = extra_params.get("lang", "ch") or "ch"
            params["language"] = lang
            params["ppocr_version"] = extra_params.get("ppocr_full_version", None)"""

if anchor2 in content:
    content = content.replace(anchor2, new_params)
    changes += 1
    print(f"[2/5] ✓ ppocr_full in build_engine_params")
else:
    print(f"[2/5] ✗ anchor2 not found")


# ============================================================
# 3. Add language dropdown for ppocr_full in _populate_lang_combo
# ============================================================
# After easyocr_universal section
anchor3 = """        elif eid == "easyocr_universal":
            self.lang_combo.addItems(["English (EasyOCR)", "Fran\u00e7ais (EasyOCR)", "Italiano (EasyOCR)", "Espa\u00f1ol (EasyOCR)"])
            self.lang_combo.setEnabled(True)"""
new_lang_dropdown = """        elif eid == "easyocr_universal":
            self.lang_combo.addItems(["English (EasyOCR)", "Fran\u00e7ais (EasyOCR)", "Italiano (EasyOCR)", "Espa\u00f1ol (EasyOCR)"])
            self.lang_combo.setEnabled(True)
        elif eid == "ppocr_full":
            # 拉丁语系 (Latin, 50种)
            self.lang_combo.addItems([
                "\u4e2d\u6587", "English", "Fran\u00e7ais", "Espa\u00f1ol", "Italiano",
                "Deutsch", "Portugu\u00eas",
                "--- \u62c9\u4e01\u8bed\u7cfb ---",
                "Afrikaans", "Az\u0259rbaycan", "Bosanski", "\u010ce\u0161tina", "Cymraeg",
                "Dansk", "Eesti", "Euskara", "F\u00f6\u00f6r bunne", "Frysk",
                "Gaeilge", "Galego", "Hrvatski", "Bahasa Indonesia", "\u00cdslenska",
                "Kurd\u00ee", "Latina", "Latvie\u0161u", "Lietuvi\u0173", "Magyar",
                "Malti", "M\u0101ori", "Melayu", "Nederlands", "Norsk",
                "Occitan", "Pisin", "Polski", "Rom\u00e2n\u0103", "Shqip",
                "Sloven\u010dina", "Sloven\u0161\u010dina", "Soomaali", "Svenska", "Kiswahili",
                "Tagalog", "T\u00fcrk\u00e7e", "O\u02bczbek", "Ti\u1ebfng Vi\u1ec7t",
                "Suomi", "Corsu", "Rumantsch", "Catal\u00e0", "Runasimi",
                # 阿拉伯语系 (Arabic, 8种)
                "--- \u963f\u62c9\u4f2f\u8bed\u7cfb ---",
                "\u0627\u0644\u0639\u0631\u0628\u064a\u0629", "\u0641\u0627\u0631\u0633\u06cc",
                "\u0626\u06c7\u064a\u063a\u064f\u0631", "\u0627\u0631\u062f\u0648",
                "\u067e\u069a\u062a\u0648", "Kurmanc\u00ee",
                "\u0633\u0646\u068c\u064a", "\u0628\u064e\u0644\u0648\u0686\u06cc",
                # 西里尔语系 (Cyrillic, 33种 + EastSlavic 3)
                "--- \u897f\u91cc\u5c14\u8bed\u7cfb ---",
                "\u0420\u0443\u0441\u0441\u043a\u0438\u0439", "\u0411\u0435\u043b\u0430\u0440\u0443\u0441\u043a\u0430\u044f",
                "\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430",
                "\u0421\u0440\u043f\u0441\u043a\u0438 (\u045b\u0438\u0440\u0438\u043b\u0438\u0446\u0430)",
                "\u0411\u044a\u043b\u0433\u0430\u0440\u0441\u043a\u0438", "\u041c\u043e\u043d\u0433\u043e\u043b",
                "\u0410\u0431\u0430\u0437\u0430", "\u0410\u0434\u044b\u0433\u0430\u0431\u0437\u044d",
                "\u041a\u044a\u044f\u0431\u0436\u044d\u0440\u0434\u0437\u044d\u0439", "\u0410\u0432\u0430\u0440",
                "\u0414\u0430\u0440\u0433\u0432\u0430", "\u0413\u0401\u0430\u043b\u0433\u04c0\u0430\u0439",
                "\u041d\u043e\u0445\u0447\u0438\u0439\u043d", "\u041b\u0430\u043a\u043a\u0443",
                "\u041b\u0435\u0437\u0433\u0438", "\u0422\u0430\u0431\u0430\u0441\u0430\u0440\u0430\u043d",
                "\u049a\u0430\u0437\u0430\u049b", "\u041a\u044b\u0440\u0433\u044b\u0437",
                "\u0422\u043e\u0497\u0438\u043a\u04e3", "\u041c\u0430\u043a\u0435\u0434\u043e\u043d\u0441\u043a\u0438",
                "\u0422\u0430\u0442\u0430\u0440", "\u0427\u04d1\u0432\u0430\u0448",
                "\u0411\u0430\u0448\u04a1\u043e\u0440\u0442", "\u041e\u043b\u044b\u043c\u0430\u0440\u0438\u0439",
                "\u041c\u043e\u043b\u0434\u043e\u0432\u0435\u043d\u044f\u0441\u043a\u044d",
                "\u0423\u0434\u043c\u0443\u0440\u0442", "\u041a\u043e\u043c\u0438",
                "\u0418\u0440\u043e\u043d", "\u0411\u0443\u0440\u044f\u0430\u0434",
                "\u0425\u0430\u043b\u044c\u043c\u0433", "\u0422\u044b\u0432\u0430",
                "\u0421\u0430\u0445\u0430", "\u041a\u0430\u0440\u0430\u043a\u0430\u043b\u043f\u0430\u049b",
                # 天城文 (Devanagari, 13种)
                "--- \u5929\u57ce\u6587 ---",
                "\u0939\u093f\u0928\u094d\u0926\u0940", "\u092e\u0930\u093e\u0920\u0940",
                "\u0928\u0947\u092a\u093e\u0932\u0940", "\u092d\u094b\u091c\u092a\u0941\u0930\u0940",
                "\u092e\u0948\u0925\u093f\u0932\u0940", "\u0905\u0902\u0917\u093f\u0915\u093e",
                "\u092d\u094b\u091c\u092a\u0941\u0930\u0940", "\u092e\u093e\u0932\u094d\u0935\u093e",
                "\u0938\u0926\u094d\u0926\u093e", "\u0928\u0947\u0935\u093e\u0930\u0940",
                "\u0917\u094b\u092e\u093f\u0924", "\u0938\u0902\u0938\u094d\u0915\u0943\u0924",
                "\u092c\u093e\u0917\u0947\u0932\u0940",
                # 其他语言
                "--- \u5176\u4ed6\u8bed\u8a00 ---",
                "\ud55c\uad6d\uc5b4", "\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22",
                "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac",
                "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41", "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd",
                "\u65e5\u672c\u8a9e", "\u7e41\u9ad4\u4e2d\u6587",
            ])
            self.lang_combo.setEnabled(True)"""

if anchor3 in content:
    content = content.replace(anchor3, new_lang_dropdown)
    changes += 1
    print(f"[3/5] ✓ ppocr_full language dropdown")
else:
    print(f"[3/5] ✗ anchor3 not found")


# ============================================================
# 4. Add ppocr_full language mapping entries
# ============================================================
# Right before the EasyOCR languages section
anchor4 = """            # EasyOCR languages (only Latin: gen2 models work correctly)
            "Fran\\u00e7ais (EasyOCR)": "fr\","""
new_mapping = """            # PP-OCR Full languages
            "--- \\u62c9\\u4e01\\u8bed\\u7cfb ---": "latin",
            "--- \\u963f\\u62c9\\u4f2f\\u8bed\\u7cfb ---": "ar",
            "--- \\u897f\\u91cc\\u5c14\\u8bed\\u7cfb ---": "ru",
            "--- \\u5929\\u57ce\\u6587 ---": "hi",
            "--- \\u5176\\u4ed6\\u8bed\\u8a00 ---": "ch",
            # Latin
            "Afrikaans": "af", "Az\\u0259rbaycan": "az", "Bosanski": "bs",
            "\\u010ce\\u0161tina": "cs", "Cymraeg": "cy", "Dansk": "da",
            "Eesti": "et", "Euskara": "eu", "F\\u00f6\\u00f6r bunne": "lb",
            "Frysk": "fy", "Gaeilge": "ga", "Galego": "gl",
            "Hrvatski": "hr", "Bahasa Indonesia": "id",
            "\\u00cdslenska": "is", "Kurd\\u00ee": "ku",
            "Latina": "la", "Latvie\\u0161u": "lv", "Lietuvi\\u0173": "lt",
            "Magyar": "hu", "Malti": "mt", "M\\u0101ori": "mi",
            "Melayu": "ms", "Nederlands": "nl", "Norsk": "no",
            "Occitan": "oc", "Pisin": "pi", "Polski": "pl",
            "Rom\\u00e2n\\u0103": "ro", "Shqip": "sq",
            "Sloven\\u010dina": "sk", "Sloven\\u0161\\u010dina": "sl",
            "Soomaali": "sw", "Svenska": "sv", "Kiswahili": "sw",
            "Tagalog": "tl", "T\\u00fcrk\\u00e7e": "tr",
            "O\\u02bzbek": "uz", "Ti\\u1ebfng Vi\\u1ec7t": "vi",
            "Suomi": "fi", "Corsu": "co", "Rumantsch": "rm",
            "Catal\\u00e0": "ca", "Runasimi": "qu",
            # Arabic
            "\\u0627\\u0644\\u0639\\u0631\\u0628\\u064a\\u0629": "ar",
            "\\u0641\\u0627\\u0631\\u0633\\u06cc": "fa",
            "\\u0626\\u06c7\\u064a\\u063a\\u064f\\u0631": "ug",
            "\\u0627\\u0631\\u062f\\u0648": "ur",
            "\\u067e\\u069a\\u062a\\u0648": "ps",
            "Kurmanc\\u00ee": "ku",
            "\\u0633\\u0646\\u068c\\u064a": "sd",
            "\\u0628\\u064e\\u0644\\u0648\\u0686\\u06cc": "bal",
            # Cyrillic
            "\\u0420\\u0443\\u0441\\u0441\\u043a\\u0438\\u0439": "ru",
            "\\u0411\\u0435\\u043b\\u0430\\u0440\\u0443\\u0441\\u043a\\u0430\\u044f": "be",
            "\\u0423\\u043a\\u0440\\u0430\\u0457\\u043d\\u0441\\u044c\\u043a\\u0430": "uk",
            "\\u0421\\u0440\\u043f\\u0441\\u043a\\u0438 (\\u045b\\u0438\\u0440\\u0438\\u043b\\u0438\\u0446\\u0430)": "rs_cyrillic",
            "\\u0411\\u044a\\u043b\\u0433\\u0430\\u0440\\u0441\\u043a\\u0438": "bg",
            "\\u041c\\u043e\\u043d\\u0433\\u043e\\u043b": "mn",
            "\\u0410\\u0431\\u0430\\u0437\\u0430": "abq",
            "\\u0410\\u0434\\u044b\\u0433\\u0430\\u0431\\u0437\\u044d": "ady",
            "\\u041a\\u044a\\u044f\\u0431\\u0436\\u044d\\u0440\\u0434\\u0437\\u044d\\u0439": "kbd",
            "\\u0410\\u0432\\u0430\\u0440": "ava",
            "\\u0414\\u0430\\u0440\\u0433\\u0432\\u0430": "dar",
            "\\u0413\\u0401\\u0430\\u043b\\u0433\\u04c0\\u0430\\u0439": "inh",
            "\\u041d\\u043e\\u0445\\u0447\\u0438\\u0439\\u043d": "che",
            "\\u041b\\u0430\\u043a\\u043a\\u0443": "lbe",
            "\\u041b\\u0435\\u0437\\u0433\\u0438": "lez",
            "\\u0422\\u0430\\u0431\\u0430\\u0441\\u0430\\u0440\\u0430\\u043d": "tab",
            "\\u049a\\u0430\\u0437\\u0430\\u049b": "kk",
            "\\u041a\\u044b\\u0440\\u0433\\u044b\\u0437": "ky",
            "\\u0422\\u043e\\u0497\\u0438\\u043a\\u04e3": "tg",
            "\\u041c\\u0430\\u043a\\u0435\\u0434\\u043e\\u043d\\u0441\\u043a\\u0438": "mk",
            "\\u0422\\u0430\\u0442\\u0430\\u0440": "tt",
            "\\u0427\\u04d1\\u0432\\u0430\\u0448": "cv",
            "\\u0411\\u0430\\u0448\\u04a1\\u043e\\u0440\\u0442": "ba",
            "\\u041e\\u043b\\u044b\\u043c\\u0430\\u0440\\u0438\\u0439": "mhr",
            "\\u041c\\u043e\\u043b\\u0434\\u043e\\u0432\\u0435\\u043d\\u044f\\u0441\\u043a\\u044d": "mo",
            "\\u0423\\u0434\\u043c\\u0443\\u0440\\u0442": "udm",
            "\\u041a\\u043e\\u043c\\u0438": "kv",
            "\\u0418\\u0440\\u043e\\u043d": "os",
            "\\u0411\\u0443\\u0440\\u044f\\u0430\\u0434": "bua",
            "\\u0425\\u0430\\u043b\\u044c\\u043c\\u0433": "xal",
            "\\u0422\\u044b\\u0432\\u0430": "tyv",
            "\\u0421\\u0430\\u0445\\u0430": "sah",
            "\\u041a\\u0430\\u0440\\u0430\\u043a\\u0430\\u043b\\u043f\\u0430\\u049b": "kaa",
            # Devanagari
            "\\u0939\\u093f\\u0928\\u094d\\u0926\\u0940": "hi",
            "\\u092e\\u0930\\u093e\\u0920\\u0940": "mr",
            "\\u0928\\u0947\\u092a\\u093e\\u0932\\u0940": "ne",
            "\\u092d\\u094b\\u091c\\u092a\\u0941\\u0930\\u0940": "bh",
            "\\u092e\\u0948\\u0925\\u093f\\u0932\\u0940": "mai",
            "\\u0905\\u0902\\u0917\\u093f\\u0915\\u093e": "ang",
            "\\u092d\\u094b\\u091c\\u092a\\u0941\\u0930\\u0940": "bho",
            "\\u092e\\u093e\\u0932\\u094d\\u0935\\u093e": "mah",
            "\\u0938\\u0926\\u094d\\u0926\\u093e": "sck",
            "\\u0928\\u0947\\u0935\\u093e\\u0930\\u0940": "new",
            "\\u0917\\u094b\\u092e\\u093f\\u0924": "gom",
            "\\u0938\\u0902\\u0938\\u094d\\u0915\\u0943\\u0924": "sa",
            "\\u092c\\u093e\\u0917\\u0947\\u0932\\u0940": "bgc",
            # Other
            "\\ud55c\\uad6d\\uc5b4": "korean",
            "\\u0e20\\u0e32\\u0e29\\u0e32\\u0e44\\u0e17\\u0e22": "th",
            "\\u0395\\u03bb\\u03bb\\u03b7\\u03bd\\u03b9\\u03ba\\u03ac": "el",
            "\\u0c24\\u0c46\\u0c32\\u0c41\\u0c17\\u0c41": "te",
            "\\u0ba4\\u0bae\\u0bbf\\u0bb4\\u0bcd": "ta",
            # EasyOCR languages (only Latin: gen2 models work correctly)
            "Fran\\u00e7ais (EasyOCR)": "fr\","""

if anchor4 in content:
    content = content.replace(anchor4, new_mapping)
    changes += 1
    print(f"[4/5] ✓ ppocr_full language mapping")
else:
    print(f"[4/5] ✗ anchor4 not found")


# ============================================================
# 5. Add ppocr_full handling in the OCR start logic
# ============================================================
# Add ppocr_version passthrough
anchor5 = """        extra_params["lang"] = ocr_lang
        if engine_id == "easyocr_universal":
            extra_params["easyocr_lang"] = lang_map.get(lang_display, "en")"""
new_start = """        extra_params["lang"] = ocr_lang
        if engine_id == "easyocr_universal":
            extra_params["easyocr_lang"] = lang_map.get(lang_display, "en")
        if engine_id == "ppocr_full":
            extra_params["easyocr_lang"] = None  # not used for ppocr_full"""

if anchor5 in content:
    content = content.replace(anchor5, new_start)
    changes += 1
    print(f"[5/5] ✓ ppocr_full start logic")
else:
    print(f"[5/5] ✗ anchor5 not found")


# Save
with open(UI_PATH, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"\nDone: {changes}/5 changes applied")
