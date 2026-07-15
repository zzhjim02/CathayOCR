#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace huge PP-OCR Full mapping with concise version matching simplified dropdown"""
UI_PATH = r'D:\OCR-Proxy-v2\UMI-OCR-Proxy\umi_ocr_pdf_processor_ui.py'

with open(UI_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

old_block = r"""            # PP-OCR Full languages
            \"--- \\u62c9\\u4e01\\u8bed\\u7cfb ---\": \"latin\",
            \"--- \\u963f\\u62c9\\u4f2f\\u8bed\\u7cfb ---\": \"ar\",
            \"--- \\u897f\\u91cc\\u5c14\\u8bed\\u7cfb ---\": \"ru\",
            \"--- \\u5929\\u57ce\\u6587 ---\": \"hi\",
            \"--- \\u5176\\u4ed6\\u8bed\\u8a00 ---\": \"ch\",
            # Latin
            \"Afrikaans\": \"af\", \"Az\\u0259rbaycan\": \"az\", \"Bosanski\": \"bs\",
            \"\\u010ce\\u0161tina\": \"cs\", \"Cymraeg\": \"cy\", \"Dansk\": \"da\",
            \"Eesti\": \"et\", \"Euskara\": \"eu\", \"F\\u00f6\\u00f6r bunne\": \"lb\",
            \"Frysk\": \"fy\", \"Gaeilge\": \"ga\", \"Galego\": \"gl\",
            \"Hrvatski\": \"hr\", \"Bahasa Indonesia\": \"id\",
            \"\\u00cdslenska\": \"is\", \"Kurd\\u00ee\": \"ku\",
            \"Latina\": \"la\", \"Latvie\\u0161u\": \"lv\", \"Lietuvi\\u0173\": \"lt\",
            \"Magyar\": \"hu\", \"Malti\": \"mt\", \"M\\u0101ori\": \"mi\",
            \"Melayu\": \"ms\", \"Nederlands\": \"nl\", \"Norsk\": \"no\",
            \"Occitan\": \"oc\", \"Pisin\": \"pi\", \"Polski\": \"pl\",
            \"Rom\\u00e2n\\u0103\": \"ro\", \"Shqip\": \"sq\",
            \"Sloven\\u010dina\": \"sk\", \"Sloven\\u0161\\u010dina\": \"sl\",
            \"Soomaali\": \"sw\", \"Svenska\": \"sv\", \"Kiswahili\": \"sw\",
            \"Tagalog\": \"tl\", \"T\\u00fcrk\\u00e7e\": \"tr\",
            \"O\\u02bzbek\": \"uz\", \"Ti\\u1ebfng Vi\\u1ec7t\": \"vi\",
            \"Suomi\": \"fi\", \"Corsu\": \"co\", \"Rumantsch\": \"rm\",
            \"Catal\\u00e0\": \"ca\", \"Runasimi\": \"qu\",
            # Arabic
            \"\\u0627\\u0644\\u0639\\u0631\\u0628\\u064a\\u0629\": \"ar\",
            \"\\u0641\\u0627\\u0631\\u0633\\u06cc\": \"fa\",
            \"\\u0626\\u06c7\\u064a\\u063a\\u064f\\u0631\": \"ug\",
            \"\\u0627\\u0631\\u062f\\u0648\": \"ur\",
            \"\\u067e\\u069a\\u062a\\u0648\": \"ps\",
            \"Kurmanc\\u00ee\": \"ku\",
            \"\\u0633\\u0646\\u068c\\u064a\": \"sd\",
            \"\\u0628\\u064e\\u0644\\u0648\\u0686\\u06cc\": \"bal\",
            # Cyrillic
            \"\\u0420\\u0443\\u0441\\u0441\\u043a\\u0438\\u0439\": \"ru\",
            \"\\u0411\\u0435\\u043b\\u0430\\u0440\\u0443\\u0441\\u043a\\u0430\\u044f\": \"be\",
            \"\\u0423\\u043a\\u0440\\u0430\\u0457\\u043d\\u0441\\u044c\\u043a\\u0430\": \"uk\",
            \"\\u0421\\u0440\\u043f\\u0441\\u043a\\u0438 (\\u045b\\u0438\\u0440\\u0438\\u043b\\u0438\\u0446\\u0430)\": \"rs_cyrillic\",
            \"\\u0411\\u044a\\u043b\\u0433\\u0430\\u0440\\u0441\\u043a\\u0438\": \"bg\",
            \"\\u041c\\u043e\\u043d\\u0433\\u043e\\u043b\": \"mn\",
            \"\\u0410\\u0431\\u0430\\u0437\\u0430\": \"abq\",
            \"\\u0410\\u0434\\u044b\\u0433\\u0430\\u0431\\u0437\\u044d\": \"ady\",
            \"\\u041a\\u044a\\u044f\\u0431\\u0436\\u044d\\u0440\\u0434\\u0437\\u044d\\u0439\": \"kbd\",
            \"\\u0410\\u0432\\u0430\\u0440\": \"ava\",
            \"\\u0414\\u0430\\u0440\\u0433\\u0432\\u0430\": \"dar\",
            \"\\u0413\\u0401\\u0430\\u043b\\u0433\\u04c0\\u0430\\u0439\": \"inh\",
            \"\\u041d\\u043e\\u0445\\u0447\\u0438\\u0439\\u043d\": \"che\",
            \"\\u041b\\u0430\\u043a\\u043a\\u0443\": \"lbe\",
            \"\\u041b\\u0435\\u0437\\u0433\\u0438\": \"lez\",
            \"\\u0422\\u0430\\u0431\\u0430\\u0441\\u0430\\u0440\\u0430\\u043d\": \"tab\",
            \"\\u049a\\u0430\\u0437\\u0430\\u049b\": \"kk\",
            \"\\u041a\\u044b\\u0440\\u0433\\u044b\\u0437\": \"ky\",
            \"\\u0422\\u043e\\u0497\\u0438\\u043a\\u04e3\": \"tg\",
            \"\\u041c\\u0430\\u043a\\u0435\\u0434\\u043e\\u043d\\u0441\\u043a\\u0438\": \"mk\",
            \"\\u0422\\u0430\\u0442\\u0430\\u0440\": \"tt\",
            \"\\u0427\\u04d1\\u0432\\u0430\\u0448\": \"cv\",
            \"\\u0411\\u0430\\u0448\\u04a1\\u043e\\u0440\\u0442\": \"ba\",
            \"\\u041e\\u043b\\u044b\\u043c\\u0430\\u0440\\u0438\\u0439\": \"mhr\",
            \"\\u041c\\u043e\\u043b\\u0434\\u043e\\u0432\\u0435\\u043d\\u044f\\u0441\\u043a\\u044d\": \"mo\",
            \"\\u0423\\u0434\\u043c\\u0443\\u0440\\u0442\": \"udm\",
            \"\\u041a\\u043e\\u043c\\u0438\": \"kv\",
            \"\\u0418\\u0440\\u043e\\u043d\": \"os\",
            \"\\u0411\\u0443\\u0440\\u044f\\u0430\\u0434\": \"bua\",
            \"\\u0425\\u0430\\u043b\\u044c\\u043c\\u0433\": \"xal\",
            \"\\u0422\\u044b\\u0432\\u0430\": \"tyv\",
            \"\\u0421\\u0430\\u0445\\u0430\": \"sah\",
            \"\\u041a\\u0430\\u0440\\u0430\\u043a\\u0430\\u043b\\u043f\\u0430\\u049b\": \"kaa\",
            # Devanagari
            \"\\u0939\\u093f\\u0928\\u094d\\u0926\\u0940\": \"hi\",
            \"\\u092e\\u0930\\u093e\\u0920\\u0940\": \"mr\",
            \"\\u0928\\u0947\\u092a\\u093e\\u0932\\u0940\": \"ne\",
            \"\\u092d\\u094b\\u091c\\u092a\\u0941\\u0930\\u0940\": \"bh\",
            \"\\u092e\\u0948\\u0925\\u093f\\u0932\\u0940\": \"mai\",
            \"\\u0905\\u0902\\u0917\\u093f\\u0915\\u093e\": \"ang\",
            \"\\u092d\\u094b\\u091c\\u092a\\u0941\\u0930\\u0940\": \"bho\",
            \"\\u092e\\u093e\\u0932\\u094d\\u0935\\u093e\": \"mah\",
            \"\\u0938\\u0926\\u094d\\u0926\\u093e\": \"sck\",
            \"\\u0928\\u0947\\u0935\\u093e\\u0930\\u0940\": \"new\",
            \"\\u0917\\u094b\\u092e\\u093f\\u0924\": \"gom\",
            \"\\u0938\\u0902\\u0938\\u094d\\u0915\\u0943\\u0924\": \"sa\",
            \"\\u092c\\u093e\\u0917\\u0947\\u0932\\u0940\": \"bgc\",
            # Other
            \"\\ud55c\\uad6d\\uc5b4\": \"korean\",
            \"\\u0e20\\u0e32\\u0e29\\u0e32\\u0e44\\u0e17\\u0e22\": \"th\",
            \"\\u0395\\u03bb\\u03bb\\u03b7\\u03bd\\u03b9\\u03ba\\u03ac\": \"el\",
            \"\\u0c24\\u0c46\\u0c32\\u0c41\\u0c17\\u0c41\": \"te\",
            \"\\u0ba4\\u0bae\\u0bbf\\u0bb4\\u0bcd\": \"ta\","""

# Now find it with actual raw text matching
# The file has actual Unicode characters, not escape sequences
with open(UI_PATH, 'r', encoding='utf-8') as f:
    raw = f.read()

# Find the PP-OCR Full block using the actual characters
import re
# Find the larger block from "# PP-OCR Full languages" to the next section
start_marker = "# PP-OCR Full languages"
end_marker = "# EasyOCR languages"

start_idx = raw.find(start_marker)
end_idx = raw.find(end_marker, start_idx)

if start_idx >= 0 and end_idx > start_idx:
    old = raw[start_idx:end_idx]
    # Remove trailing whitespace at end
    old_stripped = old.rstrip('\n').rstrip()
    
    new = """            # PP-OCR Full languages
            \"English (\\u62c9\\u4e01)\": \"en\",
            \"Fran\\u00e7ais (\\u62c9\\u4e01)\": \"fr\",
            \"Espa\\u00f1ol (\\u62c9\\u4e01)\": \"es\",
            \"Italiano (\\u62c9\\u4e01)\": \"it\",
            \"Deutsch (\\u62c9\\u4e01)\": \"de\",
            \"Portugu\\u00eas (\\u62c9\\u4e01)\": \"pt\",
            \"\\u7e41\\u9ad4\\u4e2d\\u6587\": \"chinese_cht\",
            \"\\u62c9\\u4e01 (\\u81ea\\u52a8)\": \"latin\",
            \"\\u963f\\u62c9\\u4f2f (\\u81ea\\u52a8)\": \"ar\",
            \"\\u897f\\u91cc\\u5c14 (\\u81ea\\u52a8)\": \"ru\",
            \"\\u5929\\u57ce\\u6587 (\\u81ea\\u52a8)\": \"hi\","""
    
    raw = raw[:start_idx] + new + raw[end_idx:]
    with open(UI_PATH, 'w', encoding='utf-8') as f:
        f.write(raw)
    print(f"OK: Replaced PP-OCR Full mapping block ({len(old)} chars -> concise)")
else:
    print(f"ERROR: markers not found: start={start_idx}, end={end_idx}")
    # Show region around start_marker
    if start_idx >= 0:
        print(f"Snippet: {repr(raw[start_idx:start_idx+200])}")
