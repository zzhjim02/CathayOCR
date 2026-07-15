# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'D:\UMIBETA\UmiOCR-data\site-packages')

import fitz

path = r'D:\OCR待AI处理\明实录.12.明光宗实录.pdf'
try:
    doc = fitz.open(path)
    print(f'Pages: {len(doc)}')
    page = doc[0]
    pix = page.get_pixmap(dpi=72)
    print(f'Page 0: {pix.width}x{pix.height}')
    text = page.get_text()
    print(f'Page 0 text len: {len(text)}')
    print(f'First 100: {repr(text[:100])}')
    doc.close()
    print('OK')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
