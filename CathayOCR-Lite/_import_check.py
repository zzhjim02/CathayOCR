import sys
ok = True
mods = ['PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui',
        'json', 'subprocess', 'os', 'shutil', 'tempfile', 're', 'glob',
        'threading', 'queue', 'base64', 'platform', 'time', 'math', 'ctypes', 'numpy']
for m in mods:
    try:
        __import__(m)
        print('  OK', m)
    except ImportError:
        ok = False
        print('  MISS', m)
if ok:
    print('ALL OK')
else:
    sys.exit(1)
